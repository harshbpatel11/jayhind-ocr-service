"""LLM invoice-OCR service.

One FastAPI endpoint the Jayhind ERP hub calls:

    POST /parse   multipart field `file` (image or PDF)  -> ParseResponse JSON
    GET  /health

Two stages, both on the GPU:
  1. PaddleOCR-VL reads the document into markdown/text (tables preserved).
  2. An instruction LLM (default Qwen2.5-7B-Instruct) maps that into the exact
     `ExtractedInvoice` contract, then a deterministic pass guarantees the shape
     (derives GSTIN state/PAN, reconciles totals, fills defaults).

Runs anywhere with a GPU (Kaggle / Colab / RunPod / a bare CUDA box). No local
RapidOCR / geometry structuring — the LLM does the reading and the structuring.

Auth: every /parse request must carry `Authorization: Bearer <API_KEY>`. The key
is read from OCR_API_KEY (set it to keep the same key across restarts) or a random
one is generated at startup and printed by launch.py.
"""
import glob
import io
import json
import os
import re
import secrets
import tempfile
import time

# PaddlePaddle otherwise pre-allocates most of the GPU the moment it loads, which
# starves the Torch extractor model on a shared single-GPU box (Colab/Kaggle) and
# makes bitsandbytes spill to the CPU ("Some modules are dispatched on the CPU").
# Make paddle allocate on demand and cap its share so both models coexist. Must be
# set BEFORE paddle is imported (it reads these FLAGS at import).
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.35")

import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

# ─────────────────────────── configuration ──────────────────────────────────
API_KEY = os.getenv("OCR_API_KEY", "").strip() or secrets.token_urlsafe(32)
MAX_BYTES = int(os.getenv("OCR_MAX_UPLOAD_MB", "25")) * 1024 * 1024
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
PDF_DPI = int(os.getenv("OCR_PDF_DPI", "200"))

# GST state codes (first two digits of a GSTIN).
GST_STATE = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", "07": "Delhi",
    "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
    "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh",
    "24": "Gujarat", "25": "Daman and Diu",
    "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra",
    "28": "Andhra Pradesh", "29": "Karnataka", "30": "Goa", "31": "Lakshadweep",
    "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman and Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh",
    "38": "Ladakh", "97": "Other Territory", "99": "Centre Jurisdiction",
}

# ─────────────────────────── model loading ──────────────────────────────────
# Loaded once, at import. The first run downloads several GB.
print("[server] loading PaddleOCR-VL ...", flush=True)
from paddleocr import PaddleOCRVL  # noqa: E402

ocr_pipeline = PaddleOCRVL()

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def _load_extractor():
    """Load the 4-bit extractor LLM, falling back to smaller Qwen models if the
    preferred one does not fit alongside PaddleOCR-VL on a shared GPU — so the
    service still comes up instead of crashing at import."""
    candidates = [EXTRACTOR_MODEL]
    for smaller in ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
        if smaller not in candidates:
            candidates.append(smaller)
    last_err = None
    for name in candidates:
        try:
            print(f"[server] loading extractor LLM: {name} (4-bit) ...", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=_bnb, device_map="auto"
            )
            return tok, mdl
        except Exception as err:  # OOM / CPU-dispatch / download issue
            last_err = err
            print(f"[server] {name} failed to load ({type(err).__name__}: {err}); "
                  "trying a smaller model ...", flush=True)
    raise RuntimeError(f"No extractor model could be loaded: {last_err}")


tokenizer, llm = _load_extractor()
print("[server] models ready.", flush=True)


# ─────────────────────── stage 1: file -> images -> markdown ─────────────────
class UnsupportedType(Exception):
    """The upload is neither a readable image nor a PDF (terminal, 4xx)."""


def file_to_images(data: bytes, content_type: str, filename: str, out_dir: str):
    name = (filename or "").lower()
    ct = (content_type or "").lower()
    is_pdf = "pdf" in ct or name.endswith(".pdf")
    paths = []
    if is_pdf:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=PDF_DPI)
            p = os.path.join(out_dir, f"page_{i}.png")
            pix.save(p)
            paths.append(p)
        doc.close()
    else:
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise UnsupportedType(f"Not a readable image or PDF: {exc}")
        p = os.path.join(out_dir, "page_0.png")
        img.save(p, "PNG")
        paths.append(p)
    return paths


def _result_to_markdown(res) -> str:
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return md.get("markdown_texts") or md.get("markdown") or ""
    if isinstance(md, str):
        return md
    try:
        with tempfile.TemporaryDirectory() as d:
            res.save_to_markdown(save_path=d)
            files = glob.glob(os.path.join(d, "**", "*.md"), recursive=True)
            if files:
                with open(files[0], encoding="utf-8") as f:
                    return f.read()
    except Exception:
        pass
    j = getattr(res, "json", None)
    return json.dumps(j, ensure_ascii=False) if j else ""


def ocr_to_markdown(image_paths) -> str:
    parts = []
    for p in image_paths:
        for res in ocr_pipeline.predict(p):
            parts.append(_result_to_markdown(res))
    return "\n\n".join(x for x in parts if x).strip()


# ─────────────────────── stage 2: markdown -> invoice JSON ───────────────────
SCHEMA = r"""
Return ONE JSON object with EXACTLY these keys (no extra keys, no comments):
{
 "seller": {"name","address","gstin","phone","email","pincode"},
 "buyer":  {"name","address","gstin","phone","email","pincode"},
 "invoice": {"number","date"},
 "lineItems": [{"description","hsnSac","quantity","unit","rate","discount",
                "taxableAmount","gstRate","cgstAmount","sgstAmount","igstAmount","lineTotal"}],
 "taxSummary": [{"rate","taxableAmount","cgst","sgst","igst"}],
 "totals": {"subTotal","discountTotal","taxableTotal","taxTotal","roundOff","grandTotal","amountInWords"}
}
Rules:
- seller = the SUPPLIER / issuer (From / Sold By). buyer = the recipient (Bill To / Buyer).
- Money = plain numbers (2 decimals). date = "YYYY-MM-DD" or null. Unknown = null.
- Intra-state (both GSTIN state codes equal) -> fill cgstAmount+sgstAmount, set igstAmount null.
  Inter-state -> fill igstAmount, set cgstAmount+sgstAmount null.
- taxableAmount = taxable value AFTER any discount. gstRate = total GST percent (e.g. 18).
- taxableTotal is AFTER discount; grandTotal = taxableTotal + taxTotal + roundOff.
- Never invent values you cannot read. Output JSON only.
"""


def _parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1:
        s = s[a:b + 1]
    return json.loads(s)


def llm_extract(markdown_text: str) -> dict:
    messages = [
        {"role": "system", "content": "You are an expert at reading Indian GST tax invoices and returning structured JSON."},
        {"role": "user", "content": SCHEMA + "\n\nINVOICE (markdown/text):\n" + markdown_text[:12000]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(llm.device)
    out = llm.generate(**inputs, max_new_tokens=2048, do_sample=False)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_json(generated)


# ─────────────────── deterministic contract post-processing ──────────────────
def _num(x, default=None):
    try:
        if x in (None, ""):
            return default
        return round(float(x), 2)
    except Exception:
        return default


def _party(p: dict) -> dict:
    p = p or {}
    g = ((p.get("gstin") or "").strip().upper()) or None
    if g and len(g) == 15:
        state_code, pan, state_name = g[:2], g[2:12], GST_STATE.get(g[:2])
    else:
        state_code, pan, state_name = None, None, None
    phone = re.sub(r"\D", "", str(p.get("phone") or "")) or None
    return {
        "name": (p.get("name") or "").strip(),
        "address": (p.get("address") or "").strip(),
        "gstin": g, "stateCode": state_code, "stateName": state_name, "pan": pan,
        "phone": phone, "email": (p.get("email") or None), "pincode": (p.get("pincode") or None),
    }


def _line(it: dict) -> dict:
    it = it or {}
    return {
        "description": (it.get("description") or "").strip(),
        "hsnSac": (str(it.get("hsnSac")).strip() if it.get("hsnSac") else None),
        "quantity": _num(it.get("quantity"), 0) or 0,
        "unit": (it.get("unit") or None),
        "rate": _num(it.get("rate"), 0) or 0,
        "discount": _num(it.get("discount")),
        "taxableAmount": _num(it.get("taxableAmount"), 0) or 0,
        "gstRate": _num(it.get("gstRate")),
        "cgstAmount": _num(it.get("cgstAmount")),
        "sgstAmount": _num(it.get("sgstAmount")),
        "igstAmount": _num(it.get("igstAmount")),
        "lineTotal": _num(it.get("lineTotal")),
        "confidence": 0.9,
    }


def postprocess(raw: dict) -> dict:
    seller, buyer = _party(raw.get("seller")), _party(raw.get("buyer"))
    lines = [_line(x) for x in (raw.get("lineItems") or [])]
    t = raw.get("totals") or {}
    totals = {
        "subTotal": _num(t.get("subTotal"), 0) or 0,
        "discountTotal": _num(t.get("discountTotal"), 0) or 0,
        "taxableTotal": _num(t.get("taxableTotal"), 0) or 0,
        "taxTotal": _num(t.get("taxTotal"), 0) or 0,
        "roundOff": _num(t.get("roundOff"), 0) or 0,
        "grandTotal": _num(t.get("grandTotal"), 0) or 0,
        "amountInWords": (t.get("amountInWords") or None),
    }
    tax = [
        {
            "rate": _num(s.get("rate"), 0) or 0,
            "taxableAmount": _num(s.get("taxableAmount"), 0) or 0,
            "cgst": _num(s.get("cgst"), 0) or 0,
            "sgst": _num(s.get("sgst"), 0) or 0,
            "igst": _num(s.get("igst"), 0) or 0,
        }
        for s in (raw.get("taxSummary") or [])
    ]
    inv = raw.get("invoice") or {}
    foots = abs((totals["taxableTotal"] + totals["taxTotal"] + totals["roundOff"]) - totals["grandTotal"]) <= 1
    have = bool(seller["name"] and buyer["name"] and lines and totals["grandTotal"])
    overall = 0.9 if (foots and have) else (0.6 if have else 0.4)
    return {
        "schemaVersion": 1, "seller": seller, "buyer": buyer,
        "invoice": {"number": (inv.get("number") or "").strip(), "date": (inv.get("date") or None)},
        "lineItems": lines, "taxSummary": tax, "totals": totals,
        "fieldConfidence": {}, "overallConfidence": overall,
    }


# ─────────────────────────────── FastAPI ────────────────────────────────────
app = FastAPI(title="Jayhind LLM Invoice-OCR")


def _auth(authorization: str):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "paddleocr-vl", "gpu": torch.cuda.is_available()}


@app.post("/parse")
async def parse(file: UploadFile = File(...), authorization: str = Header(default="")):
    """4xx = the document is unreadable (terminal); 5xx = retryable engine error.
    The hub relies on this split, so never flatten one into the other."""
    _auth(authorization)
    started = time.monotonic()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Document is larger than {MAX_BYTES // (1024 * 1024)} MB")

    with tempfile.TemporaryDirectory() as d:
        try:
            images = file_to_images(data, file.content_type, file.filename, d)
        except UnsupportedType as exc:
            raise HTTPException(status_code=415, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read document: {exc}")
        try:
            markdown = ocr_to_markdown(images)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"OCR engine error: {exc}")
        if not markdown:
            raise HTTPException(status_code=400, detail="Document appears blank or unreadable")
        try:
            invoice = postprocess(llm_extract(markdown))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Extraction error: {exc}")
        page_count = len(images)

    return {
        "method": "ocr", "structuringMethod": "rules",
        "pageCount": page_count, "durationMs": int((time.monotonic() - started) * 1000),
        "text": markdown, "invoice": invoice,
    }
