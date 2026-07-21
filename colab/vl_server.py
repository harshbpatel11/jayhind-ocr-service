"""PaddleOCR-VL OCR server for Google Colab (x86 + free GPU).

Runs the vision-language OCR engine that SEGFAULTS on the ARM production box, and
exposes the single endpoint the local jayhind-ocr-service calls when it is started
with OCR_ENGINE=remote:

    POST /ocr-page   body: raw image bytes (image/png)   -> {width,height,tokens,text}

Tokens are axis-aligned boxes in the coordinate space of the image actually
inferred on, so the local geometry-first structuring (app/structuring/) needs no
changes. The `_tokens_from_result` mapping mirrors app/extractor.py:
`_vl_tokens_from_result` — keep them in sync.

Protect the public (Cloudflare-tunnelled) URL with a shared key: set OCR_REMOTE_KEY
here and the SAME value in the local service's .env; every request must carry it
in the `x-ocr-key` header.
"""
import io
import os

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from PIL import Image

REMOTE_KEY = os.getenv("OCR_REMOTE_KEY", "").strip()
# On the GPU host we can afford a higher working resolution than the ARM box's
# 2600 cap — sharper glyphs help recognition. (Real detail is still bounded by the
# DPI the caller rasterised at; set OCR_DPI=300 on the local box for best input.)
MAX_SIDE = int(os.getenv("OCR_MAX_IMAGE_SIDE", "3200"))


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# BEST-QUALITY mode: unlike the ARM box (where these pre-models add crash surface
# and CPU cost), the T4 runs them cheaply and they materially help real-world
# photos/scans — deskew/rotate correction (orientation) and page flattening
# (unwarping). Turn off via OCR_DOC_ORIENTATION=0 / OCR_DOC_UNWARP=0 for speed.
USE_DOC_ORIENTATION = _flag("OCR_DOC_ORIENTATION", True)
USE_DOC_UNWARP = _flag("OCR_DOC_UNWARP", True)

app = FastAPI(title="PaddleOCR-VL remote engine", version="1.0.0")

_engine = None


def _get_engine():
    """Lazily build the PaddleOCR-VL pipeline (weights download on first call)."""
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCRVL

        _engine = PaddleOCRVL(
            use_doc_orientation_classify=USE_DOC_ORIENTATION,
            use_doc_unwarping=USE_DOC_UNWARP,
        )
    return _engine


def _fit(image):
    """Cap the longest side so a full-res phone photo can't exhaust the worker."""
    longest = max(image.size)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def _tokens_from_result(res):
    """Map a PaddleOCR-VL result into flat tokens (bbox + text + score).

    Mirrors app/extractor.py:_vl_tokens_from_result so the local side receives the
    exact shape it expects. Defensive: unknown result shapes yield ([], text) and
    the local side falls back to the plain-text field.
    """
    data = res.json if hasattr(res, "json") else res
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    blocks = []
    if isinstance(data, dict):
        for key in ("parsing_res_list", "layout_parsing_result", "blocks", "boxes"):
            if isinstance(data.get(key), list):
                blocks = data[key]
                break

    tokens = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bbox = block.get("block_bbox") or block.get("bbox") or block.get("layout_bbox")
        text = block.get("block_content") or block.get("content") or block.get("text") or ""
        if not (bbox and str(text).strip()):
            continue
        try:
            x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (TypeError, ValueError, IndexError):
            continue
        lines = [ln for ln in str(text).splitlines() if ln.strip()] or [str(text)]
        step = (y1 - y0) / max(1, len(lines))
        for i, line in enumerate(lines):
            tokens.append({
                "text": line.strip(),
                "bbox": [x0, y0 + i * step, x1, y0 + (i + 1) * step],
                "confidence": float(block.get("score", 0.9) or 0.9),
            })

    if tokens:
        return tokens, ""
    md = getattr(res, "markdown", None)
    text = md.get("markdown_texts") if isinstance(md, dict) else (md or "")
    return [], str(text or "").strip()


@app.get("/health")
def health():
    return {"status": "ok", "engine": "paddleocr-vl", "model_loaded": _engine is not None}


@app.post("/warmup")
def warmup():
    """Build the pipeline ahead of the first real request (model download/load)."""
    _get_engine()
    return {"status": "ready"}


@app.post("/ocr-page")
async def ocr_page(request: Request, x_ocr_key: str = Header(default="")):
    if REMOTE_KEY and x_ocr_key != REMOTE_KEY:
        raise HTTPException(status_code=401, detail="bad or missing x-ocr-key")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")

    image = _fit(Image.open(io.BytesIO(data)).convert("RGB"))
    results = _get_engine().predict(input=np.array(image))
    if not results:
        return {"width": float(image.width), "height": float(image.height), "tokens": [], "text": ""}
    tokens, text = _tokens_from_result(results[0])
    return {"width": float(image.width), "height": float(image.height), "tokens": tokens, "text": text}
