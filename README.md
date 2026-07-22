# Jayhind Invoice-OCR

A **local, CPU-only, accuracy-first** invoice OCR + extraction service for the
Jayhind ERP. It reads an invoice (PDF, scan, phone photo, PNG/JPG/TIFF) and
returns a structured GST-invoice JSON (`ExtractedInvoice`, `schemaVersion 1`) —
the exact contract the ERP's invoice-scanning flow consumes for **purchase
invoice entry**.

Runs entirely on the box, no GPU and no network: designed for an ARM64 Ubuntu
24.04 server (Oracle Cloud Neoverse N1, 4 vCPU / 24 GB). No Tesseract, no
EasyOCR, no Colab, no Cloudflare tunnel, no torch.

## Pipeline

```
 upload (PDF / image)
   │
   ├─ ingest          PyMuPDF (PDF raster + digital-PDF text layer) / Pillow (PNG,JPG,TIFF,photo)
   ├─ preprocess      OpenCV: deskew · denoise · contrast (CLAHE) · adaptive threshold · DPI upscale
   ├─ read            RapidOCR (PP-OCR models via ONNX Runtime, CPU) → reading-order text
   ├─ layout          typed blocks + parsed tables (merged cells, reading order)
   ├─ rules           deterministic hints: GSTINs, invoice no/date, HSN, the line-item grid
   ├─ extract         Qwen3-8B-Instruct (local, CPU, llama.cpp, JSON-grammar-constrained)
   ├─ validate        GST business rules: GSTIN→state/PAN, intra/inter split, totals reconcile
   └─ score           per-field + overall confidence → ExtractedInvoice JSON
```

Every stage is behind a `typing.Protocol` and wired by one composition root
(`app/container.py`), so the reader and the extractor are swappable (the tests
run the whole pipeline with lightweight fakes — no model downloads).

> **Reader note (ARM64):** the default reader is **RapidOCR** — PaddleOCR's
> PP-OCR detection+recognition models run through **ONNX Runtime**, which is
> CPU-stable on this Neoverse N1 box. The **PaddleOCR-VL 1.6** and classic
> PaddleOCR readers are implemented too (`OCR_READER_ENGINE=paddleocr-vl` /
> `paddleocr`), but PaddlePaddle's *native* CPU inference **segfaults** on this
> aarch64 build for both — use them only on a host where paddle inference works.

## Install (ARM64 / x86, CPU)

```bash
bash scripts/install.sh          # system build tools + venv + deps (RapidOCR/ONNX + llama.cpp)
bash scripts/download_models.sh  # Qwen3-8B GGUF (~4.7 GB); RapidOCR fetches its ONNX models on first use
bash scripts/serve.sh            # serve on 127.0.0.1:8100
```

(Add the optional PaddleOCR-VL / classic readers with `INSTALL_PADDLE=1 bash
scripts/install.sh` — only useful where paddle's CPU inference works.)

Or under the workspace tmux runner: `dev start ocr` (see `dev.sh`).

## API

`GET /health` → `{"status":"ok","engine":"rapidocr","extractor":"qwen3-8b-instruct","reader_ready":true,"extractor_ready":true,"version":"2.0.0","gpu":false}`

`POST /parse` — `multipart/form-data`, field `file`. Returns **plain JSON**
(the loopback hub proxy reads `response.json()` directly):

```jsonc
{
  "method": "ocr",              // or "pdf-text" for a digital PDF
  "structuringMethod": "rules",
  "pageCount": 1,
  "durationMs": 84210,
  "text": "…reading-order text…",
  "invoice": {
    "schemaVersion": 1,
    "seller": { "name","address","gstin","stateCode","stateName","pan","phone","email","pincode" },
    "buyer":  { "…same…" },
    "invoice": { "number","date" },
    "lineItems": [ { "description","hsnSac","quantity","unit","rate","discount",
                     "taxableAmount","gstRate","cgstAmount","sgstAmount","igstAmount","lineTotal","confidence" } ],
    "taxSummary": [ { "rate","taxableAmount","cgst","sgst","igst" } ],
    "totals": { "subTotal","discountTotal","taxableTotal","taxTotal","roundOff","grandTotal","amountInWords" },
    "fieldConfidence": { "seller.gstin": 0.97, "totals.grandTotal": 0.98 },
    "overallConfidence": 0.9
  }
}
```

`POST /extract` — raw reader passthrough (`text` + `markdown`) for QA/debugging.

**Error semantics (the hub depends on these): `4xx` = document unreadable
(terminal), `5xx` = retryable engine error.**

Auth is **off** for the loopback deployment; set `OCR_API_KEY` to require
`Authorization: Bearer <key>` (must match the hub's `OCR_SERVICE_KEY`).

## Performance — accuracy over speed

The stated targets (1 pg < 4 s, 20 pg < 60 s, 50 pg < 2 min) are met by the
**OCR + layout + rules** stages. The **Qwen3-8B extractor on a 4-vCPU CPU** runs
at only a few tokens/second, so a single invoice's extraction takes **tens of
seconds to minutes** — a deliberate trade-off: this service is tuned for maximum
extraction accuracy, not latency (that is why the hub/child OCR timeouts are 10+
minutes). To trade accuracy for speed, point `OCR_LLM_MODEL_PATH` at a smaller
GGUF (Qwen3-4B / 1.7B) or set `OCR_EXTRACTOR_ENGINE=rules` (deterministic,
model-free).

```bash
python scripts/benchmark.py --file tests/fixtures/sample_invoice.pdf
python scripts/loadtest.py  --file tests/fixtures/sample_invoice.pdf --requests 20 --concurrency 4
```

## Configuration

All env vars are `OCR_`-prefixed with production-safe defaults — see
[.env.example](.env.example) and [app/config.py](app/config.py).

## Tests & quality

```bash
.venv/bin/python -m pytest      # 82 unit + integration tests (no models needed)
.venv/bin/ruff check app tests  # PEP 8
```

Type hints throughout, async I/O, repository pattern + dependency injection,
reusable single-purpose modules.

## Project layout

| Path | What |
|---|---|
| `app/domain/` | wire contract (`models.py`), pipeline types, stage `Protocol`s |
| `app/ingestion/` | upload → page images (PDF/image/TIFF, text-layer fast path) |
| `app/preprocessing/` | OpenCV deskew/denoise/threshold/contrast/resize |
| `app/ocr/` | RapidOCR reader (default) + PaddleOCR-VL / classic readers + null reader |
| `app/layout/` | markdown → typed blocks + parsed tables |
| `app/rules/` | GST rules, dates, pre-LLM hints, business-rule validation |
| `app/extraction/` | Qwen3-8B (llama.cpp) + rules fallback + prompt/JSON |
| `app/confidence/` | per-field + overall confidence |
| `app/pipeline/` | orchestrator + typed errors |
| `app/container.py` | composition root (DI) |
| `app/api/` + `app/main.py` | FastAPI routes + app factory |
| `scripts/` | install / download / serve / benchmark / loadtest |
| `tests/` | pytest suite + `fixtures/sample_invoice.pdf` |
