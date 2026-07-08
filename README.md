# OCR Service — Invoice Scanning sidecar

A small FastAPI service that turns an uploaded invoice (PDF or image) into
**reading-order text plus per-token bounding boxes and confidences**. It is the
only place PaddleOCR lives; the NestJS backend talks to it over HTTP so the
extraction engine can be swapped later (e.g. for a vision LLM) without touching
the rest of the pipeline.

See `INVOICE_SCANNING_PLAN.md` §3.1 at the project root.

## Two extraction paths (chosen automatically)

| Input | Path | Engine |
|---|---|---|
| PDF **with** an embedded text layer | `pdf-text` | `pdfplumber` — words + boxes, **no OCR** |
| PDF **without** a text layer (scan) | `ocr` | `pymupdf` rasterises pages → PaddleOCR |
| Image (JPG/PNG/WEBP) | `ocr` | PaddleOCR |

PaddleOCR runs with `use_textline_orientation=True` so rotated/upside-down scan
lines are handled. Tokens are sorted into reading order (top-to-bottom, then
left-to-right, with a line-clustering tolerance) before the text is joined.

Doc-orientation and doc-unwarping pre-models are **disabled**: invoices are
already page-shaped, those models cost seconds per page, and they add crash
surface on ARM. Line-level rotation is still corrected.

> ### ⚠️ ARM / aarch64 note
> PaddleOCR 3.x defaults to the **PP-OCRv6** models, which **segfault on ARM**
> (verified on `aarch64`, paddlepaddle 3.2.2). Multi-threaded CPU inference
> segfaults too. This service therefore defaults to **PP-OCRv5 mobile** with
> **`cpu_threads=1`**, which is stable on both ARM and x86 and preserves word
> spacing (PP-OCRv4 runs words together). On x86 you may raise `OCR_CPU_THREADS`
> to 2–4 for a straight speed win, and can switch to the larger `_server_`
> models via `OCR_DET_MODEL` / `OCR_REC_MODEL`.

## Setup (CPU — recommended)

```bash
cd ocr-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # ~2 GB with PaddlePaddle; first run downloads models
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Model weights (~10 MB detection + ~10 MB recognition + angle classifier) are
downloaded to `~/.paddleocr/` on the first OCR request, not at install time.

**Light install (digital PDFs only)** — skip the two Paddle lines in
`requirements.txt`. The service starts fine; `/health` reports
`"ocr_available": false` and image/scanned-PDF requests return a clear 503
instead of crashing.

### GPU

Replace `paddlepaddle` with the CUDA build matching your toolkit, then set
`OCR_USE_GPU=true`:

```bash
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==3.0.0   # see paddlepaddle.org.cn for the CUDA-specific wheel
OCR_USE_GPU=true uvicorn app.main:app --port 8100
```

CPU is sufficient for tens–hundreds of invoices/day (~1.5–2 GB RAM, roughly
1–4 s per page). Reach for GPU only at thousands/day.

### Docker

```bash
docker build -t trendy-ocr-service .
docker run -p 8100:8100 -v ocr-models:/root/.paddleocr trendy-ocr-service
```

Mount `/root/.paddleocr` so model weights survive container restarts.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OCR_USE_GPU` | `false` | Use the CUDA PaddlePaddle build |
| `OCR_LANG` | `en` | PaddleOCR language pack |
| `OCR_DET_MODEL` | `PP-OCRv5_mobile_det` | Text-detection model (see ARM note) |
| `OCR_REC_MODEL` | `PP-OCRv5_mobile_rec` | Text-recognition model |
| `OCR_CPU_THREADS` | `1` | Paddle CPU threads (>1 segfaults on ARM) |
| `OCR_DPI` | `200` | Rasterisation DPI for scanned PDFs |
| `OCR_MAX_PAGES` | `10` | Hard cap on pages processed per document |
| `OCR_TEXT_LAYER_MIN_CHARS` | `120` | Chars/page needed to treat a PDF as digital |

The backend reaches the service at `OCR_SERVICE_URL` (default
`http://localhost:8100`).

## API

### `GET /health`

```json
{ "status": "ok", "ocr_available": true, "gpu": false }
```

### `POST /extract` (multipart, field `file`)

```json
{
  "method": "pdf-text",
  "pageCount": 1,
  "durationMs": 84,
  "text": "TAX INVOICE\nSeller Pvt Ltd\nGSTIN 24AAACT2727Q1ZW\n...",
  "pages": [{
    "index": 0,
    "width": 612.0,
    "height": 792.0,
    "text": "TAX INVOICE\n...",
    "tokens": [
      { "text": "TAX", "bbox": [72.0, 60.1, 96.4, 72.3], "confidence": 1.0 }
    ],
    "tables": [ { "rows": [["Description","HSN","Qty"]] } ]
  }]
}
```

`confidence` is `1.0` for every token on the `pdf-text` path (the characters are
exact, not recognised). Errors return `{"detail": "..."}` with 400 (bad file),
415 (unsupported type) or 503 (OCR engine unavailable).

## Tests

```bash
source .venv/bin/activate
pip install pytest httpx
python tests/make_fixtures.py     # regenerate the sample invoices
pytest
```
