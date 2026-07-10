# OCR Service — Invoice Scanning sidecar

A FastAPI service that turns an uploaded invoice (PDF or image) into a
**structured invoice** (supplier & buyer, line items, taxes, totals) — fully
offline, no data leaves the host. It does two things:

- **Extraction** — reading-order text + per-token boxes (`POST /extract`).
- **Structuring** — a geometry-first, rules-only parser converts that into the
  `ExtractedInvoice` the NestJS backend consumes (`POST /parse`). This logic used
  to live in TypeScript; it now lives here (`app/structuring/`) so it can read
  token **geometry** directly, which is what makes party-name / line-item
  detection robust across invoice layouts. Node only calls `/parse` and matches
  the result against the DB.

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

### OCR engines & tiers

`OCR_ENGINE` picks the engine for images / scanned PDFs (digital PDFs always use
the exact text-layer path, so this only affects photographed/scanned inputs):

- **`classic`** (default) — PP-OCRv5 detection + recognition. Two model tiers via
  `OCR_MODEL_TIER`:
  - `fast` (default) — **PP-OCRv5 mobile** + 200 DPI. Stable and quick (~13–19s/page).
  - `accurate` — **PP-OCRv5 server** + 300 DPI. Better recognition, but measured
    **>2 min/page on aarch64/CPU** — prefer it on x86 / GPU / many-core hosts.
- **`vl`** — **PaddleOCR-VL**, a local ~0.9B vision-language model (needs the
  `paddlex[ocr]` extra; weights ~1.8 GB). Best on hard scans, but see the ARM note.

> ### ⚠️ ARM / aarch64 note
> On `aarch64` (paddlepaddle 3.2.2) three things crash and are avoided:
> - PaddleOCR 3.x's default **PP-OCRv6** models **segfault** → we pin **PP-OCRv5**.
> - **Multi-threaded** CPU inference segfaults → `cpu_threads=1`.
> - **`OCR_ENGINE=vl` (PaddleOCR-VL) segfaults on inference** on this ARM/CPU box.
>   A SIGSEGV crashes the worker and cannot be caught, so **do not enable `vl` on
>   ARM** — run it on an x86 / GPU host. The engine is wired (with a fallback for
>   non-crash errors) for exactly those environments.
>
> On x86 you may raise `OCR_CPU_THREADS` to 2–4, set `OCR_MODEL_TIER=accurate`,
> or `OCR_ENGINE=vl`.

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
| `OCR_ENGINE` | `classic` | `classic` (PP-OCRv5) or `vl` (PaddleOCR-VL — x86/GPU only, see ARM note) |
| `OCR_MODEL_TIER` | `fast` | `fast` (mobile, 200 DPI) or `accurate` (server, 300 DPI) |
| `OCR_USE_GPU` | `false` | Use the CUDA PaddlePaddle build |
| `OCR_LANG` | `en` | PaddleOCR language pack |
| `OCR_DET_MODEL` | tier default | Override the text-detection model |
| `OCR_REC_MODEL` | tier default | Override the text-recognition model |
| `OCR_CPU_THREADS` | `1` | Paddle CPU threads (>1 segfaults on ARM) |
| `OCR_DPI` | tier default | Rasterisation DPI for scanned PDFs |
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

### `POST /parse` (multipart, field `file`) — the production endpoint

Extraction **+ structuring**: returns the `ExtractedInvoice` the backend consumes.

```json
{
  "method": "pdf-text",
  "structuringMethod": "rules",
  "pageCount": 1,
  "durationMs": 96,
  "text": "TAX INVOICE\n...",
  "invoice": {
    "schemaVersion": 1,
    "seller": { "name": "VIJAY SALES", "gstin": "24AAHCV3778L1ZQ", "stateName": "Gujarat", "pan": "AAHCV3778L", "address": "..." },
    "buyer":  { "name": "Jayhind", "gstin": "24AJGPP6816J1ZY", "...": "..." },
    "invoice": { "number": "SS/2026/0412", "date": "2026-07-05" },
    "lineItems": [ { "description": "...", "hsnSac": "8471", "quantity": 100, "rate": 250, "taxableAmount": 25000, "gstRate": 18, "cgstAmount": 2250, "sgstAmount": 2250, "igstAmount": null, "confidence": 1.0 } ],
    "taxSummary": [ { "rate": 18, "taxableAmount": 56200, "cgst": 5058, "sgst": 5058, "igst": 0 } ],
    "totals": { "taxableTotal": 56200, "taxTotal": 10116, "roundOff": 0, "grandTotal": 66316, "amountInWords": null },
    "fieldConfidence": { "seller.gstin": 1.0, "seller.name": 0.9, "totals.grandTotal": 1.0 }
  }
}
```

`fieldConfidence` (0..1 per field) drives the review screen's "check this"
highlights. Accuracy of the structurer is scored by
`tests/accuracy_report.py` against `tests/fixtures/layout_golden.json`
(see `tests/ACCURACY_BASELINE.md`).

## Tests

```bash
source .venv/bin/activate
pip install pytest httpx
python tests/make_fixtures.py     # regenerate the sample invoices
pytest
```
