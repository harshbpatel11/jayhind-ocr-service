# Jayhind LLM Invoice-OCR

A small GPU service that turns an invoice (image or PDF) into a **structured GST
invoice JSON**. Runs anywhere with a GPU — **Kaggle, Google Colab, RunPod, or any
CUDA box** — and is called by the Jayhind ERP hub over one HTTP endpoint.

- **Reader:** PaddleOCR-VL (vision-language OCR — reads the document, keeps tables)
- **Extractor:** an instruction LLM (Qwen2.5-7B-Instruct by default) maps the text
  into the exact `ExtractedInvoice` contract; a deterministic pass then derives
  GSTIN state/PAN, reconciles totals and fills defaults.

No RapidOCR, no local geometry structuring — the LLM does the reading **and** the
structuring.

---

## Run it (one command)

On any machine with a GPU + internet:

```bash
git clone https://github.com/harshbpatel11/jayhind-ocr-service.git
cd jayhind-ocr-service
bash run.sh
```

`run.sh` installs everything and starts the server + a public HTTPS tunnel, then
prints:

```
TUNNEL     = ngrok
PUBLIC_URL = https://xxxxx.ngrok-free.app
API_KEY    = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Leave it running.** First start downloads a few GB of model weights.

> **Use ngrok, not a Cloudflare quick tunnel.** Set `NGROK_TOKEN` (free token at
> ngrok.com) before `run.sh`. A `trycloudflare.com` tunnel drops any request that
> runs longer than ~100s with **HTTP 524**, and this GPU pipeline routinely takes
> longer than that per document — so on Cloudflare the hub reports "OCR engine is
> unreachable". ngrok has no such response cap.

### Kaggle

1. New Notebook → settings: **Accelerator = GPU T4 x2**, **Internet = On**
   (Internet needs a phone-verified account).
2. In a cell:
   ```python
   import os
   os.environ['NGROK_TOKEN'] = '<your ngrok authtoken>'   # avoids the ~100s cap
   !git clone https://github.com/harshbpatel11/jayhind-ocr-service.git
   %cd jayhind-ocr-service
   !bash run.sh
   ```
3. Copy the `PUBLIC_URL` + `API_KEY` it prints. Keep the tab open.

### Google Colab

Runtime → Change runtime type → **GPU**, then the same lines as above (the
`notebook.ipynb` in this repo has a dedicated `NGROK_TOKEN` cell).

---

## Connect it to the ERP hub

In `jayhind-admin-back/.env`:

```
OCR_SERVICE_URL=<PUBLIC_URL>
OCR_SERVICE_KEY=<API_KEY>
```

then `dev restart admin-back`. The hub sends `Authorization: Bearer <API_KEY>` on
every call, and allows up to `OCR_SERVICE_TIMEOUT_MS` (default 300000) per parse.
To go back to a local engine, blank `OCR_SERVICE_KEY` and point `OCR_SERVICE_URL`
back at it.

> The tunnel URL changes every run — re-paste it into `.env` and
> `dev restart admin-back` each time. A reserved ngrok domain (paid) keeps the
> same URL across restarts.

---

## API

**`GET /health`** → `{"status":"ok","engine":"paddleocr-vl","gpu":true}`

**`POST /parse`** — `multipart/form-data`, field `file` (image or PDF), header
`Authorization: Bearer <API_KEY>`. Returns:

```jsonc
{
  "method": "ocr",
  "structuringMethod": "rules",
  "pageCount": 1,
  "durationMs": 8421,
  "text": "<full document text>",
  "invoice": {
    "schemaVersion": 1,
    "seller": { "name", "address", "gstin", "stateCode", "stateName", "pan", "phone", "email", "pincode" },
    "buyer":  { "...": "same shape" },
    "invoice": { "number", "date" },
    "lineItems": [ { "description","hsnSac","quantity","unit","rate","discount",
                     "taxableAmount","gstRate","cgstAmount","sgstAmount","igstAmount","lineTotal","confidence" } ],
    "taxSummary": [ { "rate","taxableAmount","cgst","sgst","igst" } ],
    "totals": { "subTotal","discountTotal","taxableTotal","taxTotal","roundOff","grandTotal","amountInWords" },
    "fieldConfidence": {},
    "overallConfidence": 0.9
  }
}
```

Error semantics (the hub depends on these): **4xx** = the document is unreadable
(terminal), **5xx** = retryable engine/infra error.

Quick test:

```bash
curl -H "Authorization: Bearer <API_KEY>" <PUBLIC_URL>/health
curl -H "Authorization: Bearer <API_KEY>" -F file=@invoice.jpg <PUBLIC_URL>/parse
```

---

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `OCR_API_KEY` | random each start | Fixed key so the hub `.env` need not change between restarts |
| `EXTRACTOR_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | The extraction LLM |
| `NGROK_TOKEN` | — | **Recommended.** Use ngrok (no ~100s response cap) instead of a Cloudflare quick tunnel |
| `PORT` | `8000` | Server port |
| `OCR_PDF_DPI` | `200` | PDF rasterisation DPI |
| `OCR_MAX_UPLOAD_MB` | `25` | Max upload size |
| `PADDLE_INDEX` | CUDA 12.6 wheel index | Change to match your CUDA (`run.sh`) |

## Files

| File | What |
|---|---|
| `server.py` | FastAPI app: `/health` + `/parse`, both model stages, post-processing |
| `launch.py` | Starts the server + tunnel, prints URL + key |
| `run.sh` | Install deps + launch (one command) |
| `requirements.txt` | Python deps (torch/CUDA assumed preinstalled) |

> ⚠️ Invoices are your tenants' customer data. A hosted GPU + public tunnel means
> that data leaves your box. The API key blocks outsiders; keep the URL private.
