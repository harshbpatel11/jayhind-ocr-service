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
prints a `PUBLIC_URL` + `API_KEY`. **Leave it running.** First start downloads a
few GB of model weights.

**Two tunnel modes:**

| Mode | How | URL |
|---|---|---|
| **Quick** (default) | nothing to set | new random `https://xxxxx.trycloudflare.com` each run |
| **Named** (recommended) | set `CF_TUNNEL_TOKEN` | a **fixed** hostname you own, e.g. `https://ocr.aakhaja.com` — never changes |

```
TUNNEL     = cloudflare (named — stable URL)
PUBLIC_URL = https://ocr.aakhaja.com
API_KEY    = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Stable URL — one-time setup

A named tunnel needs a domain on your Cloudflare account (Free plan is fine):

1. **Zero Trust → Networks → Tunnels → Create a tunnel** → connector **Cloudflared**
   → name it → copy the **token** (the `eyJ...` string after `--token`). Don't run
   the install commands it shows — `run.sh` runs `cloudflared` for you.
2. **Route tunnel → Public hostname:** subdomain `ocr`, your domain, service
   `HTTP` → `localhost:8000`. Save (this auto-creates the DNS).
3. Run with two env vars set (both are **secrets** — never commit them):
   `CF_TUNNEL_TOKEN` (from step 1) and a fixed `OCR_API_KEY` (any long random
   string, so the key is stable too). Now the URL **and** the key stay fixed —
   set the hub `.env` once and never touch it again on a restart.

> **The ~100s edge cap applies to both modes.** A Cloudflare tunnel drops any single
> request longer than ~100s with **HTTP 524** (free/pro plans); the hub then reports
> "OCR engine is unreachable". A named tunnel fixes the URL, **not** this cap — a
> parse that routinely exceeds ~100s needs a faster engine.

### Kaggle

1. New Notebook → settings: **Accelerator = GPU T4 x2**, **Internet = On**
   (Internet needs a phone-verified account).
2. In a cell (set the two secrets for a stable URL; omit `CF_TUNNEL_TOKEN` for a
   quick tunnel):
   ```python
   import os
   os.environ['CF_TUNNEL_TOKEN'] = '<your named-tunnel token>'
   os.environ['OCR_API_KEY']     = '<a long random key you reuse>'
   !git clone https://github.com/harshbpatel11/jayhind-ocr-service.git
   %cd jayhind-ocr-service
   !bash run.sh
   ```
3. Copy the `PUBLIC_URL` + `API_KEY` it prints. Keep the tab open.

### Google Colab

Runtime → Change runtime type → **GPU**, then the same lines as above. The
`notebook.ipynb` in this repo runs the same steps cell by cell and reads the two
secrets from Colab's **Secrets** panel (🔑) so they never get saved into the
notebook — add `CF_TUNNEL_TOKEN` and `OCR_API_KEY` there.

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

> With a **quick** tunnel the URL changes every run — re-paste it into `.env` and
> `dev restart admin-back` each time. With a **named** tunnel + a fixed
> `OCR_API_KEY`, both values stay constant: set `.env` once and skip this on every
> future restart.

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
| `CF_TUNNEL_TOKEN` | — | **Secret.** Set it to run a named Cloudflare tunnel (stable URL). Unset ⇒ quick tunnel |
| `CF_TUNNEL_HOSTNAME` | `ocr.aakhaja.com` | The named tunnel's public hostname (for the printed `PUBLIC_URL`) |
| `EXTRACTOR_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | The extraction LLM |
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
