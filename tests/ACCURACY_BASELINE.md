# Invoice structuring — accuracy baseline

Scored by `tests/accuracy_report.py` against `fixtures/layout_golden.json`
(12 layout families reproduced from real user sample invoices). Six field checks
per fixture — `seller`, `buyer`, `inv#`, `lines`, `taxable`, `grand` — plus
`seller_gstin` / `buyer_gstin` / `invoice_date` wherever the golden pins them
→ **83 checks** total. The name check rejects polluted names (a trailing
"Date 10-07-2026" fails; the old substring check let it pass).

## BEFORE — old TypeScript text-line parser (`invoice-parsing.const.ts`)

**18 / 66 (27%)** — captured 2026-07-10 on the current engine.

```
layout_3col.pdf        seller:✗ buyer:✗ inv#:✗ lines:✗ taxable:✗ grand:✗   buyer="Acme Pumps Pvt Ltd INV2208 Metro"
layout_card.pdf        seller:✓ buyer:✓ inv#:✗ lines:✗ taxable:✗ grand:✗
layout_fromto.pdf      seller:✗ buyer:✗ inv#:✗ lines:✗ taxable:✗ grand:✗   both names empty
layout_banner.pdf      seller:✗ buyer:✗ inv#:✓ lines:✗ taxable:✗ grand:✗   seller="TOP BANNER" buyer="Ship To"
layout_sidebar.pdf     seller:✓ buyer:✓ inv#:✗ lines:✗ taxable:✗ grand:✗   marker prefix kept in name
layout_timeline.pdf    seller:✗ buyer:✓ inv#:✗ lines:✗ taxable:✗ grand:✗   seller="Quotation → PO → …"
layout_sidebyside.pdf  seller:✗ buyer:✗ inv#:✓ lines:✗ taxable:✗ grand:✗   seller="NEWSPAPER" buyer="STYLE" (title bleed)
layout_htmlleak.pdf    seller:✗ buyer:✗ inv#:✓ lines:✓ taxable:✗ grand:✓   seller="Header Block" buyer="Layout"
layout_wellformed.pdf  seller:✗ buyer:✓ inv#:✓ lines:✓ taxable:✗ grand:✓   seller="WHOLESALE" (title bleed)
layout_pos.pdf         seller:✗ buyer:✗ inv#:✗ lines:✗ taxable:✓ grand:✗   seller="Receipt #45821" buyer="SUPER"
layout_corporate.pdf   seller:✓ buyer:✓ inv#:✗ lines:✗ taxable:✓ grand:✗   inv#/lines lost
```

Failure themes: (1) document **titles bleed into party names**; (2) **line-item
tables unrecognised** on terse headers (`Item`/`Grand`/`Taxable`/`Amt`); (3)
marker prefixes (`Supplier:`) kept in the name; (4) three-column / From-To /
banner layouts collapse under the text-line model.

## AFTER — geometry-first Python engine

**66 / 66 (100%)** — 2026-07-10. Every layout family reads the correct supplier,
buyer, invoice number, line count, taxable and grand total. Regenerate with
`.venv/bin/python tests/accuracy_report.py`.

The win comes from reading token **geometry** (2-D party blocks, column-band
tables) instead of flattened text lines, plus broadened column/label lexicons and
title-bleed handling. It applies to digital PDFs (the exact text-layer path — no
OCR model involved) and OCR output alike.

## 2026-07-13 — letterhead family + stricter scoring: **83 / 83 (100%)**

A 12th family (`layout_letterhead.pdf`, modelled on the user's `invoice-new-1`
sample: unlabelled full-width company letterhead over a bordered info grid and a
lone Bill To / Ship To box) exposed that the seller came back **completely
empty** — the letterhead ABOVE a party marker was never read. Fixed alongside a
hardening sweep; the goldens now also score GSTINs and the invoice date, and the
name check rejects pollution. All checks pass on the digital path, the clean
OCR path, and an 8°-tilt + blur + noise scan (~4s on this box):

- **Seller letterhead above a lone "Bill To"** — new geometry fallback grows a
  column window from the first value segment down to the first document-meta
  label / party marker. A lone line with no GSTIN/address/contact under it is
  rejected (a banner strip must not become a supplier).
- **Document-meta lexicon** (`Order No.`, `Due Date`, `Payment Terms`, e-way,
  IRN, vehicle…) ends letterheads and stays out of names/addresses; party meta
  (GSTIN/PAN/phone/email) stays in.
- **Merged OCR titles** — "TAXINVOICE" (RapidOCR drops spaces) is now
  recognised as a title, not a seller name.
- **Grid-row pollution** — "XYZ Retail LLP Date 10-07-2026" trims to the name;
  the text-strategy stacked reader no longer swallows the next grid row (which
  handed the supplier's GSTIN to the buyer).
- **Stated tax columns** — per-line CGST/SGST/IGST and Total columns are
  claimed explicitly (on the OCR path an unclaimed column's tokens glued onto
  the neighbouring column, turning "18%"+"270.00" into gstRate 18270). Stated
  amounts win over recomputation; missing GST rates are derived from stated
  taxes or the line total and snapped to real slabs; a "CGST Rate" column read
  as the GST rate is doubled back.
- **Multi-slab totals** — "CGST @9% … / CGST @14% …" rows sum per-rate
  (deduped); a plain "CGST <amt>" total row wins.
- **Comma-as-dot amounts** — a degraded scan reads "6,372.0" as "6.372.0";
  the trailing-amount reader now keeps that as one amount (was: grand total 0).
- **Wrapped info-grid cells** — invoice no/date fall back to table cells
  ("Invoice\nDate" / "12-\nJul-2026" heal to 2026-07-12).
- Node matching (`invoice-matching.const.ts`): space-less OCR names
  ("ABCTradersPvt.Ltd.", "WirelessKeyboard") now match — glued legal suffixes
  peel off and character bigrams carry single-token names past the candidate
  threshold.

## OCR engine benchmark (this aarch64 / CPU box)

Only images / scanned PDFs use an OCR engine — digital PDFs take the exact
text-layer path. Fields = 5 checks on `purchase_digital` (both GSTINs, invoice no,
line count, grand total).

| Input | `classic` (paddle PP-OCRv5 mobile) | **`onnx` (RapidOCR) — default** |
|---|---|---|
| clean_scan | 20.6s · 5/5 | **2.6s · 5/5** |
| low_quality_photo | ~20s · 5/5 | **4.1s · 5/5** |
| 8° tilt + blur + noise | ~20s · **4/5** | **2.7s · 5/5** |

**≈8× faster and strictly more accurate on degraded scans** → adopted as the
default engine. `classic` remains as the fallback (`OCR_ENGINE=classic`).

### Image preprocessing (`OCR_PREPROCESS=1`, default)
Grayscale → deskew → CLAHE contrast → edge-preserving denoise. Measured on paddle:
an 8° tilted photo went **3/5 → 4/5**, with **no regression** on the clean scan
(5/5 → 5/5). The deskew sign is load-bearing: getting it wrong *doubles* the tilt
(regression-tested in `tests/test_preprocess.py`).

### Engines that do NOT work here
- **Engines are mutually exclusive per process**: onnxruntime + paddlepaddle
  **segfault when loaded together** on ARM. `extractor.py` therefore imports only
  the configured engine (`find_spec`, never an import) and `onnx` never
  auto-falls-back to `classic` in-process.
- **`OCR_MODEL_TIER=accurate`** (PP-OCRv5 server, 300 DPI): >2 min/page, heavy RAM.
- **PaddleOCR-VL (`OCR_ENGINE=vl`)**: installs and downloads (~1.8 GB) but
  **SEGFAULTS on inference on aarch64/CPU** (same class as PP-OCRv6). A SIGSEGV
  cannot be caught in-process — x86 / GPU hosts only.

