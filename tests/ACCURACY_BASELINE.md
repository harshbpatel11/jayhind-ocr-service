# Invoice structuring — accuracy baseline

Scored by `tests/accuracy_report.py` against `fixtures/layout_golden.json`
(11 layout families reproduced from real user sample invoices). Six field checks
per fixture: `seller`, `buyer`, `inv#`, `lines`, `taxable`, `grand` → 66 total.

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

