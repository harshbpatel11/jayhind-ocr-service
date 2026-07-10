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

## OCR engine notes (this aarch64 / CPU box)

- **classic + fast (mobile) = default**: stable, ~13-19s/page.
- **classic + accurate (server models, 300 DPI)**: opt-in (`OCR_MODEL_TIER=accurate`).
  Measured **>2 min/page** here and heavy on RAM — prefer on x86/GPU/many-core.
- **PaddleOCR-VL (`OCR_ENGINE=vl`)**: installable (`paddlex[ocr]`, weights ~1.8 GB)
  and wired with auto-fallback, but it **SEGFAULTS on inference on this ARM/CPU
  host** (same class as PP-OCRv6). A SIGSEGV cannot be caught in-process, so do NOT
  enable it here — it needs an x86 / GPU host. Left in place for that environment.

