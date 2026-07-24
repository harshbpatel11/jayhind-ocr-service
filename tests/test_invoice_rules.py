"""Unit tests for business-rule validation (the contract backbone)."""

from __future__ import annotations

from app.config import Settings
from app.domain.pipeline_types import ExtractionContext, ReaderOutput, RuleHints
from app.rules.invoice_rules import InvoiceValidatorImpl


def _ctx() -> ExtractionContext:
    reader = ReaderOutput(pages=[], markdown="", text="", engine="fake")
    return ExtractionContext(reader=reader, hints=RuleHints(), method="ocr", page_count=1)


def _validator() -> InvoiceValidatorImpl:
    return InvoiceValidatorImpl(Settings())


BASE = {
    "seller": {"name": "ACME STEELS PRIVATE LIMITED", "gstin": "24AJGPP6816J1ZY"},
    "buyer": {"name": "JAYHIND ENTERPRISES", "gstin": "24AABCU9603R1ZX"},
    "invoice": {"number": "INV-2026-0042", "date": "03/04/2026"},
    "lineItems": [
        {"description": "Steel Coil", "hsnSac": "7208", "quantity": 10, "rate": 5000, "taxableAmount": 50000, "gstRate": 18},
        {"description": "Steel Rod", "hsnSac": "7214", "quantity": 5, "rate": 2000, "taxableAmount": 10000, "gstRate": 18},
    ],
    "totals": {"taxableTotal": 60000, "taxTotal": 10800, "grandTotal": 70800},
}


def test_intrastate_splits_cgst_sgst():
    inv = _validator().validate(BASE, _ctx())
    line = inv.line_items[0]
    assert line.cgst_amount == 4500
    assert line.sgst_amount == 4500
    assert line.igst_amount is None
    # GSTIN-derived fields are authoritative, not model-supplied.
    assert inv.seller.state_name == "Gujarat"
    assert inv.seller.pan == "AJGPP6816J"
    assert inv.invoice.date == "2026-04-03"


def test_interstate_uses_igst():
    payload = {**BASE, "buyer": {"name": "X", "gstin": "27AABCU9603R1ZX"}}
    inv = _validator().validate(payload, _ctx())
    line = inv.line_items[0]
    assert line.igst_amount == 9000
    assert line.cgst_amount is None and line.sgst_amount is None


def test_totals_foot_and_tax_summary_derived():
    inv = _validator().validate(BASE, _ctx())
    t = inv.totals
    assert t.taxable_total == 60000
    assert t.tax_total == 10800
    assert round(t.taxable_total + t.tax_total + t.round_off, 2) == t.grand_total
    assert len(inv.tax_summary) == 1
    slab = inv.tax_summary[0]
    assert slab.rate == 18 and slab.taxable_amount == 60000
    assert slab.cgst == 5400 and slab.sgst == 5400


def test_missing_grand_total_is_computed():
    payload = {**BASE, "totals": {"taxableTotal": 60000, "taxTotal": 10800}}
    inv = _validator().validate(payload, _ctx())
    assert inv.totals.grand_total == 70800


def test_small_mismatch_absorbed_into_roundoff():
    payload = {**BASE, "totals": {"taxableTotal": 60000, "taxTotal": 10800, "grandTotal": 70803}}
    inv = _validator().validate(payload, _ctx())
    assert inv.totals.round_off == 3.0
    assert round(inv.totals.taxable_total + inv.totals.tax_total + inv.totals.round_off, 2) == 70803


def test_taxable_computed_from_qty_rate_when_absent():
    payload = {
        **BASE,
        "lineItems": [{"description": "Widget", "quantity": 3, "rate": 100, "discount": 50, "gstRate": 18}],
        "totals": {},
    }
    inv = _validator().validate(payload, _ctx())
    assert inv.line_items[0].taxable_amount == 250  # 3*100 - 50


def test_hsn_backfilled_from_ocr_text_when_llm_drops_it():
    # The LLM read qty/rate/gst but left hsnSac null (a real failure mode).
    payload = {
        **BASE,
        "lineItems": [
            {"description": "Logitech MX Master 3S Mouse", "quantity": 100, "rate": 250, "taxableAmount": 25000, "gstRate": 18},
            {"description": "Dell WM126 Wireless Mouse", "quantity": 40, "rate": 780, "taxableAmount": 31200, "gstRate": 18},
        ],
    }
    reader = ReaderOutput(
        pages=[],
        markdown="",
        text=(
            "Description\nHSN\nQty\nRate\nTaxable\nGST\n"
            "Logitech MX Master 3S Mouse\n8471\n100\n250.00\n25,000.00\n18%\n"
            "Dell WM126 Wireless Mouse\n8471\n40\n780.00\n31,200.00\n18%\n"
        ),
        engine="fake",
    )
    ctx = ExtractionContext(reader=reader, hints=RuleHints(), method="ocr", page_count=1)
    inv = _validator().validate(payload, ctx)
    assert inv.line_items[0].hsn_sac == "8471"
    assert inv.line_items[1].hsn_sac == "8471"


def test_hsn_backfill_never_overrides_or_grabs_money():
    payload = {
        **BASE,
        "lineItems": [
            {"description": "Item A", "quantity": 2, "rate": 5000, "taxableAmount": 10000, "gstRate": 18, "hsnSac": "9999"},
            {"description": "Item B", "quantity": 1, "rate": 5000, "taxableAmount": 5000, "gstRate": 18},
        ],
    }
    # Item B has NO HSN on the page; its rate 5000 must NOT be grabbed as an HSN.
    reader = ReaderOutput(
        pages=[], markdown="",
        text="Item A\n8471\n2\n5000.00\nItem B\n5000.00\n1\n",
        engine="fake",
    )
    ctx = ExtractionContext(reader=reader, hints=RuleHints(), method="ocr", page_count=1)
    inv = _validator().validate(payload, ctx)
    assert inv.line_items[0].hsn_sac == "9999"  # model value kept, not overridden
    assert inv.line_items[1].hsn_sac is None      # 5000 (the rate) not grabbed


def test_empty_payload_still_contract_shaped():
    inv = _validator().validate({}, _ctx())
    assert inv.schema_version == 1
    assert inv.line_items == []
    assert inv.totals.grand_total == 0.0


def test_due_date_derived_from_invoice_date_and_terms_days():
    # "Payment Due: 30 Days" (paymentTermsDays only) → dueDate = invoice date + 30.
    payload = {**BASE, "invoice": {"number": "INV-2026-0042", "date": "03/04/2026", "paymentTermsDays": 30}}
    inv = _validator().validate(payload, _ctx())
    assert inv.invoice.payment_terms_days == 30
    assert inv.invoice.due_date == "2026-05-03"


def test_explicit_due_date_wins_over_derivation():
    payload = {
        **BASE,
        "invoice": {"number": "INV-2026-0042", "date": "03/04/2026", "dueDate": "2026-06-01", "paymentTermsDays": 30},
    }
    inv = _validator().validate(payload, _ctx())
    assert inv.invoice.due_date == "2026-06-01"


def test_due_date_falls_back_to_hints_when_model_leaves_it_blank():
    reader = ReaderOutput(pages=[], markdown="", text="", engine="fake")
    hints = RuleHints(due_date="2026-05-20", payment_terms_days=15)
    ctx = ExtractionContext(reader=reader, hints=hints, method="ocr", page_count=1)
    inv = _validator().validate(BASE, ctx)
    assert inv.invoice.due_date == "2026-05-20"
    assert inv.invoice.payment_terms_days == 15


def test_no_due_date_without_terms_or_invoice_date():
    payload = {**BASE, "invoice": {"number": "INV-2026-0042", "date": None}}
    inv = _validator().validate(payload, _ctx())
    assert inv.invoice.due_date is None
    assert inv.invoice.payment_terms_days is None
