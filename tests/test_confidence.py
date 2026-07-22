"""Unit tests for evidence-based confidence scoring."""

from __future__ import annotations

from app.confidence.scorer import ConfidenceScorerImpl
from app.config import Settings
from app.domain.pipeline_types import ExtractionContext, ReaderOutput, RuleHints
from app.rules.invoice_rules import InvoiceValidatorImpl
from tests.test_invoice_rules import BASE


def _score(payload: dict, hints: RuleHints | None = None):
    ctx = ExtractionContext(
        reader=ReaderOutput(pages=[], markdown="", text="", engine="fake"),
        hints=hints or RuleHints(),
        method="ocr",
        page_count=1,
    )
    invoice = InvoiceValidatorImpl(Settings()).validate(payload, ctx)
    return ConfidenceScorerImpl(Settings()).score(invoice, ctx)


def test_clean_invoice_scores_high():
    hints = RuleHints(amounts=[70800.0], invoice_number="INV-2026-0042", invoice_date="2026-04-03")
    inv = _score(BASE, hints)
    assert inv.overall_confidence >= 0.8
    assert inv.field_confidence["seller.gstin"] > 0.9
    assert inv.field_confidence["totals.grandTotal"] > 0.9
    # every line got a per-line score
    assert all(item.confidence > 0 for item in inv.line_items)


def test_non_footing_invoice_scores_lower():
    payload = {**BASE, "totals": {"taxableTotal": 60000, "taxTotal": 10800, "grandTotal": 99999}}
    inv = _score(payload)
    assert inv.field_confidence["totals.grandTotal"] < 0.7


def test_missing_lines_caps_confidence():
    payload = {**BASE, "lineItems": []}
    inv = _score(payload)
    assert inv.overall_confidence < 0.5


def test_invalid_gstin_scored_down():
    payload = {**BASE, "seller": {"name": "X", "gstin": "NOTAGSTIN0000X"}}
    inv = _score(payload)
    assert inv.field_confidence["seller.gstin"] <= 0.5
