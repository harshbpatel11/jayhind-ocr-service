"""Confidence scoring.

Produces the ``fieldConfidence`` map (JSON path → 0..1) that drives the ERP's
amber highlights, plus one ``overallConfidence`` headline (≈50 % header fields,
50 % line items) the Master Hub uses to route weak scans to the review queue.

Scores are evidence-based, not model self-reports: a GSTIN scores high only if it
is *structurally valid*; a grand total scores high only if the invoice actually
*foots*; a line scores high only if ``qty × rate`` and ``taxable × rate`` are
internally consistent. That makes a low score mean "check this", reliably.
"""

from __future__ import annotations

from app.config import Settings
from app.domain.interfaces import ConfidenceScorer
from app.domain.models import ExtractedInvoice, LineItem
from app.domain.pipeline_types import ExtractionContext
from app.rules import gst
from app.utils.numeric import round2

# Header fields that make up the header half of the overall score.
_HEADER_WEIGHTS = {
    "seller.name": 1.0,
    "seller.gstin": 1.0,
    "buyer.name": 1.0,
    "buyer.gstin": 0.6,
    "invoice.number": 1.0,
    "invoice.date": 0.8,
    "totals.taxableTotal": 1.0,
    "totals.grandTotal": 1.2,
}


class ConfidenceScorerImpl(ConfidenceScorer):
    """Evidence-based per-field + overall confidence."""

    def __init__(self, settings: Settings) -> None:
        self._tol = settings.amount_tolerance

    def score(self, invoice: ExtractedInvoice, context: ExtractionContext) -> ExtractedInvoice:
        field_conf: dict[str, float] = {}
        hint_amounts = set(round2(a) for a in context.hints.amounts)

        # -- header ----------------------------------------------------------
        field_conf["seller.name"] = _name_score(invoice.seller.name)
        field_conf["buyer.name"] = _name_score(invoice.buyer.name)
        field_conf["seller.gstin"] = _gstin_score(invoice.seller.gstin)
        field_conf["buyer.gstin"] = _gstin_score(invoice.buyer.gstin)
        field_conf["invoice.number"] = _present_score(
            invoice.invoice.number, grounded=invoice.invoice.number == (context.hints.invoice_number or "")
        )
        field_conf["invoice.date"] = _present_score(
            invoice.invoice.date, grounded=invoice.invoice.date == context.hints.invoice_date
        )

        foots = self._foots(invoice)
        field_conf["totals.grandTotal"] = _total_score(
            invoice.totals.grand_total, foots, grounded=round2(invoice.totals.grand_total) in hint_amounts
        )
        field_conf["totals.taxableTotal"] = _total_score(
            invoice.totals.taxable_total, self._taxable_matches_lines(invoice)
        )

        # -- line items ------------------------------------------------------
        line_scores: list[float] = []
        for i, item in enumerate(invoice.line_items):
            score = _line_score(item, self._tol)
            item.confidence = score
            field_conf[f"lineItems[{i}]"] = score
            line_scores.append(score)

        # -- overall (≈50 % header, 50 % lines) ------------------------------
        header_score = _weighted_mean(
            {k: field_conf[k] for k in _HEADER_WEIGHTS}, _HEADER_WEIGHTS
        )
        if line_scores:
            lines_score = sum(line_scores) / len(line_scores)
            overall = 0.5 * header_score + 0.5 * lines_score
        else:
            # No lines recovered — cap confidence; the document is barely usable.
            overall = 0.4 * header_score

        invoice.field_confidence = {k: round(v, 3) for k, v in field_conf.items()}
        invoice.overall_confidence = round(min(max(overall, 0.0), 1.0), 3)
        return invoice

    # -- cross-checks ---------------------------------------------------------
    def _foots(self, invoice: ExtractedInvoice) -> bool:
        t = invoice.totals
        return abs(round2(t.taxable_total + t.tax_total + t.round_off) - t.grand_total) <= self._tol

    def _taxable_matches_lines(self, invoice: ExtractedInvoice) -> bool:
        if not invoice.line_items:
            return False
        line_sum = round2(sum(item.taxable_amount for item in invoice.line_items))
        return abs(line_sum - invoice.totals.taxable_total) <= max(self._tol, invoice.totals.discount_total)


# -- field scorers ------------------------------------------------------------
def _name_score(name: str) -> float:
    length = len((name or "").strip())
    if length >= 3:
        return 0.9
    return 0.35 if length else 0.2


def _gstin_score(gstin: str | None) -> float:
    if not gstin:
        return 0.5  # legitimately absent for a B2C party
    return 0.97 if gst.is_valid_gstin(gstin) else 0.5


def _present_score(value: object, grounded: bool = False) -> float:
    present = bool(str(value or "").strip())
    if not present:
        return 0.3
    return 0.95 if grounded else 0.85


def _total_score(value: float, consistent: bool, grounded: bool = False) -> float:
    if not value:
        return 0.2
    if consistent and grounded:
        return 0.98
    if consistent:
        return 0.9
    return 0.55


def _line_score(item: LineItem, tol: float) -> float:
    score = 1.0
    if len(item.description.strip()) < 3:
        score -= 0.35
    if item.taxable_amount <= 0:
        score -= 0.3
    if item.quantity <= 0:
        score -= 0.1
    if item.rate <= 0:
        score -= 0.1
    # Internal consistency: qty × rate (− discount) ≈ taxable.
    gross = round2(item.quantity * item.rate)
    expected = round2(gross - (item.discount or 0.0))
    if item.taxable_amount > 0 and abs(expected - item.taxable_amount) > max(tol, 0.02 * item.taxable_amount):
        score -= 0.15
    # Tax consistency: taxable × rate ≈ tax charged.
    if item.gst_rate:
        tax = (item.cgst_amount or 0) + (item.sgst_amount or 0) + (item.igst_amount or 0)
        expected_tax = round2(item.taxable_amount * item.gst_rate / 100.0)
        if abs(tax - expected_tax) > max(tol, 0.02 * expected_tax):
            score -= 0.1
    return round(max(score, 0.05), 3)


def _weighted_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    return sum(scores[k] * weights[k] for k in weights) / total_w
