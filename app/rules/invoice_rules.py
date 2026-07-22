"""Business-rule validation: raw LLM dict → a guaranteed-shape invoice.

This is the deterministic backbone that makes the service trustworthy regardless
of what the model returned. It:

  * coerces every messy value to the right type (money, quantity, percent);
  * derives GSTIN → state code / state name / PAN (never trusts the model for
    these — the GSTIN is authoritative);
  * applies the intra/inter-state GST rule (CGST+SGST vs IGST) and fills a missing
    tax split from the taxable value × rate;
  * reconciles totals so ``grandTotal = taxableTotal + taxTotal + roundOff`` holds
    to the paisa, absorbing tiny gaps into roundOff;
  * back-fills the invoice number/date and tax summary from the rule hints / line
    items when the model left them blank.

The output is always a complete, contract-shaped :class:`ExtractedInvoice`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.config import Settings
from app.domain.interfaces import InvoiceValidator
from app.domain.models import (
    ExtractedInvoice,
    InvoiceMeta,
    LineItem,
    PartyBlock,
    TaxSlab,
    Totals,
)
from app.domain.pipeline_types import ExtractionContext
from app.rules import gst
from app.rules.dates import normalize_date
from app.utils.numeric import digits_only, round2, to_float, to_money


def _pick(source: dict, *keys: str) -> Any:
    for key in keys:
        if isinstance(source, dict) and source.get(key) not in (None, ""):
            return source.get(key)
    return None


#: An HSN/SAC code is a standalone 4-, 6-, or 8-digit number (not part of a
#: decimal/thousands-grouped money value like ``25,000.00`` or ``250.00``).
_HSN_TOKEN_RE = re.compile(r"(?<![\d.,])(\d{8}|\d{6}|\d{4})(?![\d.,])")


class InvoiceValidatorImpl(InvoiceValidator):
    """Deterministic contract + GST enforcement over the raw model output."""

    def __init__(self, settings: Settings) -> None:
        self._tol = settings.amount_tolerance

    def validate(self, raw: dict, context: ExtractionContext) -> ExtractedInvoice:
        raw = raw or {}
        seller = self._party(raw.get("seller") or {})
        buyer = self._party(raw.get("buyer") or {})

        inter = gst.is_inter_state(seller.gstin, buyer.gstin)
        raw_lines = raw.get("lineItems") or raw.get("line_items") or []
        lines = [self._line(item, inter) for item in raw_lines if isinstance(item, dict)]

        # The LLM sometimes reads a line's qty/rate/GST but drops the HSN/SAC cell.
        # Recover it deterministically from the OCR text (never overriding a value
        # the model DID read), so the code the invoice printed is not lost.
        reader_text = context.reader.text if context and context.reader else ""
        _backfill_hsn(lines, reader_text)

        totals = self._totals(raw.get("totals") or {}, lines)
        tax_summary = self._tax_summary(raw.get("taxSummary") or raw.get("tax_summary") or [], lines)
        meta = self._invoice_meta(raw.get("invoice") or {}, context)

        return ExtractedInvoice(
            seller=seller,
            buyer=buyer,
            invoice=meta,
            line_items=lines,
            tax_summary=tax_summary,
            totals=totals,
        )

    # -- parties --------------------------------------------------------------
    def _party(self, data: dict) -> PartyBlock:
        gstin = gst.normalize_gstin(_pick(data, "gstin", "gstIn", "gst"))
        return PartyBlock(
            name=str(_pick(data, "name") or "").strip(),
            address=str(_pick(data, "address") or "").strip(),
            gstin=gstin,
            state_code=gst.state_code_of(gstin),
            state_name=gst.state_name_of(gstin),
            pan=gst.pan_of(gstin) or _clean_pan(_pick(data, "pan")),
            phone=digits_only(_pick(data, "phone", "mobile", "contact")),
            email=_clean_str(_pick(data, "email")),
            pincode=_clean_pincode(_pick(data, "pincode", "pin", "zip")),
        )

    # -- line items -----------------------------------------------------------
    def _line(self, data: dict, inter: bool | None) -> LineItem:
        quantity = to_money(_pick(data, "quantity", "qty"), 0.0)
        rate = to_money(_pick(data, "rate", "price", "unitPrice"), 0.0)
        discount = to_float(_pick(data, "discount"))
        taxable = to_float(_pick(data, "taxableAmount", "taxable_amount", "amount"))
        if taxable is None:
            gross = round2(quantity * rate)
            taxable = round2(gross - (discount or 0.0))
        gst_rate = gst.nearest_rate_slab(to_float(_pick(data, "gstRate", "gst_rate", "taxRate")))

        cgst = to_float(_pick(data, "cgstAmount", "cgst"))
        sgst = to_float(_pick(data, "sgstAmount", "sgst"))
        igst = to_float(_pick(data, "igstAmount", "igst"))
        cgst, sgst, igst = self._split_tax(taxable, gst_rate, inter, cgst, sgst, igst)

        tax_amount = sum(v for v in (cgst, sgst, igst) if v)
        line_total = to_float(_pick(data, "lineTotal", "line_total", "total"))
        if line_total is None:
            line_total = round2(taxable + tax_amount)

        return LineItem(
            description=str(_pick(data, "description", "particulars", "item") or "").strip(),
            hsn_sac=_clean_str(_pick(data, "hsnSac", "hsn_sac", "hsn", "sac")),
            quantity=quantity,
            unit=_clean_str(_pick(data, "unit", "uom")),
            rate=rate,
            discount=discount,
            taxable_amount=taxable,
            gst_rate=gst_rate,
            cgst_amount=cgst,
            sgst_amount=sgst,
            igst_amount=igst,
            line_total=line_total,
        )

    def _split_tax(
        self,
        taxable: float,
        gst_rate: float | None,
        inter: bool | None,
        cgst: float | None,
        sgst: float | None,
        igst: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Reconcile the CGST/SGST/IGST split with the supply type + rate."""
        has_any = any(v is not None for v in (cgst, sgst, igst))
        # Decide the regime: an explicit inter/intra wins; otherwise infer from
        # whichever columns the document actually populated.
        if inter is None and has_any:
            inter = igst is not None and (cgst is None and sgst is None)

        if gst_rate:
            expected = round2(taxable * gst_rate / 100.0)
            half = round2(expected / 2.0)
        else:
            expected = half = 0.0

        if inter is True:
            igst = igst if igst is not None else (expected or None)
            return None, None, igst
        if inter is False:
            cgst = cgst if cgst is not None else (half or None)
            sgst = sgst if sgst is not None else (half or None)
            return cgst, sgst, None
        # Unknown supply type and no columns read: leave everything as-is.
        return cgst, sgst, igst

    # -- totals ---------------------------------------------------------------
    def _totals(self, data: dict, lines: list[LineItem]) -> Totals:
        line_taxable = round2(sum(item.taxable_amount for item in lines))
        line_tax = round2(
            sum((item.cgst_amount or 0) + (item.sgst_amount or 0) + (item.igst_amount or 0) for item in lines)
        )
        line_gross = round2(sum(round2(item.quantity * item.rate) for item in lines))

        taxable_total = to_float(_pick(data, "taxableTotal", "taxable_total"))
        taxable_total = taxable_total if taxable_total is not None else line_taxable
        tax_total = to_float(_pick(data, "taxTotal", "tax_total"))
        tax_total = tax_total if tax_total is not None else line_tax
        discount_total = to_money(_pick(data, "discountTotal", "discount_total"), 0.0)
        sub_total = to_float(_pick(data, "subTotal", "sub_total"))
        sub_total = sub_total if sub_total is not None else round2(max(line_gross, taxable_total + discount_total))
        round_off = to_money(_pick(data, "roundOff", "round_off", "rounding"), 0.0)

        grand_total = to_float(_pick(data, "grandTotal", "grand_total", "netPayable"))
        computed = round2(taxable_total + tax_total + round_off)
        if grand_total is None:
            grand_total = computed
        elif abs(grand_total - computed) > self._tol:
            # Absorb a small mismatch into roundOff so the invoice foots; a large
            # gap is left visible for the confidence scorer to flag.
            gap = round2(grand_total - round2(taxable_total + tax_total))
            if abs(gap) <= max(self._tol * 5, 5.0):
                round_off = gap

        return Totals(
            sub_total=sub_total,
            discount_total=discount_total,
            taxable_total=taxable_total,
            tax_total=tax_total,
            round_off=round_off,
            grand_total=grand_total,
            amount_in_words=_clean_str(_pick(data, "amountInWords", "amount_in_words", "inWords")),
        )

    # -- tax summary ----------------------------------------------------------
    def _tax_summary(self, data: list, lines: list[LineItem]) -> list[TaxSlab]:
        if data:
            slabs = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                slabs.append(
                    TaxSlab(
                        rate=to_money(_pick(row, "rate", "gstRate"), 0.0),
                        taxable_amount=to_money(_pick(row, "taxableAmount", "taxable_amount"), 0.0),
                        cgst=to_money(_pick(row, "cgst"), 0.0),
                        sgst=to_money(_pick(row, "sgst"), 0.0),
                        igst=to_money(_pick(row, "igst"), 0.0),
                    )
                )
            if slabs:
                return slabs
        # Derive from line items grouped by GST rate.
        grouped: dict[float, dict[str, float]] = defaultdict(lambda: {"taxable": 0.0, "cgst": 0.0, "sgst": 0.0, "igst": 0.0})
        for item in lines:
            key = item.gst_rate or 0.0
            bucket = grouped[key]
            bucket["taxable"] += item.taxable_amount
            bucket["cgst"] += item.cgst_amount or 0.0
            bucket["sgst"] += item.sgst_amount or 0.0
            bucket["igst"] += item.igst_amount or 0.0
        return [
            TaxSlab(
                rate=rate,
                taxable_amount=round2(vals["taxable"]),
                cgst=round2(vals["cgst"]),
                sgst=round2(vals["sgst"]),
                igst=round2(vals["igst"]),
            )
            for rate, vals in sorted(grouped.items())
        ]

    # -- invoice meta ---------------------------------------------------------
    def _invoice_meta(self, data: dict, context: ExtractionContext) -> InvoiceMeta:
        number = _clean_str(_pick(data, "number", "invoiceNumber", "no")) or context.hints.invoice_number or ""
        date_value = normalize_date(_pick(data, "date", "invoiceDate")) or context.hints.invoice_date
        return InvoiceMeta(number=str(number).strip(), date=date_value)


# -- HSN/SAC backfill ---------------------------------------------------------
def _backfill_hsn(lines: list[LineItem], reader_text: str) -> None:
    """Fill a line's ``hsn_sac`` from the OCR text when the LLM left it blank.

    Locates the line's description in the reading-order text and takes the first
    standalone 4/6/8-digit token in the window that follows — excluding the
    line's own numeric values (qty/rate/amounts), so a rate like ``5000`` is
    never mistaken for an HSN. Purely additive: a code the model already read is
    kept as-is.
    """
    if not reader_text:
        return
    lowered = reader_text.lower()
    for item in lines:
        if item.hsn_sac or not item.description.strip():
            continue
        window = _description_window(reader_text, lowered, item.description)
        if not window:
            continue
        exclude = _line_int_values(item)
        for match in _HSN_TOKEN_RE.finditer(window):
            token = match.group(1)
            if int(token) not in exclude:
                item.hsn_sac = token
                break


def _description_window(text: str, lowered: str, description: str, width: int = 160) -> str:
    """The slice of ``text`` starting at ``description`` (matched leniently)."""
    desc = description.strip().lower()
    pos = lowered.find(desc[:24])
    if pos == -1:
        words = desc.split()
        if len(words) >= 2:
            pos = lowered.find(" ".join(words[:2]))
    if pos == -1:
        return ""
    start = pos + len(desc)
    return text[start : start + width]


def _line_int_values(item: LineItem) -> set[int]:
    """Integer forms of a line's own numbers, to exclude when hunting the HSN."""
    values = [item.quantity, item.rate, item.taxable_amount, item.discount,
              item.cgst_amount, item.sgst_amount, item.igst_amount, item.line_total]
    return {int(v) for v in values if v is not None and float(v).is_integer()}


# -- small field cleaners -----------------------------------------------------
def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_pan(value: Any) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    text = text.upper()
    return text if len(text) == 10 else None


def _clean_pincode(value: Any) -> str | None:
    digits = digits_only(value)
    return digits if digits and len(digits) == 6 else None
