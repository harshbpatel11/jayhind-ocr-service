"""Structuring entry point: OcrResult → ExtractedInvoice (schema: plan §6).

Geometry-first and fully offline. Mirrors the shape the NestJS matching layer
consumes (`ExtractedInvoice` in the backend), so Python produces it and Node
keeps matching against the DB unchanged. Never throws on a poor scan — it returns
whatever it read with low confidence scores so the reviewer sees the flagged gaps.
"""
from typing import Dict, List

from .items import build_tax_summary, parse_invoice_meta, parse_items_from_table, parse_items_from_tokens, parse_totals, snap_gst_rate
from .parties import detect_parties
from .text import round2, strip_html

EXTRACTED_SCHEMA_VERSION = 1
LOW_CONFIDENCE_THRESHOLD = 0.75
AMOUNT_TOLERANCE = 1


def _baseline_confidence(ocr: Dict) -> float:
    if ocr.get("method") == "pdf-text":
        return 1.0
    tokens = [t for page in ocr.get("pages", []) for t in page.get("tokens", [])]
    if not tokens:
        return 0.0
    return sum(t.get("confidence", 0) for t in tokens) / len(tokens)


def _amounts_foot(line_items: List[Dict], totals: Dict) -> bool:
    if not line_items or not totals["grandTotal"]:
        return False
    line_sum = round2(sum(i["taxableAmount"] for i in line_items))
    taxable = totals["taxableTotal"]
    discount = totals.get("discountTotal") or 0
    # Line nets reach the taxable directly (no whole-bill discount) or after it.
    if taxable and abs(line_sum - taxable) > AMOUNT_TOLERANCE and abs(line_sum - discount - taxable) > AMOUNT_TOLERANCE:
        return False
    expected = round2(taxable + totals["taxTotal"] + totals["roundOff"])
    return abs(expected - totals["grandTotal"]) <= AMOUNT_TOLERANCE


def _resolve_gst_rate(item: Dict) -> None:
    """Fill or repair a line's GST rate from the amounts the invoice states.

    Missing rate → derive from stated CGST+SGST / IGST, else from the stated
    line total, snapped to a real slab. A rate exactly half the effective rate
    is a "CGST Rate" column read as the GST rate — doubled back.
    """
    taxable = item["taxableAmount"]
    if not taxable:
        return
    stated_tax = None
    if item["igstAmount"] is not None:
        stated_tax = item["igstAmount"]
    elif item["cgstAmount"] is not None and item["sgstAmount"] is not None:
        stated_tax = round2(item["cgstAmount"] + item["sgstAmount"])

    if item["gstRate"] is None:
        derived = None
        if stated_tax is not None:
            derived = stated_tax / taxable * 100
        elif item["lineTotal"]:
            derived = (item["lineTotal"] - taxable) / taxable * 100
        item["gstRate"] = snap_gst_rate(derived)
    elif stated_tax is not None:
        expected = round2(taxable * item["gstRate"] / 100)
        doubled = round2(taxable * item["gstRate"] * 2 / 100)
        if abs(stated_tax - expected) > AMOUNT_TOLERANCE and abs(stated_tax - doubled) <= AMOUNT_TOLERANCE:
            item["gstRate"] = round2(item["gstRate"] * 2)


def _score_line(item: Dict, baseline: float) -> float:
    score = baseline
    if not item["hsnSac"]:
        score *= 0.9
    if item["gstRate"] is None:
        score *= 0.9
    if item["quantity"] <= 0 or item["rate"] <= 0:
        score *= 0.6
    # The net taxable must sit within [0, gross]; the implied discount is
    # gross − net. (The raw `discount` cell is unreliable — it may hold "5%" —
    # so it is not used to check the line.)
    gross = round2(item["quantity"] * item["rate"])
    if gross > 0 and not (-AMOUNT_TOLERANCE <= item["taxableAmount"] <= gross + AMOUNT_TOLERANCE):
        score *= 0.6
    return round2(min(1.0, score))


def _overall_confidence(field_confidence: Dict, line_items: List[Dict]) -> float:
    """One headline score for the whole document.

    Half the weight on the header fields (party identity, invoice number/date,
    totals — absent fields already score 0), half on the mean line confidence.
    No line items at all means the parser failed at its main job, so the line
    half contributes 0 and the overall lands well under any sane threshold.
    """
    header = sum(field_confidence.values()) / len(field_confidence) if field_confidence else 0.0
    lines = (
        sum(i.get("confidence", 0) for i in line_items) / len(line_items)
        if line_items
        else 0.0
    )
    return round2(0.5 * header + 0.5 * lines)


def parse_invoice(ocr: Dict) -> Dict:
    text = strip_html(ocr.get("text") or "")
    baseline = _baseline_confidence(ocr)

    parties = detect_parties(ocr)
    seller, buyer = parties["seller"], parties["buyer"]
    meta = parse_invoice_meta(ocr)

    # Line items: recognised tables first (exact cells on the pdf-text path), then
    # token-geometry (OCR path). Each page contributes its first item table.
    line_items: List[Dict] = []
    for page in ocr.get("pages", []):
        for table in page.get("tables", []):
            items = parse_items_from_table(table)
            if items:
                line_items.extend(items)
                break
    if not line_items:
        for page in ocr.get("pages", []):
            items = parse_items_from_tokens(page)
            if items:
                line_items.extend(items)
                break

    parsed = parse_totals(text)
    totals, cgst, sgst, igst = parsed["totals"], parsed["cgst"], parsed["sgst"], parsed["igst"]
    tax_summary = build_tax_summary(line_items, cgst, sgst, igst)

    inter_state = (igst is not None and igst > 0) or any(i["igstAmount"] for i in line_items)
    for item in line_items:
        _resolve_gst_rate(item)
        if item["gstRate"] is None:
            continue
        item_tax = round2(item["taxableAmount"] * item["gstRate"] / 100)
        # Stated amounts (read off the invoice's own CGST/SGST/IGST columns)
        # win; the rate only fills what the document didn't print.
        if inter_state:
            if item["igstAmount"] is None:
                item["igstAmount"] = item_tax
        else:
            if item["cgstAmount"] is None:
                item["cgstAmount"] = round2(item_tax / 2)
            if item["sgstAmount"] is None:
                item["sgstAmount"] = round2(item_tax - item["cgstAmount"])
        if item["lineTotal"] is None:
            item["lineTotal"] = round2(item["taxableAmount"] + item_tax)

    math_ok = _amounts_foot(line_items, totals)
    math_factor = 1.0 if math_ok else 0.7
    for item in line_items:
        item["confidence"] = _score_line(item, baseline)

    field_confidence = {
        "seller.gstin": baseline if seller["gstin"] else 0,
        "seller.name": round2(baseline * 0.9) if seller["name"] else 0,
        "buyer.gstin": baseline if buyer["gstin"] else 0,
        "buyer.name": round2(baseline * 0.9) if buyer["name"] else 0,
        "invoice.number": round2(baseline * 0.95) if meta["number"] else 0,
        "invoice.date": baseline if meta["date"] else 0,
        "totals.taxableTotal": round2(baseline * math_factor) if totals["taxableTotal"] else 0,
        "totals.taxTotal": round2(baseline * math_factor) if totals["taxTotal"] else 0,
        "totals.grandTotal": round2(baseline * math_factor) if totals["grandTotal"] else 0,
    }
    # Only surface a discount-confidence signal when the invoice actually carries
    # a whole-bill discount — so plain invoices' overall score is unchanged.
    if totals.get("discountTotal"):
        field_confidence["totals.discountTotal"] = round2(baseline * math_factor)

    return {
        "schemaVersion": EXTRACTED_SCHEMA_VERSION,
        "seller": seller,
        "buyer": buyer,
        "invoice": meta,
        "lineItems": line_items,
        "taxSummary": tax_summary,
        "totals": totals,
        "fieldConfidence": field_confidence,
        "overallConfidence": _overall_confidence(field_confidence, line_items),
    }
