"""Structuring entry point: OcrResult → ExtractedInvoice (schema: plan §6).

Geometry-first and fully offline. Mirrors the shape the NestJS matching layer
consumes (`ExtractedInvoice` in the backend), so Python produces it and Node
keeps matching against the DB unchanged. Never throws on a poor scan — it returns
whatever it read with low confidence scores so the reviewer sees the flagged gaps.
"""
from typing import Dict, List

from .items import build_tax_summary, parse_invoice_meta, parse_items_from_table, parse_items_from_tokens, parse_totals
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
    if totals["taxableTotal"] and abs(line_sum - totals["taxableTotal"]) > AMOUNT_TOLERANCE:
        return False
    expected = round2(totals["taxableTotal"] + totals["taxTotal"] + totals["roundOff"])
    return abs(expected - totals["grandTotal"]) <= AMOUNT_TOLERANCE


def _score_line(item: Dict, baseline: float) -> float:
    score = baseline
    if not item["hsnSac"]:
        score *= 0.9
    if item["gstRate"] is None:
        score *= 0.9
    if item["quantity"] <= 0 or item["rate"] <= 0:
        score *= 0.6
    expected = round2(item["quantity"] * item["rate"] - (item["discount"] or 0))
    if abs(expected - item["taxableAmount"]) > AMOUNT_TOLERANCE:
        score *= 0.6
    return round2(min(1.0, score))


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

    inter_state = igst is not None and igst > 0
    for item in line_items:
        if item["gstRate"] is None:
            continue
        item_tax = round2(item["taxableAmount"] * item["gstRate"] / 100)
        if inter_state:
            item["igstAmount"] = item_tax
        else:
            item["cgstAmount"] = round2(item_tax / 2)
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

    return {
        "schemaVersion": EXTRACTED_SCHEMA_VERSION,
        "seller": seller,
        "buyer": buyer,
        "invoice": meta,
        "lineItems": line_items,
        "taxSummary": tax_summary,
        "totals": totals,
        "fieldConfidence": field_confidence,
    }
