"""Unit + golden-fixture tests for the geometry-first structuring engine.

The digital-PDF fixtures parse without OCR (fast), so the golden accuracy set
runs here as ordinary tests. Ports the spirit of the old TypeScript parser spec
and adds the layout-family coverage that motivated the rewrite.
"""
import json
import pathlib
import re

import pytest

from app.extractor import extract
from app.structuring import parse_invoice
from app.structuring.gstin import derive_basics, find_gstins, is_checksum_valid, repair_gstin
from app.structuring.items import _invoice_no_from_text, _qty_and_unit, parse_items_from_table, parse_totals
from app.structuring.parties import (
    is_document_title,
    marker_kind,
    parse_party_block,
)
from app.structuring.text import edit_distance, normalize_amount, parse_date, strip_html

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── text helpers ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1,234.00", 1234.0), ("₹ 5,000", 5000.0), ("Rs. 99.50", 99.5),
    ("(1,234.00)", -1234.0),           # accounting negative
    ("5.058.00", 5058.0),              # OCR read a thousands comma as a dot
    ("60770", 60770.0), ("abc", None), ("", None),
])
def test_normalize_amount(raw, expected):
    assert normalize_amount(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("2026-07-05", "2026-07-05"),
    ("05-07-2026", "2026-07-05"),      # day-first (Indian convention)
    ("5 Jul 2026", "2026-07-05"),
    ("5th July 2026", "2026-07-05"),   # ordinal + full month name
    ("July 5, 2026", "2026-07-05"),    # month-first (US)
    ("05.07.26", "2026-07-05"),        # dotted, 2-digit year
    ("31-02-2026", None),              # overflow rejected
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("cell,qty,unit", [
    ("2 PCS", 2, "PCS"), ("10.5 KG", 10.5, "KG"), ("3", 3, None), ("", 0, None),
])
def test_qty_and_unit(cell, qty, unit):
    assert _qty_and_unit(cell) == (qty, unit)


def test_edit_distance():
    assert edit_distance("cgst", "sgst") == 1
    assert edit_distance("invoice", "lnvoice") == 1


def test_strip_html_br_to_newline():
    assert strip_html("<b>A</b><br/>B") == "A\nB"


# ── GSTIN ─────────────────────────────────────────────────────────────────────

def test_derive_basics_state_and_pan():
    b = derive_basics("24AAHCV3778L1ZQ")
    assert b["valid"] and b["stateCode"] == "24" and b["stateName"] == "Gujarat"
    assert b["panNo"] == "AAHCV3778L"


def test_repair_gstin_letter_in_digit_slot_is_forced():
    # 'O' in a digit slot can only be '0' — no checksum needed.
    assert repair_gstin("Z4AAHCV3778L1ZQ") == "24AAHCV3778L1ZQ"


def test_repair_gstin_returns_none_when_ambiguous_and_unconfirmable():
    assert repair_gstin("!!!") is None
    assert repair_gstin("000000000000000") is None


def test_find_gstins_dedupes_in_order():
    text = "Seller 24AAHCV3778L1ZQ ... buyer 24AJGPP6816J1ZY ... 24AAHCV3778L1ZQ"
    assert find_gstins(text) == ["24AAHCV3778L1ZQ", "24AJGPP6816J1ZY"]


def test_checksum_is_not_a_validity_gate():
    # Sample GSTINs need not satisfy the check digit; repair must still accept a
    # plainly well-formed one.
    assert repair_gstin("24ABCDE1234F1Z1") == "24ABCDE1234F1Z1"


# ── titles / markers ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("line,is_title", [
    ("TAX INVOICE", True),
    ("PURCHASE TAX INVOICE", True),
    ("WHOLESALE GST INVOICE", True),
    ("ORIGINAL FOR RECIPIENT", True),
    ("Acme Pumps Pvt Ltd", False),
    ("NEWSPAPER STYLE", False),
])
def test_is_document_title(line, is_title):
    assert is_document_title(line) is is_title


@pytest.mark.parametrize("text,kind,rest", [
    ("Supplier: Acme", "seller", "Acme"),
    ("Bill To: Metro", "buyer", "Metro"),
    ("From: Acme Pumps", "seller", "Acme Pumps"),
    ("Ship To: Warehouse", None, ""),      # ship-to is NOT the buyer
    ("Acme Pumps Pvt Ltd", None, ""),
])
def test_marker_kind(text, kind, rest):
    assert marker_kind(text) == (kind, rest)


def test_parse_party_block_keeps_pan_and_state_out_of_address():
    block = parse_party_block(
        "Supplier: Shree Trading Co.\n21 GIDC Estate, Ahmedabad\n"
        "GSTIN: 24AAHCV3778L1ZQ\nPAN: AAHCV3778L\nState: Gujarat"
    )
    assert block["name"] == "Shree Trading Co."
    assert block["gstin"] == "24AAHCV3778L1ZQ"
    assert "PAN" not in block["address"] and "GSTIN" not in block["address"]
    assert block["stateName"] == "Gujarat"


def test_parse_party_block_reads_phone_email_and_pincode():
    """These three are printed but are not derivable from the GSTIN, so the party
    form falls back to them when the GST auto-fill leaves a field empty."""
    block = parse_party_block(
        "Supplier: Prime Industrial Supplies Pvt. Ltd.\n"
        "Unit-7, Naroda GIDC, Ahmedabad, Gujarat - 382330\n"
        "GSTIN: 24AABCP3456M1Z7\nPAN: AABCP3456M\nState: Gujarat (24)\n"
        "Phone: +91-79-40001234\nEmail: accounts@prime-demo.in"
    )
    assert block["phone"] == "917940001234"
    assert block["email"] == "accounts@prime-demo.in"
    assert block["pincode"] == "382330"
    # Contact lines are meta lines, so they never leak into the address.
    assert "Phone" not in block["address"] and "@" not in block["address"]


def test_parse_party_block_bare_mobile_and_no_contact():
    """An unlabelled 10-digit mobile is still found; a block carrying no contact
    details yields None rather than a stray slice of some other number."""
    with_mobile = parse_party_block("Metro Infra Ltd\n12 Ring Road, Surat\n9876543210")
    assert with_mobile["phone"] == "9876543210"

    # No PIN in the address, and an invoice number is never mistaken for one.
    bare = parse_party_block("Acme Co\nPlot 4, MIDC\nInvoice No: INV-2026-1102")
    assert bare["phone"] is None and bare["email"] is None and bare["pincode"] is None


# ── totals (broadened lexicons) ───────────────────────────────────────────────

def test_parse_totals_terse_labels():
    t = parse_totals("Taxable 51500\nCGST 4635\nSGST 4635\nGrand 60770")
    assert t["totals"]["taxableTotal"] == 51500
    assert t["totals"]["grandTotal"] == 60770
    assert t["cgst"] == 4635 and t["sgst"] == 4635


def test_parse_totals_plain_total_is_grand():
    assert parse_totals("Milk 62\nTotal 185")["totals"]["grandTotal"] == 185


# ── line items: real-world variety ────────────────────────────────────────────

def test_multiline_description_merges_into_previous_item():
    table = {"rows": [
        ["Description", "Qty", "Rate", "Amount"],
        ["Industrial Motor", "2", "12500", "25000"],
        ["with thermal overload protection", "", "", ""],   # wrapped description
        ["Copper Cable", "50", "240", "12000"],
    ]}
    items = parse_items_from_table(table)
    assert len(items) == 2
    assert "thermal overload" in items[0]["description"]


def test_discount_and_inline_unit():
    table = {"rows": [
        ["Description", "Qty", "Rate", "Disc", "Amount"],
        ["Widget", "3 PCS", "100", "50", ""],   # amount blank → qty*rate - disc
    ]}
    item = parse_items_from_table(table)[0]
    assert item["quantity"] == 3 and item["unit"] == "PCS"
    assert item["discount"] == 50 and item["taxableAmount"] == 250


def test_invoice_number_voucher_label():
    assert _invoice_no_from_text("Voucher No: V-2026-77") == "V-2026-77"


# ── golden accuracy set (the layout families) ─────────────────────────────────

def _golden():
    return json.loads((FIXTURES / "layout_golden.json").read_text())


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _name_ok(got, exp):
    g, e = _norm(got), _norm(exp)
    return g == "" if e == "" else bool(g) and (g == e or e in g or g in e)


@pytest.mark.parametrize("g", _golden(), ids=lambda g: g["file"])
def test_layout_golden(g):
    ocr = extract((FIXTURES / g["file"]).read_bytes(), "application/pdf")
    inv = parse_invoice(ocr)
    assert _name_ok(inv["seller"]["name"], g["seller_name"]), f"seller={inv['seller']['name']!r}"
    assert _name_ok(inv["buyer"]["name"], g["buyer_name"]), f"buyer={inv['buyer']['name']!r}"
    assert len(inv["lineItems"]) == g["line_count"]
    if g.get("grand") is not None:
        assert abs(inv["totals"]["grandTotal"] - g["grand"]) <= 1
    if g["invoice_no"]:
        assert _norm(inv["invoice"]["number"]) == _norm(g["invoice_no"])
