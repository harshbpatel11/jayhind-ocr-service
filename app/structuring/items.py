"""Line items, totals, tax summary and invoice meta.

Two line-item paths: the exact pdfplumber `tables` on the digital-PDF fast path,
and a token-geometry column-band reader for the OCR path (no table survives an
image). Both feed one `_item_from_row` builder. Column keyword lexicons are broad
on purpose — real invoices head the amount column "Amt"/"Amount"/"Total" and the
description column "Item"/"Description"/"Particulars".
"""
import re
from typing import Dict, List, Optional

from .geometry import cx, group_lines, header_region, token_text, x0, x1, y0
from .text import round2, normalize_amount

# ── Column role lexicons (checked in priority order per header cell) ──────────
# The per-tax amount columns (CGST/SGST/IGST) are claimed explicitly — on the
# OCR path an unclaimed column's tokens glue onto the nearest claimed centre,
# turning "18%" + "270.00" into gstRate 18270. "CGST %" / "CGST Rate" still
# falls through to gstRate. "Total" claims taxable on simple bills but becomes
# the line total once a Taxable/Amount column exists.
_COL_RULES = [
    ("description", re.compile(r"^(item|description|particular|particulars|goods|product|desc)\b", re.I)),
    ("hsn", re.compile(r"^(hsn|sac)\b", re.I)),
    ("quantity", re.compile(r"^(qty|quantity|nos)\b", re.I)),
    ("unit", re.compile(r"^(unit|uom)\b", re.I)),
    ("rate", re.compile(r"^(rate|price|mrp)\b", re.I)),
    ("cgstAmount", re.compile(r"^cgst\b(?!\s*(?:%|rate))", re.I)),
    ("sgstAmount", re.compile(r"^(?:sgst|utgst)\b(?!\s*(?:%|rate))", re.I)),
    ("igstAmount", re.compile(r"^igst\b(?!\s*(?:%|rate))", re.I)),
    ("gstRate", re.compile(r"^(gst|igst|cgst|sgst|utgst|tax)\s*(%|rate)?$", re.I)),
    ("taxable", re.compile(r"^(amount|amt|value|taxable|total)\b", re.I)),
    ("discount", re.compile(r"^(disc|discount)\b", re.I)),
    ("lineTotal", re.compile(r"^(total|net|amount|line\s*total)\b", re.I)),
]

#: GST slab rates a derived per-line rate is snapped to (incl. the 2025 40%
#: demerit slab). A derived value more than 0.5pp from every slab is discarded.
GST_SLABS = [0, 0.1, 0.25, 1, 1.5, 3, 5, 7.5, 12, 18, 28, 40]


def snap_gst_rate(derived: Optional[float]) -> Optional[float]:
    if derived is None or derived < 0:
        return None
    slab = min(GST_SLABS, key=lambda s: abs(s - derived))
    return float(slab) if abs(slab - derived) <= 0.5 else None
_SKIP_DESC = re.compile(r"^(total|sub\s*-?\s*total|subtotal|grand|taxable|cgst|sgst|igst|round|amount\s*in\s*words)", re.I)
_FOOTER = re.compile(r"^\s*(sub\s*-?\s*total|subtotal|total|taxable|grand|cgst|sgst|igst|round|bank|terms|declaration|authori)", re.I)


def _clean(cell) -> str:
    return re.sub(r"\s+", " ", str(cell or "")).strip()


def _map_columns(header: List[str]) -> Dict[str, int]:
    """Header cell texts → {role: column index}. First cell to claim a role wins."""
    mapping: Dict[str, int] = {}
    for index, cell in enumerate(header):
        text = _clean(cell)
        if not text:
            continue
        for role, pattern in _COL_RULES:
            if role not in mapping and pattern.match(text):
                mapping[role] = index
                break
    return mapping


def _qty_and_unit(cell: str) -> tuple:
    """Split a quantity cell that carries its unit inline ("2 PCS", "10.5 KG")."""
    number = normalize_amount(cell)
    if number is not None:
        return number, None
    lead = re.match(r"\s*([\d,]+(?:\.\d+)?)", cell)
    qty = normalize_amount(lead[1]) if lead else None
    unit = re.search(r"([A-Za-z]{1,6})\s*$", cell)
    return (qty or 0), (unit[1] if unit else None)


def _is_continuation(item: Dict) -> bool:
    """A row that carries only wrapped description text (no numbers) — it belongs
    to the previous line item, not a new one."""
    return item["quantity"] == 0 and item["rate"] == 0 and item["taxableAmount"] == 0


def _item_from_row(cells: List[str], cols: Dict[str, int]) -> Optional[Dict]:
    def at(role):
        i = cols.get(role)
        return _clean(cells[i]) if i is not None and i < len(cells) else ""

    description = at("description")
    if not description or _SKIP_DESC.match(description):
        return None

    quantity, unit_from_qty = _qty_and_unit(at("quantity"))
    rate = normalize_amount(at("rate")) or 0
    discount = normalize_amount(at("discount"))
    taxable = normalize_amount(at("taxable"))
    if taxable is None:
        taxable = round2(quantity * rate - (discount or 0))
    gst = normalize_amount(at("gstRate").replace("%", ""))
    if gst is not None and not 0 <= gst <= 40:
        gst = None  # a "Tax" column carrying amounts, or OCR glue — not a rate
    return {
        "description": description,
        "hsnSac": at("hsn") or None,
        "quantity": quantity,
        "unit": at("unit") or unit_from_qty,
        "rate": rate,
        "discount": discount,
        "taxableAmount": taxable,
        "gstRate": gst,
        "cgstAmount": normalize_amount(at("cgstAmount")),
        "sgstAmount": normalize_amount(at("sgstAmount")),
        "igstAmount": normalize_amount(at("igstAmount")),
        "lineTotal": normalize_amount(at("lineTotal")),
        "confidence": 0.0,
    }


def _append(items: List[Dict], item: Optional[Dict]) -> None:
    """Add a parsed row, folding a description-only continuation into the item
    above it (multi-line descriptions)."""
    if not item:
        return
    if _is_continuation(item):
        if items:
            items[-1]["description"] = f"{items[-1]['description']} {item['description']}".strip()
        return
    items.append(item)


def parse_items_from_table(table: Dict) -> List[Dict]:
    rows = table.get("rows") or []
    header_index = next(
        (i for i, row in enumerate(rows)
         if any(_COL_RULES[0][1].match(_clean(c)) for c in row)),
        -1,
    )
    if header_index < 0:
        return []
    cols = _map_columns(rows[header_index])
    if "description" not in cols:
        return []
    items: List[Dict] = []
    for row in rows[header_index + 1:]:
        _append(items, _item_from_row([_clean(c) for c in row], cols))
    return items


def parse_items_from_tokens(page: Dict) -> List[Dict]:
    """OCR-path line items: find the header row, fix a column centre per role, then
    assign each body token to its nearest column by x-centre."""
    tokens = page.get("tokens") or []
    lines = group_lines(tokens)
    header_i = next(
        (i for i, line in enumerate(lines)
         if sum(1 for t in line if any(p.match(token_text(t)) for _, p in _COL_RULES)) >= 2),
        -1,
    )
    if header_i < 0:
        return []

    header = lines[header_i]
    col_centres, col_roles, seen = [], [], set()
    for t in sorted(header, key=x0):
        text = token_text(t)
        for role, pattern in _COL_RULES:
            if role not in seen and pattern.match(text):
                col_centres.append(cx(t))
                col_roles.append(role)
                seen.add(role)
                break
    if "description" not in col_roles:
        return []

    items: List[Dict] = []
    cols = {role: i for i, role in enumerate(col_roles)}
    for line in lines[header_i + 1:]:
        text = " ".join(token_text(t) for t in line)
        if _FOOTER.match(text):
            break
        cells = ["" for _ in col_centres]
        for t in line:
            nearest = min(range(len(col_centres)), key=lambda c: abs(cx(t) - col_centres[c]))
            cells[nearest] = (cells[nearest] + " " + token_text(t)).strip()
        _append(items, _item_from_row(cells, cols))
    return items


# ── Totals ────────────────────────────────────────────────────────────────────

def _trailing_amount(line: str) -> Optional[float]:
    # A whole number run (with optional thousands commas + decimals). Must NOT
    # cap the integer part at 3 digits — "60770" is one amount, not "607" + "70".
    # Dot groups may repeat: OCR reads "6,372.0" as "6.372.0", which is ONE
    # amount (normalize_amount repairs the comma-as-dot slip), not "6.372" + "0".
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)*", line)
    return normalize_amount(matches[-1]) if matches else None


def _labelled(lines: List[str], label: re.Pattern) -> Optional[float]:
    for line in lines:
        if label.search(line):
            value = _trailing_amount(line)
            if value is not None:
                return value
    return None


def _labelled_tax(lines: List[str], word: str) -> Optional[float]:
    """The document total for one tax head (CGST/SGST/IGST).

    A plain "CGST … 486.00" row is the total and wins. When only per-rate rows
    exist ("CGST @9% … 243.00", "CGST @14% … 350.00" — a multi-slab invoice),
    the total is their sum, deduplicated by rate so a tax-summary table that
    repeats the footer rows doesn't double-count.
    """
    label = re.compile(rf"\b{word}\b", re.IGNORECASE)
    rated = re.compile(rf"\b{word}\b\s*@?\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
    by_rate: dict = {}
    for line in lines:
        if not label.search(line):
            continue
        value = _trailing_amount(line)
        if value is None:
            continue
        rate_hit = rated.search(line)
        if not rate_hit:
            return value
        by_rate.setdefault(float(rate_hit[1]), value)
    return round2(sum(by_rate.values())) if by_rate else None


def parse_totals(text: str) -> Dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    taxable = _labelled(lines, re.compile(r"\btaxable\b|^sub\s*-?\s*total\b|\bsubtotal\b|before\s*tax", re.I)) or 0
    cgst = _labelled_tax(lines, "cgst")
    sgst = _labelled_tax(lines, "sgst")
    igst = _labelled_tax(lines, "igst")
    round_off = _labelled(lines, re.compile(r"round\s*(ing)?\s*off", re.I)) or 0

    grand = (
        _labelled(lines, re.compile(r"grand", re.I))
        or _labelled(lines, re.compile(r"invoice\s*(?:total|value)|amount\s*payable|net\s*(?:payable|amount)|total\s*payable", re.I))
        or _last_total(lines)
        or 0
    )

    tax_total = round2((cgst or 0) + (sgst or 0) + (igst or 0))
    words = next((ln for ln in lines if re.search(r"amount\s*in\s*words|rupees", ln, re.I)), None)
    if words:
        # Drop the label so only the words themselves remain ("Amount in
        # Words: Indian Rupees … Only" / "Amount Chargeable (in words) …").
        words = re.sub(r"^.*?(?:amount\s*(?:chargeable\s*)?\(?\s*in\s*words\s*\)?)\s*[:.\-]?\s*",
                       "", words, flags=re.IGNORECASE).strip() or words
    return {
        "totals": {
            "taxableTotal": taxable,
            "taxTotal": tax_total or round2(max(0, grand - taxable - round_off)),
            "roundOff": round_off,
            "grandTotal": grand,
            "amountInWords": words,
        },
        "cgst": cgst, "sgst": sgst, "igst": igst,
    }


def _last_total(lines: List[str]) -> Optional[float]:
    """A plain 'Total' row as a last resort for the grand total, skipping the
    sub-total / taxable rows it must not be confused with."""
    for line in reversed(lines):
        if re.search(r"\btotal\b", line, re.I) and not re.search(r"sub|taxable", line, re.I):
            value = _trailing_amount(line)
            if value is not None:
                return value
    return None


def build_tax_summary(items: List[Dict], cgst, sgst, igst) -> List[Dict]:
    inter_state = igst is not None and igst > 0
    by_rate: Dict[float, float] = {}
    for item in items:
        if item["gstRate"] is None:
            continue
        by_rate[item["gstRate"]] = round2(by_rate.get(item["gstRate"], 0) + item["taxableAmount"])

    summary = []
    for rate in sorted(by_rate):
        taxable = by_rate[rate]
        tax = round2(taxable * rate / 100)
        if inter_state:
            summary.append({"rate": rate, "taxableAmount": taxable, "cgst": 0, "sgst": 0, "igst": tax})
        else:
            half = round2(tax / 2)
            summary.append({"rate": rate, "taxableAmount": taxable, "cgst": half, "sgst": round2(tax - half), "igst": 0})
    return summary


# ── Invoice number & date ─────────────────────────────────────────────────────
_DATE_TOKEN = re.compile(r"[0-9]{1,4}[-/.\s][A-Za-z0-9]{1,9}[-/.\s][0-9]{2,4}")
_NO_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/\\\-]{1,31}")
_INV_LABEL = re.compile(r"\b(invoice|bill|inv|receipt|voucher|document|doc)\b\s*(?:no|number|#)?\s*[:.#\-]?\s*(.+)$", re.I)
_PARTY_MARKER = re.compile(r"\b(bill\s*to|billed\s*to|invoice\s*to|sold\s*to|buyer|consignee|customer)\b", re.I)


def _first_no_value(text: str) -> str:
    for token in text.split():
        candidate = _NO_VALUE.match(token)
        if candidate and re.search(r"\d", candidate[0]) and not _DATE_TOKEN.fullmatch(candidate[0]):
            return candidate[0]
    return ""


def _invoice_no_from_text(text: str) -> str:
    for line in text.split("\n"):
        if _PARTY_MARKER.search(line) and not re.search(r"\binvoice\s*(no|number|#)", line, re.I):
            continue
        m = _INV_LABEL.search(line)
        if not m:
            continue
        value = _first_no_value(m[2])
        if value:
            return value
    return ""


#: Labels for the fuzzy fallback — recovers a number when OCR mangled the *label*
#: ("lnvoice No: SS/2026/0412"). `bill no` collides with the "Bill To" party
#: marker, so marker lines are skipped first.
_INV_NO_LABELS = ["invoiceno", "invoicenumber", "invno", "billno", "billnumber", "invoice"]
_SELLER_MARKER = re.compile(r"\b(seller|supplier|sold\s*by|vendor)\b", re.I)


def _invoice_no_fuzzy(text: str) -> str:
    from .text import edit_distance
    for line in text.split("\n"):
        if _PARTY_MARKER.search(line) or _SELLER_MARKER.search(line):
            continue
        sep = re.search(r"[:#]", line)
        if not sep:
            continue
        label = re.sub(r"[^a-z]", "", line[: sep.start()].lower())
        if not (5 <= len(label) <= 16):
            continue
        if any(edit_distance(label, cand) <= 2 for cand in _INV_NO_LABELS):
            value = _first_no_value(line[sep.start() + 1:])
            if value:
                return value
    return ""


def _invoice_no_from_geometry(page: Dict) -> str:
    from .geometry import split_segments
    from .parties import marker_kind

    tokens = header_region(page)
    width = page.get("width", 0) or 1
    lines = group_lines(tokens)
    seg_lines = [split_segments(line, width, i) for i, line in enumerate(lines)]
    for row_index, row in enumerate(seg_lines):
        for pos, seg in enumerate(row):
            norm = re.sub(r"[^a-z ]", " ", seg.text.lower()).strip()
            if not re.match(r"^(invoice|bill\s*no|inv\s*no|receipt)\b", norm):
                continue
            if marker_kind(seg.text)[0]:  # a party marker, not the invoice-no label
                continue
            m = _INV_LABEL.search(seg.text)
            if m:
                value = _first_no_value(m[2])
                if value:
                    return value
            for right in row[pos + 1:]:
                value = _first_no_value(right.text)
                if value:
                    return value
            for below in seg_lines[row_index + 1:]:
                for s in below:
                    if s.x0 <= seg.x1 and s.x1 >= seg.x0 and not marker_kind(s.text)[0]:
                        value = _first_no_value(s.text)
                        if value:
                            return value
                break
    return ""


#: Label cells of an invoice-info grid ("Invoice No.", "Invoice\nDate" — cells
#: keep their content even when the page text wraps them across lines).
_INV_NO_CELL = re.compile(r"(?:tax\s+)?(?:invoice|bill|inv)\s*(?:no|number|#)?\s*[.:#]?", re.IGNORECASE)
_DATE_CELL = re.compile(r"(?:invoice|bill|inv)?\s*dated?\s*[.:#]?", re.IGNORECASE)


def _meta_value_from_tables(ocr: Dict, label: re.Pattern) -> str:
    """The cell to the right of a label cell, across every extracted table."""
    for page in ocr.get("pages", []):
        for table in page.get("tables", []):
            for row in table.get("rows") or []:
                cells = [_clean(c) for c in row]
                for i, cell in enumerate(cells):
                    if cell and label.fullmatch(cell) and i + 1 < len(cells) and cells[i + 1]:
                        return cells[i + 1]
    return ""


def parse_invoice_meta(ocr: Dict) -> Dict:
    text = ocr.get("text") or ""
    number = _invoice_no_from_text(text)
    if not number:
        number = _first_no_value(_meta_value_from_tables(ocr, _INV_NO_CELL))
    if not number and ocr.get("pages"):
        number = _invoice_no_from_geometry(ocr["pages"][0])
    if not number:
        number = _invoice_no_fuzzy(text)

    from .text import parse_date
    date_match = re.search(r"\b(?:invoice\s*date|date(?:d)?)\s*[:.\-]?\s*(" + _DATE_TOKEN.pattern + r")", text, re.I)
    date = parse_date(date_match[1]) if date_match else None
    if not date:
        date = parse_date(_meta_value_from_tables(ocr, _DATE_CELL))
    if not date:
        found = _DATE_TOKEN.search(text)
        date = parse_date(found[0]) if found else None
    return {"number": number, "date": date}
