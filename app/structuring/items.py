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
_COL_RULES = [
    ("description", re.compile(r"^(item|description|particular|particulars|goods|product|desc)\b", re.I)),
    ("hsn", re.compile(r"^(hsn|sac)\b", re.I)),
    ("quantity", re.compile(r"^(qty|quantity|nos)\b", re.I)),
    ("unit", re.compile(r"^(unit|uom)\b", re.I)),
    ("rate", re.compile(r"^(rate|price|mrp)\b", re.I)),
    ("gstRate", re.compile(r"^(gst|igst|cgst|sgst|tax)\s*(%|rate)?$", re.I)),
    ("taxable", re.compile(r"^(amount|amt|value|taxable|total)\b", re.I)),
    ("discount", re.compile(r"^(disc|discount)\b", re.I)),
]
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


def _item_from_row(cells: List[str], cols: Dict[str, int]) -> Optional[Dict]:
    def at(role):
        i = cols.get(role)
        return _clean(cells[i]) if i is not None and i < len(cells) else ""

    description = at("description")
    if not description or _SKIP_DESC.match(description):
        return None

    quantity = normalize_amount(at("quantity")) or 0
    rate = normalize_amount(at("rate")) or 0
    taxable = normalize_amount(at("taxable"))
    if taxable is None:
        taxable = round2(quantity * rate)
    gst = normalize_amount(at("gstRate").replace("%", ""))
    return {
        "description": description,
        "hsnSac": at("hsn") or None,
        "quantity": quantity,
        "unit": at("unit") or None,
        "rate": rate,
        "discount": normalize_amount(at("discount")),
        "taxableAmount": taxable,
        "gstRate": gst,
        "cgstAmount": None, "sgstAmount": None, "igstAmount": None,
        "lineTotal": None,
        "confidence": 0.0,
    }


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
    items = []
    for row in rows[header_index + 1:]:
        item = _item_from_row([_clean(c) for c in row], cols)
        if item:
            items.append(item)
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

    items = []
    for line in lines[header_i + 1:]:
        text = " ".join(token_text(t) for t in line)
        if _FOOTER.match(text):
            break
        cells = ["" for _ in col_centres]
        for t in line:
            nearest = min(range(len(col_centres)), key=lambda c: abs(cx(t) - col_centres[c]))
            cells[nearest] = (cells[nearest] + " " + token_text(t)).strip()
        cols = {role: i for i, role in enumerate(col_roles)}
        item = _item_from_row(cells, cols)
        if item:
            items.append(item)
    return items


# ── Totals ────────────────────────────────────────────────────────────────────

def _trailing_amount(line: str) -> Optional[float]:
    # A whole number run (with optional thousands commas + decimals). Must NOT
    # cap the integer part at 3 digits — "60770" is one amount, not "607" + "70".
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", line)
    return normalize_amount(matches[-1]) if matches else None


def _labelled(lines: List[str], label: re.Pattern) -> Optional[float]:
    for line in lines:
        if label.search(line):
            value = _trailing_amount(line)
            if value is not None:
                return value
    return None


def parse_totals(text: str) -> Dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    taxable = _labelled(lines, re.compile(r"\btaxable\b|^sub\s*-?\s*total\b|\bsubtotal\b|before\s*tax", re.I)) or 0
    cgst = _labelled(lines, re.compile(r"\bcgst\b", re.I))
    sgst = _labelled(lines, re.compile(r"\bsgst\b", re.I))
    igst = _labelled(lines, re.compile(r"\bigst\b", re.I))
    round_off = _labelled(lines, re.compile(r"round\s*(ing)?\s*off", re.I)) or 0

    grand = (
        _labelled(lines, re.compile(r"grand", re.I))
        or _labelled(lines, re.compile(r"invoice\s*(?:total|value)|amount\s*payable|net\s*(?:payable|amount)|total\s*payable", re.I))
        or _last_total(lines)
        or 0
    )

    tax_total = round2((cgst or 0) + (sgst or 0) + (igst or 0))
    words = next((ln for ln in lines if re.search(r"amount\s*in\s*words|rupees", ln, re.I)), None)
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
_INV_LABEL = re.compile(r"\b(invoice|bill|inv|receipt)\b\s*(?:no|number|#)?\s*[:.#\-]?\s*(.+)$", re.I)
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


def parse_invoice_meta(ocr: Dict) -> Dict:
    text = ocr.get("text") or ""
    number = _invoice_no_from_text(text)
    if not number and ocr.get("pages"):
        number = _invoice_no_from_geometry(ocr["pages"][0])
    if not number:
        number = _invoice_no_fuzzy(text)

    from .text import parse_date
    date_match = re.search(r"\b(?:invoice\s*date|date(?:d)?)\s*[:.\-]?\s*(" + _DATE_TOKEN.pattern + r")", text, re.I)
    date = parse_date(date_match[1]) if date_match else None
    if not date:
        found = _DATE_TOKEN.search(text)
        date = parse_date(found[0]) if found else None
    return {"number": number, "date": date}
