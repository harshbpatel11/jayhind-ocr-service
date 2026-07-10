"""Party-block detection — supplier & buyer names/addresses/GSTINs.

Multiple strategies, best-per-party wins (a named block, GSTIN preferred):

  1. table    — marker cell in a party table → the value cell beside/below it.
                Handles boxed / grid / three-column / side-by-side headers.
  2. geometry — marker token → the column below it or cell to its right.
  3. text     — inline markers on a line ("From: Acme", "Supplier: A | Buyer: B")
                and stacked blocks; also the HTML-leak case (<br> → newlines).
  4. letterhead — no markers anywhere → the top non-title block is the seller
                (POS receipts, plain letterheads).

The document *title* is never a party: it sits above the anchors, so
anchor-relative reading kills the old "title-becomes-the-name" bug. Output blocks
match the TS `ExtractedPartyBlock` (`{name, address, gstin, stateCode, stateName,
pan, phone, email, pincode}`).

`phone` / `email` / `pincode` are printed on the invoice but are *not* encoded in
a GSTIN and are absent from the GST registry lookup, so the document is the only
source for them — the party quick-add form falls back to these when the GSTIN
auto-fill leaves a field empty.
"""
import re
from typing import Dict, List, Optional, Tuple

from .geometry import Segment, group_lines, header_region, split_segments, token_text
from .gstin import GST_STATE_NAME_BY_CODE, GSTIN_SCAN_REGEX, derive_basics, find_gstins
from .text import clean_name, strip_html

# ── Markers (longest phrase first so "bill to" wins over "bill") ──────────────
# "ship to" is deliberately NOT a buyer marker (it is the delivery address).
_BUYER_PHRASES = ["billed to", "bill to", "invoice to", "sold to", "buyer", "consignee", "customer", "to"]
_SELLER_PHRASES = ["sold by", "supplier", "seller", "vendor", "from", "m/s", "mfr", "manufacturer"]
_ALL_PHRASES = [(p, "buyer") for p in _BUYER_PHRASES] + [(p, "seller") for p in _SELLER_PHRASES]

_EMPTY = {"name": "", "address": "", "gstin": None, "stateCode": None, "stateName": None,
          "pan": None, "phone": None, "email": None, "pincode": None}


def _norm_lead(text: str) -> str:
    return re.sub(r"[^a-z/ ]", " ", text.lower()).strip()


def marker_kind(text: str) -> Tuple[Optional[str], str]:
    """('seller'|'buyer'|None, inline_rest). `inline_rest` is the text after the
    marker on the same fragment ("From: Acme" → 'Acme'); '' for a bare label."""
    lead = _norm_lead(text)
    for phrase in _BUYER_PHRASES:
        if lead == phrase or lead.startswith(phrase + " "):
            return "buyer", _after_marker(text, phrase)
    for phrase in _SELLER_PHRASES:
        if lead == phrase or lead.startswith(phrase + " "):
            return "seller", _after_marker(text, phrase)
    return None, ""


def _after_marker(text: str, phrase: str) -> str:
    words = phrase.split()
    pattern = r"^\s*" + r"[\s./]*".join(re.escape(w) for w in words) + r"\s*[:;.\-]?\s*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()


# ── Title / label / meta recognisers ─────────────────────────────────────────
_TITLE_QUALIFIER = re.compile(
    r"\b(?:original|duplicate|triplicate|extra|copy|for|recipient|transporter|supplier|buyer|"
    r"customer|purchase|sales?|retail|wholesale|gst|composite|revised|supplementary|commercial|tax)\b",
    re.IGNORECASE,
)
_TITLE_CORE = {"invoice", "einvoice", "billofsupply", "cashmemo", "creditnote", "debitnote",
               "proforma", "proformainvoice", "deliverychallan", "challan"}
_COPY_MARKER = re.compile(r"\b(original|duplicate|triplicate|copy)\b", re.IGNORECASE)


def is_document_title(line: str) -> bool:
    text = (line or "").strip()
    if not text or len(text) > 60:
        return False
    core = re.sub(r"[^a-z]", "", _TITLE_QUALIFIER.sub(" ", text).lower())
    if not core:
        return bool(_COPY_MARKER.search(text))
    return core in _TITLE_CORE


_LABEL_ONLY = re.compile(
    r"^(gstin|gst\s*no\.?|state|pan|bill(ed)?\s*to|buyer|consignee|ship\s*to|seller|supplier|"
    r"sold\s*by|sold\s*to|vendor|from|customer|invoice\s*to|menu|bank|terms|details)\s*[:.\-]?\s*$",
    re.IGNORECASE,
)
_META_LINE = re.compile(
    r"^\s*(?:gstin|gst\s*(?:in|no|number|reg)\b|gst\s*[:.#\-]|pan(?:\s*(?:no|number|card))?\b|"
    r"place\s*of\s*supply|phone|tel(?:ephone)?|mob(?:ile)?|contact\s*(?:no|number)|e-?mail|website|"
    r"invoice\s*(?:no|number|date)|bill\s*(?:no|date)|cin|irn|ack|reverse\s*charge)\b",
    re.IGNORECASE,
)
_META_WITH_SEP = re.compile(
    r"^\s*(?:state(?:\s*code)?|dated?|due\s*date|po\s*(?:no|number)?|ref(?:erence)?)\s*[:.#\-]",
    re.IGNORECASE,
)
#: Bare field-label words that can head a header cell but never name a party.
_FIELD_LABEL_WORDS = {
    "invoice", "date", "due", "po", "ref", "reference", "place", "hsn", "sac", "qty",
    "rate", "amount", "gstin", "pan", "state", "bank", "terms", "menu", "details",
    "no", "number", "sr", "sno",
}


def is_meta_line(line: str) -> bool:
    return bool(_META_LINE.match(line) or _META_WITH_SEP.match(line))


def is_label_only(line: str) -> bool:
    return bool(_LABEL_ONLY.match(line.strip()))


_MARKER_STRIP = re.compile(
    r"\b(?:bill(?:ed)?\s*to|invoice\s*to|ship\s*to|sold\s*by|sold\s*to|buyer|consignee|customer|"
    r"seller|supplier|vendor|from|m/s)\b",
    re.IGNORECASE,
)
_LEAD_PUNCT = re.compile(r"^[\s:;.,\-|]+")
_TRAIL_PUNCT = re.compile(r"[\s:;,\-|]+$")
_ADDRESS_KW = re.compile(
    r"\b(road|street|st|nagar|marg|sector|plot|survey|unit|floor|gidc|estate|opp|near|dist|"
    r"lane|block|phase|colony|society|complex|tower|building)\b", re.IGNORECASE,
)
_LEGAL_SUFFIX = re.compile(
    r"\b(pvt|private|ltd|limited|llp|inc|co|company|corp|corporation|industries|enterprises|"
    r"traders|trading|sons|agencies|associates|international)\b", re.IGNORECASE,
)
_ADDRESS_START = re.compile(
    r"^(gstin|gst|pan|state|ph|phone|tel|mob|mobile|email|www|http|plot|survey|unit|floor|"
    r"road|street|nagar|marg|sector|gidc|estate|opp|near|dist|no\.?)$", re.IGNORECASE,
)


def _strip_markers(line: str) -> str:
    out = _MARKER_STRIP.sub(" ", line)
    out = _TRAIL_PUNCT.sub("", _LEAD_PUNCT.sub("", out))
    return re.sub(r"\s{2,}", " ", out).strip()


# ── Contact details (phone / email / pincode) ────────────────────────────────
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
#: A labelled number: "Phone: +91-79-40001234", "Mob. 98765 43210". The value run
#: allows digits/space/dash/parens and must end on a digit, so a trailing
#: "Email:" or a second comma-separated number is never swallowed.
_PHONE_LABELLED_RE = re.compile(
    r"(?:phone|telephone|tel|mob(?:ile)?|contact(?:\s*(?:no|number))?|cell|ph)\b[\s:.#\-]*"
    r"(\+?\d[\d\s\-()]{7,}\d)",
    re.IGNORECASE,
)
#: An unlabelled 10-digit Indian mobile (leads 6–9) anywhere in the block.
_MOBILE_RE = re.compile(r"(?<!\d)([6-9]\d{9})(?!\d)")
#: A standalone 6-digit PIN. Digit-bounded, so it cannot sit inside a longer
#: number; a GSTIN/PAN always carries letters, so neither can match here.
_PINCODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _extract_contact(block_text: str, address: str):
    """(phone, email, pincode) for one party block.

    Phone keeps its digits only (with any 91/0 trunk prefix) — the consumer
    normalises to a bare 10-digit number. The PIN is read from the *address*
    rather than the whole block: the block also holds phone numbers, and an
    8–12 digit phone run must never surrender a 6-digit slice as a PIN.
    """
    email_hit = _EMAIL_RE.search(block_text)

    phone = None
    labelled = _PHONE_LABELLED_RE.search(block_text)
    if labelled:
        digits = re.sub(r"\D", "", labelled[1])
        # 10 = bare mobile/landline, 11 = 0-trunk, 12 = 91-prefixed.
        if 10 <= len(digits) <= 12:
            phone = digits
    if not phone:
        bare = _MOBILE_RE.search(block_text)
        phone = bare[1] if bare else None

    # The PIN trails an Indian address ("… Ahmedabad, Gujarat - 382330").
    pins = _PINCODE_RE.findall(address or "")

    return phone, (email_hit[0] if email_hit else None), (pins[-1] if pins else None)


def _looks_like_address(line: str) -> bool:
    return bool(re.search(r"\d", line) or "," in line or _ADDRESS_KW.search(line))


def _trim_name(text: str) -> str:
    """Cut a name candidate at the first token that starts an address/GSTIN, for
    the OCR/leak case where name + address land on one line."""
    hit = GSTIN_SCAN_REGEX.search(text.upper())
    if hit:
        text = text[: hit.start()]
    tokens = text.split()
    cut = len(tokens)
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        bare = re.sub(r"[^A-Za-z0-9]", "", tok)
        if _ADDRESS_START.match(bare) or re.fullmatch(r"\d{1,6}", bare):
            cut = i
            break
    return " ".join(tokens[:cut])


def _name_from_lines(lines: List[str]) -> str:
    """The party name from a block's text lines."""
    candidates: List[str] = []
    for line in lines:
        if is_document_title(line) or is_label_only(line) or is_meta_line(line):
            continue
        stripped = _strip_markers(line)
        if stripped:
            candidates.append(stripped)
    if not candidates:
        return ""

    # A name wrapped over several short address-free lines ("Metro\nInfra\nLtd").
    if (len(candidates) >= 2 and not any(_looks_like_address(c) for c in candidates)
            and sum(len(c.split()) for c in candidates) <= 6):
        return clean_name(" ".join(candidates))

    parts = [_trim_name(candidates[0])]
    for c in candidates[1:]:
        if _LEGAL_SUFFIX.search(c) and not re.search(r"\d", c):
            parts.append(c)
        else:
            break
    return clean_name(" ".join(parts))


def _is_value_text(text: str) -> bool:
    """Is this fragment a plausible party value (a name), not a label/marker/meta?"""
    t = (text or "").strip()
    if not t or marker_kind(t)[0] or is_label_only(t) or is_meta_line(t) or is_document_title(t):
        return False
    if re.sub(r"[^a-z]", "", t.lower()) in _FIELD_LABEL_WORDS:
        return False
    return bool(re.search(r"[A-Za-z]", t))


def _value_from_cell(cell: str) -> Optional[str]:
    """A party name out of a cell: inline text after a marker, else the cell if it
    is itself a value."""
    kind, inline = marker_kind(cell)
    if kind and _is_value_text(inline):
        return inline
    if _is_value_text(cell):
        return cell
    return None


def parse_party_block(block_text: str) -> Dict:
    """Parse one party block's text → name / address / GSTIN / state / PAN /
    phone / email / pincode."""
    normalized = strip_html(block_text)
    lines = [ln.strip() for ln in normalized.split("\n") if ln.strip()]
    if not lines:
        return dict(_EMPTY)

    gstin = (find_gstins(normalized) or [None])[0]
    basics = derive_basics(gstin) if gstin else None
    name = _name_from_lines(lines)

    address_parts = []
    for line in lines:
        if is_document_title(line) or is_label_only(line) or is_meta_line(line):
            continue
        stripped = _strip_markers(line)
        if not stripped or (name and (stripped in name or clean_name(_trim_name(stripped)) == name)):
            continue
        address_parts.append(stripped)
    address = ", ".join(address_parts)

    state_name = basics["stateName"] if basics else None
    if not state_name:
        for line in lines:
            m = re.match(r"^state\s*[:.\-]\s*(.+)$", line.strip(), re.IGNORECASE)
            if m and m[1].split("(")[0].strip() in GST_STATE_NAME_BY_CODE.values():
                state_name = m[1].split("(")[0].strip()
                break

    # Read contacts off the *unfiltered* block: phone/email sit on meta lines,
    # which `address` deliberately drops.
    phone, email, pincode = _extract_contact(normalized, address)

    return {
        "name": name,
        "address": address,
        "gstin": basics["gstNo"] if (basics and basics["valid"]) else gstin,
        "stateCode": basics["stateCode"] if basics else None,
        "stateName": state_name,
        "pan": basics["panNo"] if basics else None,
        "phone": phone,
        "email": email,
        "pincode": pincode,
    }


# ── Strategy 1: party table ───────────────────────────────────────────────────
_ITEM_HDR = re.compile(
    r"^(item|description|particular|goods|product|qty|quantity|rate|price|amount|amt|hsn|sac|gst)\b",
    re.IGNORECASE,
)


def _is_item_table(rows: List[List]) -> bool:
    return any(sum(1 for c in row if _ITEM_HDR.match(str(c or "").strip())) >= 2 for row in rows)


def _detect_by_table(page: Dict) -> Dict:
    result = {"seller": dict(_EMPTY), "buyer": dict(_EMPTY), "markers": 0}
    for table in page.get("tables") or []:
        rows = [[str(c or "").strip() for c in row] for row in (table.get("rows") or [])]
        if not rows or _is_item_table(rows):
            continue
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                kind, inline = marker_kind(cell)
                if kind not in ("seller", "buyer"):
                    continue
                result["markers"] += 1
                value = None
                if _is_value_text(inline):
                    value = inline
                elif c + 1 < len(row):
                    value = _value_from_cell(row[c + 1])
                if not value and r + 1 < len(rows) and c < len(rows[r + 1]):
                    value = _value_from_cell(rows[r + 1][c])
                if value and not result[kind]["name"]:
                    result[kind] = parse_party_block(value)
    return result


# ── Strategy 2: geometry (segments) ───────────────────────────────────────────

def _overlaps(seg: Segment, lo: float, hi: float) -> bool:
    return seg.x0 <= hi and seg.x1 >= lo


def _block_lines_for_anchor(anchor: Segment, kind: str, seg_lines: List[List[Segment]]) -> List[str]:
    """Text lines belonging to a party anchor. Stops the column scan only at the
    *opposite* party's marker — a party's own label leaked into its cell
    ("<br>Buyer<br>Blue Star…") must not end the block."""
    opposite = "seller" if kind == "buyer" else "buyer"
    lines: List[str] = []
    _, inline = marker_kind(anchor.text)
    if inline:
        lines.append(_split_at_other_marker(inline))
    else:
        # Grid: value to the right, unless there is a real column below.
        row = seg_lines[anchor.line_index]
        right = [s for s in row if s.x0 > anchor.x1 and _is_value_text(s.text)]
        below_first = next(
            (s for below in seg_lines[anchor.line_index + 1:]
             for s in below if _overlaps(s, anchor.x0, anchor.x1)),
            None,
        )
        if right and (below_first is None or not _is_value_text(below_first.text)
                      or marker_kind(below_first.text)[0] == opposite):
            return [right[0].text]

    # Column below the anchor — the address / GSTIN lines (also runs after an
    # inline name so "Bill To: X<br>addr<br>GSTIN" keeps X's address and GSTIN).
    lo, hi = anchor.x0, anchor.x1
    for below in seg_lines[anchor.line_index + 1:]:
        col = [s for s in below if _overlaps(s, lo, hi)]
        if not col:
            if lines:
                break
            continue
        stop = False
        for s in col:
            if marker_kind(s.text)[0] == opposite:
                stop = True
                break
            lines.append(s.text)
            lo, hi = min(lo, s.x0), max(hi, s.x1)
        if stop:
            break
    return lines


def _side_block(seg_lines: List[List[Segment]], pivot: float, side: str) -> Dict:
    """Parse the header block on one side of an x pivot — the *unlabelled* party
    (e.g. a supplier letterhead in the left column when only "Bill To" is
    labelled). Markers and title lines are excluded by parse_party_block."""
    lines = []
    for row in seg_lines:
        for s in row:
            on_side = s.x1 < pivot if side == "left" else s.x0 > pivot
            if on_side and not marker_kind(s.text)[0]:
                lines.append(s.text)
    return parse_party_block("\n".join(lines))


def _split_at_other_marker(text: str) -> str:
    cut = re.search(r"\s*[|,]?\s*\b(?:buyer|bill(?:ed)?\s*to|seller|supplier|consignee|sold\s*by)\b",
                    text, re.IGNORECASE)
    return text[: cut.start()].strip() if cut else text


def _detect_by_geometry(page: Dict) -> Dict:
    tokens = header_region(page)
    width = page.get("width", 0) or 1
    # Drop whole title lines ("TAX INVOICE") so a centred title straddling the
    # column divider cannot become a party name — but never a line that carries a
    # party marker (a "Supplier … Buyer" label row also reduces to "invoice").
    def _is_title_line(ln):
        if any(marker_kind(token_text(t))[0] for t in ln):
            return False
        return is_document_title(" ".join(token_text(t) for t in ln))
    lines = [ln for ln in group_lines(tokens) if not _is_title_line(ln)]
    seg_lines = [split_segments(line, width, i) for i, line in enumerate(lines)]
    flat = [s for row in seg_lines for s in row]

    seller_anchor = next((s for s in flat if marker_kind(s.text)[0] == "seller"), None)
    buyer_anchor = next((s for s in flat if marker_kind(s.text)[0] == "buyer"), None)
    seller = parse_party_block("\n".join(_block_lines_for_anchor(seller_anchor, "seller", seg_lines))) if seller_anchor else dict(_EMPTY)
    buyer = parse_party_block("\n".join(_block_lines_for_anchor(buyer_anchor, "buyer", seg_lines))) if buyer_anchor else dict(_EMPTY)

    # Only one party is labelled → the other is the block on the far side of it
    # (a supplier letterhead beside a "Bill To" buyer, and vice-versa).
    if buyer_anchor and not seller["name"]:
        seller = _side_block(seg_lines, buyer_anchor.x0, "left")
    elif seller_anchor and not buyer["name"]:
        buyer = _side_block(seg_lines, seller_anchor.x1, "right")

    return {"seller": seller, "buyer": buyer, "markers": bool(seller_anchor) + bool(buyer_anchor)}


# ── Strategy 3: text (inline embedded markers, stacked, HTML leak) ────────────

def _split_inline_parties(line: str) -> Dict[str, str]:
    """Extract seller/buyer from one line carrying inline markers, e.g.
    'Supplier: Acme | Buyer: Metro' → {'seller':'Acme','buyer':'Metro'}."""
    hits = []
    for phrase, kind in _ALL_PHRASES:
        for m in re.finditer(r"\b" + r"\s*".join(re.escape(w) for w in phrase.split()) + r"\b\s*[:;.\-]",
                             line, re.IGNORECASE):
            hits.append((m.start(), m.end(), kind))
    hits.sort()
    out: Dict[str, str] = {}
    for i, (_, end, kind) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(line)
        value = _TRAIL_PUNCT.sub("", line[end:stop]).strip(" |,-")
        if kind not in out and _is_value_text(value):
            out[kind] = value
    return out


def _detect_by_text(text: str) -> Dict:
    normalized = strip_html(text)
    lines = normalized.split("\n")
    seller = dict(_EMPTY)
    buyer = dict(_EMPTY)
    markers = 0

    for line in lines:
        inline = _split_inline_parties(line)
        if "seller" in inline and not seller["name"]:
            seller = parse_party_block(inline["seller"])
            markers += 1
        if "buyer" in inline and not buyer["name"]:
            buyer = parse_party_block(inline["buyer"])
            markers += 1

    # Stacked blocks: a marker heads a block that runs to the next marker.
    if not seller["name"] or not buyer["name"]:
        header = lines[: _find_body_start(lines)] if _find_body_start(lines) > 0 else lines
        idx = [(i, marker_kind(line)[0]) for i, line in enumerate(header) if marker_kind(line)[0]]
        for pos, (i, kind) in enumerate(idx):
            end = idx[pos + 1][0] if pos + 1 < len(idx) else len(header)
            block = parse_party_block("\n".join(header[i:end]))
            if block["name"] and not (seller if kind == "seller" else buyer)["name"]:
                if kind == "seller":
                    seller = block
                else:
                    buyer = block
                markers += 1
    return {"seller": seller, "buyer": buyer, "markers": markers}


_BODY_KEYWORDS = {"description", "particulars", "hsn", "sac", "qty", "quantity", "rate", "amount", "taxable", "item"}
_FOOTER = re.compile(r"^\s*(?:sub\s*-?\s*total|total|taxable|grand|cgst|sgst|igst|round|bank|terms|declaration|authori)", re.IGNORECASE)


def _find_body_start(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        if _FOOTER.match(line) and re.search(r"\d", line):
            return i
        tokens = {t for t in re.split(r"[^a-z]+", line.lower()) if t}
        if len(tokens & _BODY_KEYWORDS) >= 2:
            return i
    return -1


# ── Strategy 4: letterhead fallback (no markers at all) ───────────────────────

def _letterhead_seller(ocr: Dict) -> Dict:
    """The topmost non-title/non-meta text block — a store/company letterhead with
    no explicit "Supplier" label (POS receipts, plain bills)."""
    page = (ocr.get("pages") or [None])[0]
    lines = ["\n".join(token_text(t) for t in line) for line in group_lines(header_region(page))] if page else \
        strip_html(ocr.get("text") or "").split("\n")
    block = parse_party_block("\n".join(lines[:4]))
    return block


# ── Merge ─────────────────────────────────────────────────────────────────────

def _score(block: Dict) -> int:
    return 0 if not block["name"] else 2 + (1 if block["gstin"] else 0)


def _pick(*blocks: Dict) -> Dict:
    return max(blocks, key=_score)


def detect_parties(ocr: Dict) -> Dict:
    pages = ocr.get("pages") or []
    page = pages[0] if pages else None
    table = _detect_by_table(page) if page else {"seller": dict(_EMPTY), "buyer": dict(_EMPTY), "markers": 0}
    geo = _detect_by_geometry(page) if page else {"seller": dict(_EMPTY), "buyer": dict(_EMPTY), "markers": 0}
    txt = _detect_by_text(ocr.get("text") or "")

    seller = _pick(table["seller"], geo["seller"], txt["seller"])
    buyer = _pick(table["buyer"], geo["buyer"], txt["buyer"])

    had_markers = table["markers"] or geo["markers"] or txt["markers"]
    if not seller["name"] and not had_markers:
        seller = _letterhead_seller(ocr)

    # A standalone header GSTIN with no party of its own → seller then buyer.
    if not seller["gstin"] or not buyer["gstin"]:
        for g in find_gstins(strip_html(ocr.get("text") or "")):
            basics = derive_basics(g)
            if seller["name"] and not seller["gstin"]:
                seller.update(gstin=basics["gstNo"], stateCode=basics["stateCode"],
                              stateName=basics["stateName"], pan=basics["panNo"])
            elif buyer["name"] and not buyer["gstin"] and g != seller["gstin"]:
                buyer.update(gstin=basics["gstNo"], stateCode=basics["stateCode"],
                             stateName=basics["stateName"], pan=basics["panNo"])
    return {"seller": seller, "buyer": buyer}
