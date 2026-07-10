"""Pure text / number / date helpers for invoice structuring.

Ports the dependency-free primitives that used to live in the TypeScript parser
(`invoice-parsing.const.ts`). No I/O, no geometry — unit-tested in
`tests/test_structuring.py`.
"""
import re
from datetime import date
from typing import Optional

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_AMOUNT_CLEAN = re.compile(r"[₹$€]|\bRs\.?|\bINR\b", re.IGNORECASE)


def round2(n: float) -> float:
    """Round to 2dp the way money is rounded (mirrors posting.const `round2`)."""
    return round(n + 1e-9, 2)


def normalize_amount(raw) -> Optional[float]:
    """Strip currency symbols/commas/spaces → float. None when not numeric.

    Repairs the common OCR slip of reading a thousands comma as a period
    ("5.058.00"): a real amount never has two decimal points, so the last dot is
    kept as the decimal and the rest dropped.
    """
    if raw is None:
        return None
    s = _AMOUNT_CLEAN.sub("", str(raw).strip()).strip()
    negated = bool(re.fullmatch(r"\(.*\)", s))
    if negated:
        s = s[1:-1].strip()
    s = re.sub(r"[,\s]", "", s)
    if s.count(".") > 1:
        cut = s.rfind(".")
        s = s[:cut].replace(".", "") + s[cut:]
    if not s or not re.fullmatch(r"-?\d*\.?\d+", s):
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negated else value


def _expand_year(year: int) -> int:
    if year >= 1000:
        return year
    return 2000 + year if year < 70 else 1900 + year


def _build_date(year: int, month: int, day: int) -> Optional[str]:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    try:
        date(year, month, day)  # rejects overflow like 31-02
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_date(raw) -> Optional[str]:
    """Parse the date formats Indian invoices use → `yyyy-MM-dd`.

    Day-first is assumed for ambiguous numeric dates (the Indian convention).
    """
    if not raw:
        return None
    s = str(raw).strip()

    # Drop ordinal suffixes: "5th July" → "5 July".
    s = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)

    iso = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if iso:
        return _build_date(int(iso[1]), int(iso[2]), int(iso[3]))

    # Day first: "05-Jul-2026" / "5 July 2026".
    named = re.search(r"\b(\d{1,2})[-/.\s]([A-Za-z]{3,9})[-/.\s,]*(\d{2,4})\b", s)
    if named:
        month = MONTHS.get(named[2].lower())
        if month:
            return _build_date(_expand_year(int(named[3])), month, int(named[1]))

    # Month first: "July 5, 2026" / "Jul 5 2026".
    month_first = re.search(r"\b([A-Za-z]{3,9})[-/.\s]+(\d{1,2})[-/.\s,]+(\d{2,4})\b", s)
    if month_first:
        month = MONTHS.get(month_first[1].lower())
        if month:
            return _build_date(_expand_year(int(month_first[3])), month, int(month_first[2]))

    numeric = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", s)
    if numeric:
        return _build_date(_expand_year(int(numeric[3])), int(numeric[2]), int(numeric[1]))

    return None


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — enough for one-word OCR slips."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[len(b)]


_HTML_BR = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")


def strip_html(text: str, br_to_newline: bool = True) -> str:
    """Remove stray HTML markup that leaks into some invoice PDFs as literal text.

    `<br>` becomes a newline (it is a real line break in the source layout) so a
    name/address/GSTIN that got glued onto one visual line is separated again;
    every other tag is dropped.
    """
    if "<" not in text:
        return text
    out = _HTML_BR.sub("\n" if br_to_newline else " ", text)
    out = _HTML_TAG.sub("", out)
    return out


LEADING_PUNCTUATION = re.compile(r"^[\s:;.,\-|]+")
TRAILING_PUNCTUATION = re.compile(r"[\s:;,\-|]+$")


def clean_name(value: str) -> str:
    """Tidy a name without discarding the punctuation that is part of it
    ("Shree Trading Co.", "Pvt. Ltd.")."""
    value = LEADING_PUNCTUATION.sub("", value)
    value = TRAILING_PUNCTUATION.sub("", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value[:150]
