"""Invoice-date normalisation to ``YYYY-MM-DD``.

Indian invoices overwhelmingly print ``dd/mm/yyyy`` (day-first), so an ambiguous
``03/04/2026`` is read as 3 April, not 4 March. ISO input passes through
untouched; anything unparseable returns ``None`` (the field then stays blank for
the reviewer rather than being guessed).
"""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s*[-/ ]?\s*([A-Za-z]{3,9})\s*[-/ ,]?\s*(\d{2,4})\b"
)


def _year(raw: int) -> int:
    return raw + 2000 if raw < 100 else raw


def _safe(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(value: str | None) -> str | None:
    """Return ``YYYY-MM-DD`` for the first date-like token, else ``None``."""
    if not value:
        return None
    text = str(value).strip()

    iso = _ISO_RE.search(text)
    if iso:
        return _safe(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    textual = _TEXT_RE.search(text)
    if textual:
        month = _MONTHS.get(textual.group(2).lower()[:4]) or _MONTHS.get(textual.group(2).lower()[:3])
        if month:
            return _safe(_year(int(textual.group(3))), month, int(textual.group(1)))

    numeric = _NUMERIC_RE.search(text)
    if numeric:
        day, month, year = (int(numeric.group(1)), int(numeric.group(2)), _year(int(numeric.group(3))))
        if month > 12 and day <= 12:  # clearly mm/dd — swap
            day, month = month, day
        return _safe(year, month, day)
    return None
