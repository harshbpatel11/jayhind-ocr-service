"""Numeric coercion helpers shared across rules, validation and scoring.

LLMs and OCR return money as messy strings: ``"1,234.50"``, ``"₹ 1234.5"``,
``"(500)"``, ``"18%"``, ``None``. These helpers turn that into clean floats the
way the ERP rounds money (2 dp), so every stage agrees to the paisa.
"""

from __future__ import annotations

import re
from typing import Any

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def round2(value: float) -> float:
    """Round to 2 dp the way the ERP rounds money (``Math.round`` equivalent)."""
    return float(int((value * 100) + (0.5 if value >= 0 else -0.5)) / 100.0)


def round3(value: float) -> float:
    """Round to 3 dp (quantities)."""
    return float(int((value * 1000) + (0.5 if value >= 0 else -0.5)) / 1000.0)


def to_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort parse of a money/number-ish value.

    Handles thousands separators, currency symbols, percent signs, and
    accountant-style ``(1,234)`` negatives. Returns ``default`` for the
    unparseable / empty / ``None`` cases.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):  # guard: bool is an int subclass
        return default
    if isinstance(value, (int, float)):
        return round2(float(value))
    text = str(value).strip()
    if not text:
        return default
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.replace(",", "").replace("₹", "").replace("Rs", "").replace("%", "")
    match = _NUMERIC_RE.search(cleaned)
    if not match:
        return default
    try:
        number = float(match.group(0))
    except ValueError:
        return default
    if negative:
        number = -abs(number)
    return round2(number)


def to_money(value: Any, default: float = 0.0) -> float:
    """Like :func:`to_float` but never ``None`` — for required money fields."""
    parsed = to_float(value, None)
    return default if parsed is None else parsed


def digits_only(value: Any) -> str | None:
    """Strip everything but digits (phone numbers, pincodes). ``None`` if empty."""
    if value is None:
        return None
    out = re.sub(r"\D", "", str(value))
    return out or None
