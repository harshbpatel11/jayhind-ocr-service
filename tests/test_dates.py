"""Unit tests for invoice-date normalisation (day-first Indian format)."""

from __future__ import annotations

import pytest

from app.rules.dates import normalize_date


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("03/04/2026", "2026-04-03"),   # day-first
        ("3-4-26", "2026-04-03"),
        ("2026-04-03", "2026-04-03"),   # ISO passthrough
        ("15 Apr 2026", "2026-04-15"),
        ("15-April-2026", "2026-04-15"),
        ("04/13/2026", "2026-04-13"),   # month > 12 → mm/dd fallback swap
        ("Date: 01.02.2025", "2025-02-01"),
        ("", None),
        (None, None),
        ("not a date", None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected
