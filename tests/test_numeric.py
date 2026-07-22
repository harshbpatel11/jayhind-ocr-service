"""Unit tests for numeric coercion helpers."""

from __future__ import annotations

import pytest

from app.utils.numeric import digits_only, round2, round3, to_float, to_money


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1,234.50", 1234.50),
        ("₹ 1234.5", 1234.5),
        ("Rs 2,00,000", 200000.0),
        ("(500)", -500.0),
        ("18%", 18.0),
        (1234, 1234.0),
        (12.345, 12.35),
        (None, None),
        ("", None),
        ("n/a", None),
        (True, None),  # bool must not be treated as a number
    ],
)
def test_to_float(value, expected):
    assert to_float(value) == expected


def test_to_money_never_none():
    assert to_money(None) == 0.0
    assert to_money("abc", 7.0) == 7.0
    assert to_money("1,000.00") == 1000.0


def test_round2_half_up():
    assert round2(2.005) == 2.01
    assert round2(-2.005) == -2.01
    assert round2(0.1 + 0.2) == 0.3


def test_round3():
    assert round3(1.2345) == 1.234 or round3(1.2345) == 1.235


@pytest.mark.parametrize(
    "value,expected",
    [("+91 98765-43210", "919876543210"), ("380015", "380015"), (None, None), ("--", None)],
)
def test_digits_only(value, expected):
    assert digits_only(value) == expected
