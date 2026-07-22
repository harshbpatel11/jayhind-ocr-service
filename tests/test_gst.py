"""Unit tests for the GST domain rules."""

from __future__ import annotations

import pytest

from app.rules import gst

SELLER = "24AJGPP6816J1ZY"  # Gujarat
BUYER_SAME = "24AABCU9603R1ZX"  # Gujarat
BUYER_OTHER = "27AABCU9603R1ZX"  # Maharashtra


def test_normalize_and_validate():
    assert gst.normalize_gstin(" 24ajgpp6816j1zy ") == SELLER
    assert gst.is_valid_gstin(SELLER)
    assert not gst.is_valid_gstin("24AJGPP6816J1Z")  # 14 chars
    assert not gst.is_valid_gstin("99XXXXX0000X0X0")  # wrong shape


def test_state_pan_derivation():
    assert gst.state_code_of(SELLER) == "24"
    assert gst.state_name_of(SELLER) == "Gujarat"
    assert gst.pan_of(SELLER) == "AJGPP6816J"
    assert gst.state_name_of(BUYER_OTHER) == "Maharashtra"


def test_inter_state():
    assert gst.is_inter_state(SELLER, BUYER_OTHER) is True
    assert gst.is_inter_state(SELLER, BUYER_SAME) is False
    assert gst.is_inter_state(SELLER, None) is None  # unknown → undecided


def test_find_gstins_in_text():
    text = f"Seller {SELLER} sold to buyer {BUYER_OTHER}. Repeat {SELLER}."
    assert gst.find_gstins(text) == [SELLER, BUYER_OTHER]


@pytest.mark.parametrize("raw,snapped", [(17.9, 18.0), (18.0, 18.0), (4.8, 5.0), (28.4, 28.0), (13.0, 13.0)])
def test_nearest_rate_slab(raw, snapped):
    assert gst.nearest_rate_slab(raw) == snapped
