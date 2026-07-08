"""Reading-order reconstruction and de-skew — engine-independent geometry."""
import math

import pytest

from app.extractor import _estimate_skew, _polygons_to_tokens
from app.reading_order import group_into_lines, tokens_to_text


def token(text, x1, y1, x2=None, y2=None):
    return {"text": text, "bbox": [x1, y1, x2 if x2 is not None else x1 + 20, y2 if y2 is not None else y1 + 10], "confidence": 1.0}


def test_empty_input():
    assert group_into_lines([]) == []
    assert tokens_to_text([]) == ""


def test_tokens_sort_top_to_bottom_then_left_to_right():
    # Deliberately shuffled: bottom-right first.
    tokens = [token("World", 40, 30), token("Hello", 10, 30), token("Invoice", 10, 10)]
    assert tokens_to_text(tokens) == "Invoice\nHello World"


def test_tokens_on_the_same_visual_line_are_grouped_despite_baseline_jitter():
    # +2px jitter is well within tolerance (0.6 x median height of 10).
    tokens = [token("GSTIN", 10, 50), token("24AAACT2727Q1ZW", 60, 52)]
    lines = group_into_lines(tokens)
    assert len(lines) == 1
    assert tokens_to_text(tokens) == "GSTIN 24AAACT2727Q1ZW"


def test_distinct_rows_stay_separate():
    tokens = [token("Rod 12mm", 10, 100), token("Sheet 2mm", 10, 130)]
    assert len(group_into_lines(tokens)) == 2


def test_blank_tokens_are_dropped_from_text():
    tokens = [token("Total", 10, 10), token("   ", 50, 10), token("1180.00", 90, 10)]
    assert tokens_to_text(tokens) == "Total 1180.00"


# ── De-skew (rotated photos) ─────────────────────────────────────────────────


def poly(x0, y0, w, h, angle_deg=0.0, cx=500.0, cy=500.0):
    """Axis-aligned quad [tl, tr, br, bl], optionally rotated about (cx, cy)."""
    pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    if not angle_deg:
        return pts
    a = math.radians(angle_deg)
    return [
        ((x - cx) * math.cos(a) - (y - cy) * math.sin(a) + cx,
         (x - cx) * math.sin(a) + (y - cy) * math.cos(a) + cy)
        for x, y in pts
    ]


def test_estimate_skew_detects_rotation():
    polys = [poly(100, 100 + i * 40, 400, 20, angle_deg=3) for i in range(5)]
    assert math.degrees(_estimate_skew(polys)) == pytest.approx(3, abs=0.3)


def test_estimate_skew_is_zero_for_square_pages():
    polys = [poly(100, 100 + i * 40, 400, 20) for i in range(5)]
    assert _estimate_skew(polys) == pytest.approx(0, abs=1e-6)


def test_estimate_skew_ignores_short_boxes():
    polys = [poly(100, 100, 2, 20), poly(100, 200, 400, 20)]  # 2px-wide box gives no angle
    assert _estimate_skew(polys) == pytest.approx(0, abs=1e-6)


def test_deskew_realigns_a_rotated_line_onto_one_row():
    # One text line split into three boxes, rotated 3° — raw y centres differ by
    # ~20px across the page, more than a 20px line height. After de-skew they align.
    polys = [poly(100 + i * 200, 300, 180, 20, angle_deg=3) for i in range(3)]
    tokens = _polygons_to_tokens(["a", "b", "c"], [0.9, 0.9, 0.9], polys, (1000.0, 1000.0))
    centres = [(t["bbox"][1] + t["bbox"][3]) / 2 for t in tokens]
    assert max(centres) - min(centres) < 3  # within a fraction of a line height
    assert len(group_into_lines(tokens)) == 1
