"""Unit tests for reading-order recovery (shared by detection-based readers)."""

from __future__ import annotations

from app.ocr.reading_order import poly_top_left, reading_order_text


def test_poly_top_left_from_points():
    poly = [[100, 50], [200, 52], [200, 70], [100, 68]]
    y, x = poly_top_left(poly)
    assert y == 50 and x == 100


def test_poly_top_left_from_flat_box():
    assert poly_top_left([10, 20, 40, 35]) == (20.0, 10.0)


def test_reading_order_groups_rows_and_sorts():
    # Two rows; tokens given out of order, close y = same row.
    lines = [
        (100.0, 300.0, "World"),
        (102.0, 50.0, "Hello"),
        (40.0, 10.0, "TITLE"),
    ]
    assert reading_order_text(lines) == "TITLE\nHello  World"


def test_reading_order_empty():
    assert reading_order_text([]) == ""
