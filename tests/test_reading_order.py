"""Reading-order reconstruction — the one piece of engine-independent logic."""
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
