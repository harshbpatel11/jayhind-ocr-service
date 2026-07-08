"""Reconstruct human reading order from positioned tokens.

OCR engines and PDF text layers both emit tokens in an arbitrary order. Invoices
are laid out in rows (a line item, a total, a label/value pair), so we cluster
tokens into visual lines by their vertical centre, then sort each line
left-to-right. Pure functions — no engine imports — so they are unit-testable.
"""
from statistics import median
from typing import Dict, List

from .config import LINE_TOLERANCE_RATIO


def _centre_y(token: Dict) -> float:
    _, y1, _, y2 = token["bbox"]
    return (y1 + y2) / 2.0


def _height(token: Dict) -> float:
    _, y1, _, y2 = token["bbox"]
    return abs(y2 - y1)


def group_into_lines(tokens: List[Dict]) -> List[List[Dict]]:
    """Cluster tokens into visual lines, each sorted left-to-right.

    The tolerance scales with the document's median token height, so it adapts
    to both a 200-DPI scan (tall tokens) and a vector PDF (short ones).
    """
    if not tokens:
        return []

    heights = [h for h in (_height(t) for t in tokens) if h > 0]
    tolerance = (median(heights) if heights else 1.0) * LINE_TOLERANCE_RATIO

    lines: List[List[Dict]] = []
    for token in sorted(tokens, key=_centre_y):
        centre = _centre_y(token)
        # A token joins the current line when its centre is within tolerance of
        # the line's own centre; tokens arrive top-down so only the last line
        # is a candidate.
        if lines and abs(centre - _centre_y(lines[-1][0])) <= tolerance:
            lines[-1].append(token)
        else:
            lines.append([token])

    for line in lines:
        line.sort(key=lambda t: t["bbox"][0])
    return lines


def tokens_to_text(tokens: List[Dict]) -> str:
    """Join tokens into reading-order text: spaces within a line, newlines between."""
    return "\n".join(
        " ".join(t["text"] for t in line if t["text"].strip())
        for line in group_into_lines(tokens)
    ).strip()
