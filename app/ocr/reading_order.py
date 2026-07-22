"""Reading-order recovery shared by the detection+recognition readers.

Detection-based OCR (PP-OCR via Paddle or ONNX) returns text lines with bounding
boxes but no order. These helpers group boxes into rows by vertical proximity,
then order rows top→down and tokens left→right — recovering the page's reading
order for the layout analyzer, rule hints and the LLM.
"""

from __future__ import annotations


def poly_top_left(poly) -> tuple[float, float]:
    """Top-y, left-x of a polygon (4 points) or a flat ``[x1,y1,x2,y2]`` box."""
    try:
        pts = list(poly)
        if pts and hasattr(pts[0], "__len__"):
            ys = [float(p[1]) for p in pts]
            xs = [float(p[0]) for p in pts]
            return min(ys), min(xs)
        return float(pts[1]), float(pts[0])
    except Exception:
        return 0.0, 0.0


def reading_order_text(lines: list[tuple[float, float, str]], line_tol: float = 12.0) -> str:
    """Order ``(y, x, text)`` triples into reading-order text.

    Tokens whose top-y is within ``line_tol`` px are treated as one row.
    """
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda t: (t[0], t[1]))
    rows: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    current_y: float | None = None
    for y, x, text in ordered:
        if current_y is None or abs(y - current_y) <= line_tol:
            current.append((y, x, text))
            current_y = y if current_y is None else (current_y + y) / 2.0
        else:
            rows.append(current)
            current = [(y, x, text)]
            current_y = y
    if current:
        rows.append(current)
    return "\n".join(
        "  ".join(text for _, _, text in sorted(row, key=lambda t: t[1])) for row in rows
    ).strip()
