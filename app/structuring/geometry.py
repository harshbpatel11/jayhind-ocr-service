"""Token geometry: visual lines, x-gap segments and column bands.

Everything downstream that needs to understand *where* text sits on the page
(party columns, table columns, right-aligned totals) is built on these. Pure
functions over the token list the extractor produces
(`{"text", "bbox":[x1,y1,x2,y2], "confidence"}`).
"""
from statistics import median
from typing import Dict, List, Optional

from .text import strip_html


def x0(t: Dict) -> float: return t["bbox"][0]
def y0(t: Dict) -> float: return t["bbox"][1]
def x1(t: Dict) -> float: return t["bbox"][2]
def y1(t: Dict) -> float: return t["bbox"][3]
def cx(t: Dict) -> float: return (t["bbox"][0] + t["bbox"][2]) / 2
def cy(t: Dict) -> float: return (t["bbox"][1] + t["bbox"][3]) / 2
def height(t: Dict) -> float: return abs(t["bbox"][3] - t["bbox"][1])


def token_text(t: Dict) -> str:
    """A token's visible text with any leaked HTML markup removed (tags → space,
    so a GSTIN glued behind `Estate<br/>` still scans)."""
    return strip_html(t["text"], br_to_newline=False).strip()


def group_lines(tokens: List[Dict]) -> List[List[Dict]]:
    """Cluster tokens into visual lines (top→bottom), each sorted left→right.

    Tolerance scales with the median token height, so it adapts to a 300-DPI scan
    (tall tokens) and a vector PDF (short ones) alike.
    """
    tokens = [t for t in tokens if token_text(t)]
    if not tokens:
        return []
    heights = sorted(h for h in (height(t) for t in tokens) if h > 0)
    tolerance = (heights[len(heights) // 2] if heights else 1.0) * 0.6

    lines: List[List[Dict]] = []
    for token in sorted(tokens, key=cy):
        if lines and abs(cy(token) - cy(lines[-1][0])) <= tolerance:
            lines[-1].append(token)
        else:
            lines.append([token])
    for line in lines:
        line.sort(key=x0)
    return lines


class Segment:
    """A run of tokens on one visual line with no large horizontal gap between
    them — i.e. one logical cell ("Supplier", or "Acme Pumps Pvt Ltd")."""

    __slots__ = ("tokens", "text", "x0", "x1", "cx", "line_index")

    def __init__(self, tokens: List[Dict], line_index: int):
        self.tokens = tokens
        self.text = " ".join(token_text(t) for t in tokens).strip()
        self.x0 = min(x0(t) for t in tokens)
        self.x1 = max(x1(t) for t in tokens)
        self.cx = (self.x0 + self.x1) / 2
        self.line_index = line_index


def split_segments(line: List[Dict], page_width: float, line_index: int) -> List[Segment]:
    """Split a visual line into segments wherever a wide horizontal gap appears.

    Within-word gaps are a couple of points; the gap between two header columns is
    tens of points, so a threshold at ~3.5% of the page width separates
    "Supplier | Invoice | Buyer" into three segments while keeping "Acme Pumps
    Pvt Ltd" whole.
    """
    if not line:
        return []
    gap = max(page_width * 0.035, 12.0)
    segments: List[List[Dict]] = [[line[0]]]
    for token in line[1:]:
        if x0(token) - x1(segments[-1][-1]) > gap:
            segments.append([token])
        else:
            segments[-1].append(token)
    return [Segment(seg, line_index) for seg in segments]


def cluster_values(values: List[float], gap: float) -> List[float]:
    """1-D clustering: sort, split where the gap exceeds `gap`, return cluster
    centres. Used to find column bands from segment x-centres."""
    if not values:
        return []
    ordered = sorted(values)
    clusters: List[List[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if v - clusters[-1][-1] > gap:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [sum(c) / len(c) for c in clusters]


def find_table_top(tokens: List[Dict]) -> Optional[float]:
    """The y of the line-items table header row, if one is present.

    The header is the topmost visual line carrying two or more column-keyword
    tokens (Item/Description/Qty/Rate/Amount/HSN…). Everything above it is the
    party/invoice header region.
    """
    import re
    keyword = re.compile(
        r"^(item|description|particular|particulars|goods|product|desc|sr|s\.?no|hsn|sac|"
        r"qty|quantity|unit|uom|rate|price|mrp|amount|amt|value|gst|igst|cgst|tax|disc|discount)\b",
        re.IGNORECASE,
    )
    for line in group_lines(tokens):
        hits = sum(1 for t in line if keyword.match(token_text(t)))
        if hits >= 2:
            return min(y0(t) for t in line)
    return None


def header_region(page: Dict) -> List[Dict]:
    """Tokens that sit above the line-items table (the party/invoice header)."""
    tokens = page.get("tokens") or []
    top = find_table_top(tokens)
    if top is None:
        top = page.get("height", 0) * 0.45
    return [t for t in tokens if y1(t) <= top]
