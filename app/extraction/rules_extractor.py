"""Deterministic, model-free extractor.

Builds the raw invoice dict purely from the rule hints + the detected line-item
table. It is intentionally modest — no NLP, no model — so it serves three roles:

  1. a graceful fallback when the LLM is unavailable or errors,
  2. the extractor used for ``OCR_EXTRACTOR_ENGINE=rules`` deployments, and
  3. the engine that lets the whole pipeline be unit-tested with zero heavy
     dependencies.

The validator + confidence scorer run over its output exactly as they do over the
LLM's, so the contract shape and totals reconciliation are identical.
"""

from __future__ import annotations

from app.domain.interfaces import InvoiceExtractor
from app.domain.pipeline_types import ExtractionContext
from app.utils.numeric import to_float

# Column-header keyword → canonical line field.
_COLUMN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("hsnSac", ("hsn", "sac")),
    ("quantity", ("qty", "quantity")),
    ("unit", ("unit", "uom")),
    ("gstRate", ("gst %", "gst%", "tax %", "gst rate", "igst %", "cgst %")),
    ("discount", ("disc", "discount")),
    ("rate", ("rate", "price", "mrp")),
    ("taxableAmount", ("taxable", "net amount", "net value", "amount", "value")),
    ("lineTotal", ("total",)),
    ("description", ("description", "particular", "item", "product", "goods", "name")),
]


class RulesExtractor(InvoiceExtractor):
    """Model-free extractor from grounded hints + the line-item table."""

    name = "rules"

    def warm_up(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def extract(self, context: ExtractionContext) -> dict:
        hints = context.hints
        gstins = hints.gstins
        return {
            "seller": {"name": "", "gstin": gstins[0] if len(gstins) >= 1 else None},
            "buyer": {"name": "", "gstin": gstins[1] if len(gstins) >= 2 else None},
            "invoice": {"number": hints.invoice_number, "date": hints.invoice_date},
            "lineItems": _rows_to_line_items(hints.line_item_rows),
            "taxSummary": [],
            "totals": {"grandTotal": hints.grand_total},
        }


def _map_columns(header: list[str | None]) -> dict[int, str]:
    """Map each header column index to a canonical field (first keyword wins)."""
    mapping: dict[int, str] = {}
    used: set[str] = set()
    lowered = [(i, (h or "").strip().lower()) for i, h in enumerate(header)]
    for field, keywords in _COLUMN_KEYWORDS:
        if field in used:
            continue
        for i, text in lowered:
            if i in mapping or not text:
                continue
            if any(k in text for k in keywords):
                mapping[i] = field
                used.add(field)
                break
    return mapping


def _rows_to_line_items(rows: list[list[str | None]]) -> list[dict]:
    if not rows or len(rows) < 2:
        return []
    mapping = _map_columns(rows[0])
    if "description" not in mapping.values() and "taxableAmount" not in mapping.values():
        return []
    items: list[dict] = []
    for row in rows[1:]:
        item: dict = {}
        for i, value in enumerate(row):
            field = mapping.get(i)
            if not field or value is None:
                continue
            item[field] = value.strip() if isinstance(value, str) else value
        # Skip summary/footer rows that carry no description and no positive money.
        has_desc = bool(str(item.get("description", "")).strip())
        has_money = to_float(item.get("taxableAmount")) not in (None, 0) or to_float(item.get("rate")) not in (None, 0)
        if has_desc or has_money:
            items.append(item)
    return items
