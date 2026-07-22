"""Deterministic pre-LLM structuring: reader output → grounded hints.

Before the LLM runs, mine the document for values that can be read verbatim with
high confidence — GSTINs, the invoice number/date, HSN codes, pincodes, the money
column and the most table-like grid. These :class:`RuleHints`:

  * ground the LLM prompt (it is told the exact GSTINs/number/date on the page,
    so it copies rather than hallucinates), and
  * give the validator an independent reference to cross-check the model against.

Pure functions over strings/tables — no model, fully unit-tested.
"""

from __future__ import annotations

import re

from app.domain.interfaces import RuleProcessor
from app.domain.pipeline_types import LayoutTable, ReaderOutput, RuleHints
from app.rules import gst
from app.rules.dates import normalize_date
from app.utils.numeric import to_float

_INVOICE_NO_RE = re.compile(
    r"(?:invoice|bill|inv|tax\s*invoice)\s*(?:no\.?|number|#|:)\s*[:#]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9/\-]{1,30})",
    re.IGNORECASE,
)
_DATE_LABEL_RE = re.compile(
    r"(?:invoice\s*date|bill\s*date|dated|date)\s*[:\-]?\s*"
    r"([0-9]{1,2}[/\-. ][A-Za-z0-9]{2,9}[/\-. ][0-9]{2,4}|[0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.IGNORECASE,
)
_PINCODE_RE = re.compile(r"\b([1-9][0-9]{5})\b")
_HSN_RE = re.compile(r"(?:hsn|sac)\s*(?:code)?\s*[:\-]?\s*([0-9]{4,8})", re.IGNORECASE)
_GRAND_TOTAL_RE = re.compile(
    r"(?:grand\s*total|amount\s*payable|net\s*payable|total\s*amount|invoice\s*total|bill\s*total)"
    r"[^0-9\-]*([0-9][0-9,]*\.?[0-9]*)",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\b|\b\d+\.\d{2}\b")

_LINE_HEADER_HINTS = ("description", "particular", "item", "product", "goods", "hsn", "qty", "quantity", "rate", "amount")


class RuleProcessorImpl(RuleProcessor):
    """Extract grounded hints from the reader output."""

    def process(self, reader_output: ReaderOutput) -> RuleHints:
        text = reader_output.text or reader_output.markdown
        hints = RuleHints()

        hints.gstins = gst.find_gstins(reader_output.markdown + "\n" + text)
        hints.invoice_number = _first_group(_INVOICE_NO_RE, text)
        hints.invoice_date = normalize_date(_first_group(_DATE_LABEL_RE, text))
        hints.pincodes = _unique(_PINCODE_RE.findall(text))
        hints.hsn_codes = _unique(m for m in _HSN_RE.findall(text))
        hints.grand_total = _max_money(_GRAND_TOTAL_RE.findall(text))
        hints.amounts = _money_values(text)

        line_table = _pick_line_item_table(reader_output.tables)
        if line_table is not None:
            hints.line_item_rows = line_table.rows
            hints.extra["lineItemColumns"] = _header_row(line_table)
        return hints


# -- helpers ------------------------------------------------------------------
def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else None


def _unique(values) -> list[str]:
    seen: list[str] = []
    for value in values:
        v = str(value).strip()
        if v and v not in seen:
            seen.append(v)
    return seen


def _money_values(text: str) -> list[float]:
    out: list[float] = []
    for token in _MONEY_RE.findall(text or ""):
        value = to_float(token)
        if value is not None and value > 0:
            out.append(value)
    return out


def _max_money(tokens) -> float | None:
    values = [v for v in (to_float(t) for t in tokens) if v is not None]
    return max(values) if values else None


def _pick_line_item_table(tables: list[LayoutTable]) -> LayoutTable | None:
    """The table most likely to be the line-item grid.

    Score each table by (a) a header row mentioning description/qty/rate/amount
    and (b) how many rows contain a money value — the line-item grid is the
    biggest, most numeric table with a recognisable header.
    """
    best: LayoutTable | None = None
    best_score = 0.0
    for table in tables:
        if table.n_rows < 2 or table.n_cols < 2:
            continue
        header = " ".join(c or "" for c in table.rows[0]).lower()
        header_hits = sum(1 for hint in _LINE_HEADER_HINTS if hint in header)
        numeric_rows = sum(
            1 for row in table.rows[1:] if any(to_float(c) not in (None, 0) for c in row)
        )
        score = header_hits * 2.0 + numeric_rows
        if score > best_score:
            best, best_score = table, score
    return best


def _header_row(table: LayoutTable) -> list[str]:
    return [(c or "").strip() for c in table.rows[0]] if table.rows else []
