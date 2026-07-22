"""Internal pipeline value objects.

These dataclasses carry data BETWEEN pipeline stages (loader → preprocessor →
reader → layout → rules → extractor → validator → scorer). They are deliberately
separate from ``models.py`` (the external wire contract): a stage boundary should
be free to evolve without touching the ERP-facing JSON.

Everything here is a plain, immutable-ish dataclass with type hints — no I/O, no
framework types — so each stage is trivially unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SourceKind(str, Enum):
    """How the upload was decoded."""

    IMAGE = "image"
    PDF_SCAN = "pdf-scan"
    PDF_TEXT = "pdf-text"


@dataclass(slots=True)
class PageImage:
    """One rasterised page ready for the reader.

    ``image`` is an RGB ``uint8`` ndarray (H, W, 3). ``text_layer`` is the
    embedded text for a digital PDF page (empty for a scan/photo), which lets the
    pipeline skip OCR on pages that already carry selectable text.
    """

    index: int
    image: np.ndarray
    dpi: int
    source: SourceKind
    text_layer: str = ""

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def has_text_layer(self) -> bool:
        return bool(self.text_layer.strip())


@dataclass(slots=True)
class LoadedDocument:
    """Result of ingestion: every page as an image, plus provenance."""

    pages: list[PageImage]
    #: ``pdf-text`` when EVERY page carried a usable embedded text layer, else
    #: ``ocr`` — mirrors the contract's ``ParseResult.method``.
    method: str

    @property
    def page_count(self) -> int:
        return len(self.pages)


class BlockType(str, Enum):
    """Layout-block roles recovered by the reader / layout analyzer."""

    TITLE = "title"
    TEXT = "text"
    TABLE = "table"
    KEY_VALUE = "key_value"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


@dataclass(slots=True)
class TableCell:
    """A single (possibly merged) table cell."""

    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


@dataclass(slots=True)
class LayoutTable:
    """A detected table as a dense grid plus its merged-cell detail."""

    rows: list[list[str | None]]
    cells: list[TableCell] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)


@dataclass(slots=True)
class LayoutBlock:
    """One reading-order block on a page."""

    kind: BlockType
    text: str
    reading_order: int
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0
    table: LayoutTable | None = None


@dataclass(slots=True)
class PageLayout:
    """The reader's structured view of one page."""

    index: int
    width: int
    height: int
    markdown: str
    text: str
    blocks: list[LayoutBlock] = field(default_factory=list)

    @property
    def tables(self) -> list[LayoutTable]:
        return [b.table for b in self.blocks if b.table is not None]


@dataclass(slots=True)
class ReaderOutput:
    """The reader's structured view of the whole document."""

    pages: list[PageLayout]
    #: Full reading-order markdown across pages (tables preserved).
    markdown: str
    #: Full reading-order plain text across pages.
    text: str
    #: Which reader produced this (``paddleocr-vl``, ``pdf-text``, ``null`` …).
    engine: str = "unknown"

    @property
    def tables(self) -> list[LayoutTable]:
        out: list[LayoutTable] = []
        for page in self.pages:
            out.extend(page.tables)
        return out


@dataclass(slots=True)
class RuleHints:
    """Deterministic candidates recovered from the layout BEFORE the LLM runs.

    These are *hints*, not answers: they seed the LLM prompt (grounding it in
    values that were read verbatim from the page) and let the validator
    cross-check the model's output. Every field is optional and may be wrong —
    the LLM and the business-rule validator have the final say.
    """

    gstins: list[str] = field(default_factory=list)
    invoice_number: str | None = None
    invoice_date: str | None = None
    pincodes: list[str] = field(default_factory=list)
    hsn_codes: list[str] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)
    grand_total: float | None = None
    #: Raw table rows most likely to be the line-item grid.
    line_item_rows: list[list[str | None]] = field(default_factory=list)
    #: Free-form extras keyed for downstream consumers/tests.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionContext:
    """Everything the extractor + validator need, bundled for DI."""

    reader: ReaderOutput
    hints: RuleHints
    method: str
    page_count: int
