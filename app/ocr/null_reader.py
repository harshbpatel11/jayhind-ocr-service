"""Deterministic reader with no model dependency.

Used (a) in unit tests, and (b) on hosts without the Paddle stack, so the whole
pipeline — loader → preprocess → layout → rules → extractor → validate → score —
is exercisable end-to-end with zero heavy dependencies.

It reads ONLY what is already text on the page: a digital PDF's embedded text
layer. A pure scan/photo yields empty text (there is nothing to recognise without
an OCR model), which the pipeline surfaces as an unreadable document.
"""

from __future__ import annotations

from app.domain.interfaces import DocumentReader
from app.domain.pipeline_types import (
    BlockType,
    LayoutBlock,
    PageImage,
    PageLayout,
    ReaderOutput,
)


class NullReader(DocumentReader):
    """Passthrough reader: uses embedded PDF text, recognises nothing."""

    engine = "null"

    def warm_up(self) -> None:  # nothing to load
        return None

    def is_ready(self) -> bool:
        return True

    def read(self, pages: list[PageImage]) -> ReaderOutput:
        page_layouts: list[PageLayout] = []
        for page in pages:
            text = page.text_layer.strip()
            blocks = (
                [LayoutBlock(kind=BlockType.TEXT, text=text, reading_order=0)]
                if text
                else []
            )
            page_layouts.append(
                PageLayout(
                    index=page.index,
                    width=page.width,
                    height=page.height,
                    markdown=text,
                    text=text,
                    blocks=blocks,
                )
            )
        markdown = "\n\n".join(p.markdown for p in page_layouts if p.markdown).strip()
        text = "\n\n".join(p.text for p in page_layouts if p.text).strip()
        return ReaderOutput(pages=page_layouts, markdown=markdown, text=text, engine=self.engine)
