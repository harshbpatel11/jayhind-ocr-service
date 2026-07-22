"""Shared fixtures + fakes for the test suite.

The heavy stages (PaddleOCR-VL, Qwen3-8B) are replaced by fakes so the ENTIRE
pipeline — loader → preprocess → layout → rules → extractor → validate → score —
is tested deterministically with zero model downloads.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.domain.interfaces import DocumentReader, InvoiceExtractor
from app.domain.pipeline_types import ExtractionContext, PageLayout, ReaderOutput

# A realistic PaddleOCR-VL markdown output: header key-values + a line-item table.
SAMPLE_MARKDOWN_INTRASTATE = """\
# TAX INVOICE

Sold By: ACME STEELS PRIVATE LIMITED
GSTIN: 24AJGPP6816J1ZY
Address: Plot 12, GIDC, Ahmedabad
Bill To: JAYHIND ENTERPRISES
GSTIN: 24AABCU9603R1ZX

Invoice No: INV-2026-0042
Invoice Date: 03/04/2026

| Sr | Description | HSN | Qty | Rate | Taxable | GST % | Amount |
|----|-------------|------|-----|------|---------|-------|--------|
| 1 | Steel Coil | 7208 | 10 | 5000 | 50000 | 18 | 59000 |
| 2 | Steel Rod | 7214 | 5 | 2000 | 10000 | 18 | 11800 |

Taxable Total: 60000
Total GST: 10800
Grand Total: 70800.00
"""

SAMPLE_MARKDOWN_INTERSTATE = SAMPLE_MARKDOWN_INTRASTATE.replace(
    "GSTIN: 24AABCU9603R1ZX", "GSTIN: 27AABCU9603R1ZX"
)


class FakeReader(DocumentReader):
    """A reader that returns a preset markdown document (stands in for VL)."""

    engine = "fake"

    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def warm_up(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def read(self, pages) -> ReaderOutput:
        page = PageLayout(index=0, width=1000, height=1400, markdown=self._markdown, text=self._markdown)
        return ReaderOutput(pages=[page], markdown=self._markdown, text=self._markdown, engine=self.engine)


class FakeLlmExtractor(InvoiceExtractor):
    """An extractor that returns a fixed raw dict (stands in for Qwen)."""

    name = "fake-llm"

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def warm_up(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def extract(self, context: ExtractionContext) -> dict:
        return self._payload


@pytest.fixture
def settings() -> Settings:
    return Settings(
        reader_engine="null",
        extractor_engine="rules",
        api_key="",
        preprocess_enabled=False,
    )
