"""Unit tests for deterministic pre-LLM hint extraction."""

from __future__ import annotations

from app.domain.pipeline_types import PageLayout, ReaderOutput
from app.layout.layout_analyzer import LayoutAnalyzerImpl
from app.rules.rule_processor import RuleProcessorImpl
from tests.conftest import SAMPLE_MARKDOWN_INTRASTATE


def _hints(markdown: str):
    page = PageLayout(index=0, width=1000, height=1400, markdown=markdown, text=markdown)
    output = ReaderOutput(pages=[page], markdown=markdown, text=markdown, engine="fake")
    output = LayoutAnalyzerImpl().analyze(output)
    return RuleProcessorImpl().process(output)


def test_extracts_core_hints():
    hints = _hints(SAMPLE_MARKDOWN_INTRASTATE)
    assert hints.gstins == ["24AJGPP6816J1ZY", "24AABCU9603R1ZX"]
    assert hints.invoice_number == "INV-2026-0042"
    assert hints.invoice_date == "2026-04-03"
    assert hints.grand_total == 70800.0


def test_finds_line_item_table():
    hints = _hints(SAMPLE_MARKDOWN_INTRASTATE)
    # header + 2 rows
    assert len(hints.line_item_rows) == 3
    assert "Description" in (hints.extra.get("lineItemColumns") or [])
