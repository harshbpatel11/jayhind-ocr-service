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


def test_extracts_explicit_due_date():
    hints = _hints(SAMPLE_MARKDOWN_INTRASTATE + "\nDue Date: 15/05/2026\n")
    assert hints.due_date == "2026-05-15"
    assert hints.payment_terms_days is None


def test_extracts_payment_terms_days_without_a_date():
    # "Payment Due: 30 Days" must NOT be mistaken for a date — only the days count.
    hints = _hints(SAMPLE_MARKDOWN_INTRASTATE + "\nPayment Due: 30 Days\n")
    assert hints.payment_terms_days == 30
    assert hints.due_date is None


def test_extracts_net_terms():
    hints = _hints(SAMPLE_MARKDOWN_INTRASTATE + "\nTerms: Net 45\n")
    assert hints.payment_terms_days == 45
