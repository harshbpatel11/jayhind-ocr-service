"""Unit tests for markdown → typed layout blocks + table parsing."""

from __future__ import annotations

from app.domain.pipeline_types import BlockType, PageLayout, ReaderOutput
from app.layout.layout_analyzer import LayoutAnalyzerImpl
from tests.conftest import SAMPLE_MARKDOWN_INTRASTATE


def _analyze(markdown: str) -> ReaderOutput:
    page = PageLayout(index=0, width=1000, height=1400, markdown=markdown, text=markdown)
    output = ReaderOutput(pages=[page], markdown=markdown, text=markdown, engine="fake")
    return LayoutAnalyzerImpl().analyze(output)


def test_parses_a_table():
    output = _analyze(SAMPLE_MARKDOWN_INTRASTATE)
    tables = output.tables
    assert len(tables) == 1
    table = tables[0]
    # header + 2 data rows
    assert table.n_rows == 3
    assert table.rows[0][1] == "Description"
    assert table.rows[1][1] == "Steel Coil"
    assert table.rows[2][3] == "5"  # Qty of second line


def test_classifies_title():
    output = _analyze("# TAX INVOICE\n\nSome body text here.")
    kinds = {b.kind for b in output.pages[0].blocks}
    assert BlockType.TITLE in kinds


def test_ragged_rows_are_padded():
    md = "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n"
    output = _analyze(md)
    table = output.tables[0]
    assert all(len(row) == table.n_cols for row in table.rows)
