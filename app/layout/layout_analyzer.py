"""Turn the reader's reading-order markdown into typed layout blocks.

PaddleOCR-VL already recovers reading order and emits GitHub-flavoured markdown,
so this stage is deterministic post-processing: split the markdown into blocks,
classify each (title / table / key-value / text), and parse pipe-tables into a
dense grid (merged cells become repeated/blank neighbours). The structured tables
are what the rule processor mines for the line-item grid, and what grounds the
LLM prompt.

Pure string work — no model, fully unit-tested.
"""

from __future__ import annotations

import re

from app.domain.interfaces import LayoutAnalyzer
from app.domain.pipeline_types import (
    BlockType,
    LayoutBlock,
    LayoutTable,
    PageLayout,
    ReaderOutput,
)

_KV_RE = re.compile(r"^[^:|]{1,40}:\s*\S")


class LayoutAnalyzerImpl(LayoutAnalyzer):
    """Deterministic markdown → typed blocks + parsed tables."""

    def analyze(self, reader_output: ReaderOutput) -> ReaderOutput:
        for page in reader_output.pages:
            # Only (re)build blocks the reader left unstructured (e.g. VL markdown,
            # or the null reader's single text block). A reader that already
            # produced rich blocks is left untouched.
            if _needs_reblocking(page):
                page.blocks = _blocks_from_markdown(page.markdown)
        return reader_output


def _needs_reblocking(page: PageLayout) -> bool:
    return bool(page.markdown) and (
        not page.blocks or all(b.table is None for b in page.blocks)
    )


def _blocks_from_markdown(markdown: str) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    order = 0
    lines = markdown.splitlines()
    i = 0
    n = len(lines)
    buffer: list[str] = []

    def flush_text() -> None:
        nonlocal order
        text = "\n".join(buffer).strip()
        buffer.clear()
        if not text:
            return
        for chunk in _split_paragraphs(text):
            kind = _classify(chunk)
            blocks.append(LayoutBlock(kind=kind, text=chunk, reading_order=order))
            _bump()

    def _bump() -> None:
        nonlocal order
        order += 1

    while i < n:
        line = lines[i]
        if _is_table_row(line):
            flush_text()
            table_lines, i = _consume_table(lines, i)
            table = _parse_markdown_table(table_lines)
            blocks.append(
                LayoutBlock(
                    kind=BlockType.TABLE,
                    text="\n".join(table_lines).strip(),
                    reading_order=order,
                    table=table,
                )
            )
            _bump()
            continue
        buffer.append(line)
        i += 1
    flush_text()
    return blocks


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _classify(chunk: str) -> BlockType:
    first = chunk.splitlines()[0].strip()
    if first.startswith("#"):
        return BlockType.TITLE
    if len(chunk) <= 60 and chunk.isupper() and any(c.isalpha() for c in chunk):
        return BlockType.TITLE
    if all(_KV_RE.match(line.strip()) for line in chunk.splitlines() if line.strip()):
        return BlockType.KEY_VALUE
    return BlockType.TEXT


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    """A markdown table separator like ``|----|:--:|----|`` (dashes/colons only)."""
    body = line.replace("|", "").replace(" ", "")
    return bool(body) and set(body) <= {"-", ":"}


def _consume_table(lines: list[str], start: int) -> tuple[list[str], int]:
    out: list[str] = []
    i = start
    while i < len(lines) and _is_table_row(lines[i]):
        out.append(lines[i])
        i += 1
    return out, i


def _parse_markdown_table(table_lines: list[str]) -> LayoutTable:
    rows: list[list[str | None]] = []
    for line in table_lines:
        if _is_separator_row(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append([c if c else None for c in cells])
    width = max((len(r) for r in rows), default=0)
    for row in rows:  # pad ragged rows so downstream indexing is safe
        row.extend([None] * (width - len(row)))
    return LayoutTable(rows=rows)
