"""End-to-end pipeline tests with fake reader/extractor (no heavy models)."""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

from app.confidence.scorer import ConfidenceScorerImpl
from app.config import Settings
from app.domain.interfaces import InvoiceExtractor
from app.domain.pipeline_types import ExtractionContext
from app.extraction.rules_extractor import RulesExtractor
from app.ingestion.document_loader import DocumentLoaderImpl
from app.layout.layout_analyzer import LayoutAnalyzerImpl
from app.persistence.artifact_repository import NullArtifactRepository
from app.pipeline.errors import RetryableEngineError
from app.pipeline.orchestrator import OcrPipeline
from app.preprocessing.image_preprocessor import ImagePreprocessorImpl
from app.rules.invoice_rules import InvoiceValidatorImpl
from app.rules.rule_processor import RuleProcessorImpl
from tests.conftest import (
    SAMPLE_MARKDOWN_INTERSTATE,
    SAMPLE_MARKDOWN_INTRASTATE,
    FakeLlmExtractor,
    FakeReader,
)
from tests.test_invoice_rules import BASE


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((120, 200, 3), 255, dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def _pipeline(reader, extractor: InvoiceExtractor, settings: Settings | None = None) -> OcrPipeline:
    s = settings or Settings(preprocess_enabled=False)
    return OcrPipeline(
        settings=s,
        loader=DocumentLoaderImpl(s),
        preprocessor=ImagePreprocessorImpl(s),
        reader=reader,
        layout=LayoutAnalyzerImpl(),
        rules=RuleProcessorImpl(),
        extractor=extractor,
        validator=InvoiceValidatorImpl(s),
        scorer=ConfidenceScorerImpl(s),
        artifacts=NullArtifactRepository(),
        fallback_extractor=RulesExtractor(),
    )


def test_rules_extractor_recovers_line_items_from_table():
    pipe = _pipeline(FakeReader(SAMPLE_MARKDOWN_INTRASTATE), RulesExtractor())
    result = pipe.run_sync(_png_bytes(), "image/png", "scan.png")
    inv = result.invoice
    assert result.method == "ocr"
    assert len(inv.line_items) == 2
    first = inv.line_items[0]
    assert first.hsn_sac == "7208" and first.quantity == 10 and first.rate == 5000
    assert first.taxable_amount == 50000
    # intra-state → CGST + SGST
    assert first.cgst_amount == 4500 and first.sgst_amount == 4500
    # totals foot
    t = inv.totals
    assert round(t.taxable_total + t.tax_total + t.round_off, 2) == t.grand_total == 70800


def test_llm_extractor_happy_path_high_confidence():
    pipe = _pipeline(FakeReader(SAMPLE_MARKDOWN_INTRASTATE), FakeLlmExtractor(BASE))
    result = pipe.run_sync(_png_bytes(), "image/png", "scan.png")
    inv = result.invoice
    assert inv.seller.name == "ACME STEELS PRIVATE LIMITED"
    assert inv.seller.state_name == "Gujarat"
    assert inv.overall_confidence >= 0.8


def test_interstate_document_uses_igst():
    pipe = _pipeline(FakeReader(SAMPLE_MARKDOWN_INTERSTATE), RulesExtractor())
    result = pipe.run_sync(_png_bytes(), "image/png", "scan.png")
    assert result.invoice.line_items[0].igst_amount == 9000


def test_extractor_failure_falls_back():
    class Boom(InvoiceExtractor):
        name = "boom"

        def warm_up(self) -> None: ...

        def extract(self, context: ExtractionContext) -> dict:
            raise RuntimeError("model exploded")

    pipe = _pipeline(FakeReader(SAMPLE_MARKDOWN_INTRASTATE), Boom())
    result = pipe.run_sync(_png_bytes(), "image/png", "scan.png")
    # fallback rules extractor still recovered the lines
    assert len(result.invoice.line_items) == 2


def test_blank_document_is_terminal():
    from app.pipeline.errors import TerminalDocumentError

    pipe = _pipeline(FakeReader(""), RulesExtractor())
    with pytest.raises(TerminalDocumentError):
        pipe.run_sync(_png_bytes(), "image/png", "scan.png")


def test_async_run_matches_sync():
    pipe = _pipeline(FakeReader(SAMPLE_MARKDOWN_INTRASTATE), RulesExtractor())
    result = asyncio.run(pipe.run(_png_bytes(), "image/png", "scan.png"))
    assert len(result.invoice.line_items) == 2


def test_reader_error_is_retryable():
    class BadReader(FakeReader):
        def read(self, pages):
            raise RuntimeError("engine down")

    pipe = _pipeline(BadReader("x"), RulesExtractor())
    with pytest.raises(RetryableEngineError):
        pipe.run_sync(_png_bytes(), "image/png", "scan.png")
