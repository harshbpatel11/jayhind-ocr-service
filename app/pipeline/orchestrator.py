"""The invoice-OCR pipeline orchestrator.

Wires the stages in the exact order the architecture prescribes:

    ingest → preprocess → PaddleOCR-VL → layout analysis → rule processing
           → Qwen3-8B extraction → business-rule validation → confidence scoring

Stages are injected (never imported here), so the orchestrator is pure
composition and is unit-testable with fakes. The blocking work runs in a worker
thread; a semaphore bounds concurrency to ``pipeline_workers`` (default 1) so the
two resident models never contend for the 4 vCPUs.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from app.config import Settings
from app.domain.interfaces import (
    ArtifactRepository,
    ConfidenceScorer,
    DocumentLoader,
    ImagePreprocessor,
    InvoiceExtractor,
    InvoiceValidator,
    LayoutAnalyzer,
    RuleProcessor,
)
from app.domain.models import ParseResult
from app.domain.pipeline_types import ExtractionContext, LoadedDocument
from app.ingestion.document_loader import UnsupportedDocument
from app.pipeline.errors import RetryableEngineError, TerminalDocumentError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OcrPipeline:
    """Composition of every pipeline stage behind one async entrypoint."""

    def __init__(
        self,
        *,
        settings: Settings,
        loader: DocumentLoader,
        preprocessor: ImagePreprocessor,
        reader,
        layout: LayoutAnalyzer,
        rules: RuleProcessor,
        extractor: InvoiceExtractor,
        validator: InvoiceValidator,
        scorer: ConfidenceScorer,
        artifacts: ArtifactRepository,
        fallback_extractor: InvoiceExtractor | None = None,
    ) -> None:
        self._s = settings
        self._loader = loader
        self._pre = preprocessor
        self._reader = reader
        self._layout = layout
        self._rules = rules
        self._extractor = extractor
        self._validator = validator
        self._scorer = scorer
        self._artifacts = artifacts
        self._fallback = fallback_extractor
        self._semaphore = asyncio.Semaphore(max(1, settings.pipeline_workers))

    # -- lifecycle ------------------------------------------------------------
    def warm_up(self) -> None:
        """Load heavy models once (called on startup)."""
        self._reader.warm_up()
        self._extractor.warm_up()

    # -- entrypoints ----------------------------------------------------------
    async def run(self, data: bytes, content_type: str, filename: str) -> ParseResult:
        """Async entrypoint: bounded concurrency + off-loop execution."""
        async with self._semaphore:
            return await asyncio.to_thread(self.run_sync, data, content_type, filename)

    def run_sync(self, data: bytes, content_type: str, filename: str) -> ParseResult:
        """The blocking pipeline. Raises :class:`PipelineError` subclasses."""
        started = time.monotonic()
        request_id = uuid.uuid4().hex[:12]

        document = self._ingest(data, content_type, filename)
        self._preprocess(document)
        reader_output = self._read(document)
        self._archive_text(request_id, "reader.md", reader_output.markdown)

        if not reader_output.text.strip() and not reader_output.markdown.strip():
            raise TerminalDocumentError("Document appears blank or unreadable")

        reader_output = self._layout.analyze(reader_output)
        hints = self._rules.process(reader_output)
        context = ExtractionContext(
            reader=reader_output,
            hints=hints,
            method=document.method,
            page_count=document.page_count,
        )

        raw = self._extract(context, request_id)
        invoice = self._validator.validate(raw, context)
        invoice = self._scorer.score(invoice, context)

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "parsed %s: engine=%s method=%s pages=%d conf=%.2f in %dms",
            filename or "<upload>", reader_output.engine, document.method,
            document.page_count, invoice.overall_confidence, duration_ms,
        )
        return ParseResult(
            method=document.method,
            structuring_method="rules",
            page_count=document.page_count,
            duration_ms=duration_ms,
            text=reader_output.text,
            invoice=invoice,
        )

    # -- stages (each maps failures to the right HTTP class) ------------------
    def _ingest(self, data: bytes, content_type: str, filename: str) -> LoadedDocument:
        try:
            return self._loader.load(data, content_type, filename)
        except UnsupportedDocument as exc:
            raise TerminalDocumentError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected decode failure
            raise TerminalDocumentError(f"Could not read document: {exc}") from exc

    def _preprocess(self, document: LoadedDocument) -> None:
        for page in document.pages:
            if page.has_text_layer:
                continue  # digital PDF page skips OCR, so skip cleanup too
            try:
                page.image = self._pre.preprocess(page.image, page.dpi)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("preprocess failed on page %d: %s", page.index, exc)

    def _read(self, document: LoadedDocument):
        try:
            return self._reader.read(document.pages)
        except Exception as exc:
            raise RetryableEngineError(f"OCR engine error: {exc}") from exc

    def _extract(self, context: ExtractionContext, request_id: str) -> dict:
        try:
            raw = self._extractor.extract(context)
            self._archive_text(request_id, "raw.json", str(raw))
            return raw
        except Exception as exc:
            if self._fallback is not None:
                logger.warning("extractor %s failed (%s); using fallback %s",
                               self._extractor.name, exc, self._fallback.name)
                try:
                    return self._fallback.extract(context)
                except Exception as fallback_exc:  # pragma: no cover - double failure
                    raise RetryableEngineError(f"Extraction failed: {fallback_exc}") from fallback_exc
            raise RetryableEngineError(f"Extraction failed: {exc}") from exc

    # -- artifacts ------------------------------------------------------------
    def _archive_text(self, request_id: str, name: str, text: str) -> None:
        if self._s.save_artifacts and text:
            self._artifacts.save_text(request_id, name, text)
