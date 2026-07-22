"""Composition root — the one place concrete implementations are chosen.

Every other module depends only on the Protocols in ``app.domain.interfaces``.
The container reads :class:`~app.config.Settings` and assembles the object graph,
selecting the reader engine (PaddleOCR-VL vs the null stub) and the extractor
engine (Qwen3-8B vs the rules fallback). Swapping an implementation is a change
*here only*.
"""

from __future__ import annotations

from functools import cached_property

from app.confidence.scorer import ConfidenceScorerImpl
from app.config import Settings, load_settings
from app.domain.interfaces import (
    ArtifactRepository,
    ConfidenceScorer,
    DocumentLoader,
    DocumentReader,
    ImagePreprocessor,
    InvoiceExtractor,
    InvoiceValidator,
    LayoutAnalyzer,
    RuleProcessor,
)
from app.extraction.llm_extractor import QwenLlmExtractor
from app.extraction.rules_extractor import RulesExtractor
from app.ingestion.document_loader import DocumentLoaderImpl
from app.layout.layout_analyzer import LayoutAnalyzerImpl
from app.ocr.null_reader import NullReader
from app.persistence.artifact_repository import (
    FileArtifactRepository,
    NullArtifactRepository,
)
from app.pipeline.orchestrator import OcrPipeline
from app.preprocessing.image_preprocessor import ImagePreprocessorImpl
from app.rules.invoice_rules import InvoiceValidatorImpl
from app.rules.rule_processor import RuleProcessorImpl
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Container:
    """Lazily-built singletons wired from :class:`Settings`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    # -- stages ---------------------------------------------------------------
    @cached_property
    def loader(self) -> DocumentLoader:
        return DocumentLoaderImpl(self.settings)

    @cached_property
    def preprocessor(self) -> ImagePreprocessor:
        return ImagePreprocessorImpl(self.settings)

    @cached_property
    def reader(self) -> DocumentReader:
        engine = self.settings.reader_engine.lower()
        if engine in ("null", "none", "pdf-text"):
            logger.info("reader engine: null (no OCR model)")
            return NullReader()
        if engine in ("paddleocr-vl", "vl"):
            from app.ocr.paddle_vl_reader import PaddleVLReader

            return PaddleVLReader(self.settings)
        if engine in ("paddleocr", "ppocr", "classic"):
            from app.ocr.paddle_ocr_reader import PaddleOcrReader

            return PaddleOcrReader(self.settings)
        # Default: RapidOCR (PP-OCR via ONNX) — the CPU-stable reader.
        from app.ocr.rapidocr_reader import RapidOcrReader

        return RapidOcrReader(self.settings)

    @cached_property
    def layout(self) -> LayoutAnalyzer:
        return LayoutAnalyzerImpl()

    @cached_property
    def rule_processor(self) -> RuleProcessor:
        return RuleProcessorImpl()

    @cached_property
    def extractor(self) -> InvoiceExtractor:
        engine = self.settings.extractor_engine.lower()
        if engine in ("rules", "none", "null"):
            logger.info("extractor engine: rules (no LLM)")
            return RulesExtractor()
        return QwenLlmExtractor(self.settings)

    @cached_property
    def fallback_extractor(self) -> InvoiceExtractor | None:
        # A model-based primary always gets the deterministic fallback so a model
        # hiccup degrades to a partial result instead of a 5xx.
        if isinstance(self.extractor, RulesExtractor):
            return None
        return RulesExtractor()

    @cached_property
    def validator(self) -> InvoiceValidator:
        return InvoiceValidatorImpl(self.settings)

    @cached_property
    def scorer(self) -> ConfidenceScorer:
        return ConfidenceScorerImpl(self.settings)

    @cached_property
    def artifacts(self) -> ArtifactRepository:
        if self.settings.save_artifacts:
            return FileArtifactRepository(self.settings.artifacts_dir)
        return NullArtifactRepository()

    # -- pipeline -------------------------------------------------------------
    @cached_property
    def pipeline(self) -> OcrPipeline:
        return OcrPipeline(
            settings=self.settings,
            loader=self.loader,
            preprocessor=self.preprocessor,
            reader=self.reader,
            layout=self.layout,
            rules=self.rule_processor,
            extractor=self.extractor,
            validator=self.validator,
            scorer=self.scorer,
            artifacts=self.artifacts,
            fallback_extractor=self.fallback_extractor,
        )
