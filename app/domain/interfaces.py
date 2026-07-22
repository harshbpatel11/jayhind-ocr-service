"""Stage interfaces (the seams the DI container wires together).

Every pipeline stage is a ``typing.Protocol`` so implementations depend on the
*abstraction*, never on each other. This is what makes the pipeline modular and
extensible: the PaddleOCR-VL reader, the Qwen3-8B extractor, and their
test doubles are interchangeable as long as they satisfy these Protocols. The
composition root (``app/container.py``) is the only place that names concrete
classes.

Heavy stages (reader, extractor) additionally expose ``warm_up()`` so the
container can load models once at startup instead of on the first request.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.models import ExtractedInvoice
from app.domain.pipeline_types import (
    ExtractionContext,
    LoadedDocument,
    PageImage,
    ReaderOutput,
    RuleHints,
)


@runtime_checkable
class DocumentLoader(Protocol):
    """Bytes → rasterised pages (PDF, PNG, JPG, TIFF, phone photo)."""

    def load(self, data: bytes, content_type: str, filename: str) -> LoadedDocument: ...


@runtime_checkable
class ImagePreprocessor(Protocol):
    """Clean a single page image (deskew, denoise, threshold, contrast, DPI)."""

    def preprocess(self, image: np.ndarray, dpi: int) -> np.ndarray: ...


@runtime_checkable
class DocumentReader(Protocol):
    """OCR + layout: page images → structured, reading-order text.

    The default implementation is PaddleOCR-VL 1.6; the null reader is used in
    tests and on hosts without the Paddle stack.
    """

    #: Human-readable engine id echoed into ``ReaderOutput.engine``.
    engine: str

    def read(self, pages: list[PageImage]) -> ReaderOutput: ...

    def warm_up(self) -> None: ...


@runtime_checkable
class LayoutAnalyzer(Protocol):
    """Refine / normalise the reader's raw layout (reading order, table shapes)."""

    def analyze(self, reader_output: ReaderOutput) -> ReaderOutput: ...


@runtime_checkable
class RuleProcessor(Protocol):
    """Deterministic pre-LLM structuring: layout → grounded hints."""

    def process(self, reader_output: ReaderOutput) -> RuleHints: ...


@runtime_checkable
class InvoiceExtractor(Protocol):
    """Structured text → raw invoice dict (the LLM stage).

    Returns a plain ``dict`` in roughly the contract shape; the validator is
    responsible for coercing and guaranteeing the final schema.
    """

    #: Human-readable extractor id (``qwen3-8b-instruct``, ``rules`` …).
    name: str

    def extract(self, context: ExtractionContext) -> dict: ...

    def warm_up(self) -> None: ...


@runtime_checkable
class InvoiceValidator(Protocol):
    """Raw dict + context → a fully-typed, business-rule-correct invoice."""

    def validate(self, raw: dict, context: ExtractionContext) -> ExtractedInvoice: ...


@runtime_checkable
class ConfidenceScorer(Protocol):
    """Attach per-field + overall confidence to a validated invoice."""

    def score(self, invoice: ExtractedInvoice, context: ExtractionContext) -> ExtractedInvoice: ...


@runtime_checkable
class ArtifactRepository(Protocol):
    """Repository-pattern persistence for intermediate artifacts.

    Optional: lets an operator archive the rendered page images, reader markdown
    and raw LLM output per request for debugging/accuracy work, without any
    pipeline stage knowing where or whether things are stored.
    """

    def save_text(self, request_id: str, name: str, text: str) -> None: ...

    def save_bytes(self, request_id: str, name: str, data: bytes) -> None: ...
