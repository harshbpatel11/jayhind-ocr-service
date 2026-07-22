"""RapidOCR reader — PP-OCR models via ONNX Runtime (the DEFAULT on ARM64 CPU).

RapidOCR runs PaddleOCR's PP-OCR detection + recognition + angle models through
**ONNX Runtime**, which is stable and fast on aarch64 — unlike PaddlePaddle's
native CPU inference, which segfaults on this Neoverse N1 build for both the VL
and the classic pipelines. Same models, reliable runtime; this satisfies the
project's "PaddleOCR models + ONNX Runtime, CPU-only" requirement.

It returns recognised text boxes; this wrapper recovers reading order and emits
reading-order text for the layout analyzer, rule hints and the Qwen extractor.
"""

from __future__ import annotations

import os

from app.config import Settings
from app.domain.interfaces import DocumentReader
from app.domain.pipeline_types import PageImage, PageLayout, ReaderOutput
from app.ocr.reading_order import poly_top_left, reading_order_text
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RapidOcrReader(DocumentReader):
    """Wraps ``rapidocr_onnxruntime.RapidOCR`` for local CPU inference."""

    engine = "rapidocr"

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._engine = None

    # -- lifecycle ------------------------------------------------------------
    def warm_up(self) -> None:
        if self._engine is not None:
            return
        threads = self._s.reader_cpu_threads
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        logger.info("loading RapidOCR (ONNX, %d threads) ...", threads)
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        # Bound ONNX Runtime parallelism and enable document + textline angle
        # correction (the spec's orientation / angle-classification requirement).
        self._engine = RapidOCR(
            intra_op_num_threads=threads,
            use_angle_cls=True,
        )
        logger.info("RapidOCR ready.")

    def is_ready(self) -> bool:
        return self._engine is not None

    def _ensure(self):
        if self._engine is None:
            self.warm_up()
        return self._engine

    # -- reading --------------------------------------------------------------
    def read(self, pages: list[PageImage]) -> ReaderOutput:
        engine = self._ensure()
        page_layouts: list[PageLayout] = []
        for page in pages:
            if page.has_text_layer:
                text = page.text_layer.strip()
            else:
                text = self._read_one(engine, page)
            page_layouts.append(
                PageLayout(
                    index=page.index,
                    width=page.width,
                    height=page.height,
                    markdown=text,
                    text=text,
                )
            )
        markdown = "\n\n".join(p.markdown for p in page_layouts if p.markdown).strip()
        text = "\n\n".join(p.text for p in page_layouts if p.text).strip()
        return ReaderOutput(pages=page_layouts, markdown=markdown, text=text, engine=self.engine)

    def _read_one(self, engine, page: PageImage) -> str:
        import cv2

        # RapidOCR reads cv2-style BGR arrays; our pipeline images are RGB.
        result, _elapse = engine(cv2.cvtColor(page.image, cv2.COLOR_RGB2BGR))
        lines: list[tuple[float, float, str]] = []
        for box, text, _score in result or []:
            if not str(text).strip():
                continue
            y, x = poly_top_left(box)
            lines.append((y, x, str(text).strip()))
        return reading_order_text(lines)
