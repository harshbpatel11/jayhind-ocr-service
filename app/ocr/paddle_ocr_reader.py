"""Classic PaddleOCR (PP-OCRv5) reader — detection + recognition, CPU-stable.

The PaddleOCR-VL vision-language model, while more accurate, segfaults during
inference on this ARM64 CPU/Paddle build. The classic PP-OCR detection+recognition
pipeline is the mature, battle-tested CPU path and runs cleanly on aarch64, so it
is the DEFAULT reader here. It returns recognised text lines with bounding boxes;
this wrapper recovers reading order (top-to-bottom, left-to-right, line-grouped)
and emits reading-order text that the downstream layout analyzer, rule hints and
the Qwen extractor consume.

Heavy imports (paddle) are deferred to ``warm_up`` so the module imports on hosts
without the Paddle stack (tests use the null reader).
"""

from __future__ import annotations

import os
import tempfile

from app.config import Settings
from app.domain.interfaces import DocumentReader
from app.domain.pipeline_types import PageImage, PageLayout, ReaderOutput
from app.ocr.reading_order import poly_top_left, reading_order_text
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PaddleOcrReader(DocumentReader):
    """Wraps ``paddleocr.PaddleOCR`` (PP-OCRv5) for local CPU inference."""

    engine = "paddleocr"

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._ocr = None

    # -- lifecycle ------------------------------------------------------------
    def warm_up(self) -> None:
        if self._ocr is not None:
            return
        threads = str(self._s.reader_cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("PADDLE_NUM_THREADS", threads)

        logger.info("loading PaddleOCR (PP-OCRv5, CPU, %s threads) ...", threads)
        import paddle  # type: ignore
        from paddleocr import PaddleOCR  # type: ignore

        paddle.set_device("cpu")
        try:
            paddle.set_num_threads(self._s.reader_cpu_threads)
        except Exception:
            pass

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=self._s.use_doc_orientation,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        logger.info("PaddleOCR ready.")

    def is_ready(self) -> bool:
        return self._ocr is not None

    def _ensure(self):
        if self._ocr is None:
            self.warm_up()
        return self._ocr

    # -- reading --------------------------------------------------------------
    def read(self, pages: list[PageImage]) -> ReaderOutput:
        ocr = self._ensure()
        page_layouts: list[PageLayout] = []
        with tempfile.TemporaryDirectory() as workdir:
            for page in pages:
                if page.has_text_layer:
                    text = page.text_layer.strip()
                else:
                    text = self._read_one(ocr, page, workdir)
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

    def _read_one(self, ocr, page: PageImage, workdir: str) -> str:
        import cv2

        path = os.path.join(workdir, f"page_{page.index}.png")
        cv2.imwrite(path, cv2.cvtColor(page.image, cv2.COLOR_RGB2BGR))
        lines: list[tuple[float, float, str]] = []
        for result in ocr.predict(input=path):
            lines.extend(_extract_lines(result))
        return reading_order_text(lines)


def _extract_lines(result) -> list[tuple[float, float, str]]:
    """Pull (y, x, text) triples from one PaddleOCR prediction, however shaped."""
    payload = getattr(result, "json", None)
    data = payload.get("res", payload) if isinstance(payload, dict) else {}
    texts = data.get("rec_texts") or []
    polys = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or []
    out: list[tuple[float, float, str]] = []
    for i, text in enumerate(texts):
        if not str(text).strip():
            continue
        y, x = poly_top_left(polys[i]) if i < len(polys) else (float(i), 0.0)
        out.append((y, x, str(text).strip()))
    return out
