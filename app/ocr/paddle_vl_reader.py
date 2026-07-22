"""PaddleOCR-VL 1.6 reader — the default, accuracy-first OCR + layout stage.

PaddleOCR-VL is a vision-language document parser: given a page image it returns
reading-order **markdown** with tables, titles and key-value blocks already
recovered — so layout analysis, table detection and reading-order recovery are
handled by the model itself. This wrapper:

  * loads the pipeline once (``warm_up``), pinned to CPU with a bounded thread
    count so it coexists with the resident 8B extractor on a 4-vCPU box;
  * skips OCR on pages that already carry an embedded text layer (digital PDFs);
  * normalises the several shapes ``predict`` can return into markdown + text.

Heavy imports (paddle) are deferred to ``warm_up`` so this module imports cleanly
on a host without the Paddle stack (tests use the null reader instead).
"""

from __future__ import annotations

import glob
import json
import os
import tempfile

from app.config import Settings
from app.domain.interfaces import DocumentReader
from app.domain.pipeline_types import PageImage, PageLayout, ReaderOutput
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PaddleVLReader(DocumentReader):
    """Wraps ``paddleocr.PaddleOCRVL`` for local CPU inference."""

    engine = "paddleocr-vl"

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._pipeline = None

    # -- lifecycle ------------------------------------------------------------
    def warm_up(self) -> None:
        """Load the VL pipeline once. Raises if the Paddle stack is missing."""
        if self._pipeline is not None:
            return
        # Bound CPU parallelism BEFORE paddle imports (it reads these at init).
        threads = str(self._s.reader_cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("PADDLE_NUM_THREADS", threads)

        logger.info("loading PaddleOCR-VL (CPU, %s threads) ...", threads)
        import paddle  # type: ignore
        from paddleocr import PaddleOCRVL  # type: ignore

        paddle.set_device("cpu")
        try:
            paddle.set_num_threads(self._s.reader_cpu_threads)
        except Exception:  # older paddle builds
            pass

        # PaddleOCR-VL recovers textline orientation internally; only the
        # document-orientation classifier is a constructor toggle here.
        self._pipeline = PaddleOCRVL(
            use_doc_orientation_classify=self._s.use_doc_orientation,
        )
        logger.info("PaddleOCR-VL ready.")

    def is_ready(self) -> bool:
        return self._pipeline is not None

    def _ensure(self):
        if self._pipeline is None:
            self.warm_up()
        return self._pipeline

    # -- reading --------------------------------------------------------------
    def read(self, pages: list[PageImage]) -> ReaderOutput:
        pipeline = self._ensure()
        page_layouts: list[PageLayout] = []
        with tempfile.TemporaryDirectory() as workdir:
            for page in pages:
                if page.has_text_layer:
                    # Digital-PDF fast path: trust the embedded text, skip the model.
                    md = page.text_layer.strip()
                else:
                    md = self._read_one(pipeline, page, workdir)
                page_layouts.append(
                    PageLayout(
                        index=page.index,
                        width=page.width,
                        height=page.height,
                        markdown=md,
                        text=_markdown_to_text(md),
                    )
                )
        markdown = "\n\n".join(p.markdown for p in page_layouts if p.markdown).strip()
        text = "\n\n".join(p.text for p in page_layouts if p.text).strip()
        return ReaderOutput(pages=page_layouts, markdown=markdown, text=text, engine=self.engine)

    def _read_one(self, pipeline, page: PageImage, workdir: str) -> str:
        import cv2  # local import: only needed on the OCR path

        path = os.path.join(workdir, f"page_{page.index}.png")
        # Paddle reads BGR from disk; our pipeline images are RGB.
        cv2.imwrite(path, cv2.cvtColor(page.image, cv2.COLOR_RGB2BGR))
        parts: list[str] = []
        for result in pipeline.predict(path):
            parts.append(_result_to_markdown(result, workdir))
        return "\n\n".join(p for p in parts if p).strip()


def _result_to_markdown(result, workdir: str) -> str:
    """Coerce a PaddleOCR-VL prediction into markdown, however it is shaped."""
    md = getattr(result, "markdown", None)
    if isinstance(md, dict):
        return (md.get("markdown_texts") or md.get("markdown") or "").strip()
    if isinstance(md, str):
        return md.strip()
    try:
        out = os.path.join(workdir, "md")
        result.save_to_markdown(save_path=out)
        files = glob.glob(os.path.join(out, "**", "*.md"), recursive=True)
        if files:
            with open(files[0], encoding="utf-8") as handle:
                return handle.read().strip()
    except Exception:  # pragma: no cover - depends on paddle internals
        pass
    payload = getattr(result, "json", None)
    return json.dumps(payload, ensure_ascii=False) if payload else ""


def _markdown_to_text(markdown: str) -> str:
    """Flatten markdown to plain reading-order text (drop table pipes/hashes)."""
    lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or set(line) <= {"|", "-", ":", " "}:
            continue  # markdown table separator rows
        line = line.lstrip("#").strip()
        line = line.strip("|").replace("|", "  ").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
