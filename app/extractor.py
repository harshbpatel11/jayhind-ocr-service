"""Extraction engines: digital-PDF text layer (fast path) and PaddleOCR."""
import io
import time
from typing import Dict, List, Optional, Tuple

from . import config
from .reading_order import tokens_to_text

# PaddleOCR is optional at import time: the digital-PDF path must work in a
# light install (see README). The engine is also lazily constructed so process
# start-up stays fast and model weights download on first real use.
_ocr_engine = None
_ocr_import_error: Optional[str] = None

try:  # pragma: no cover - import guard
    from paddleocr import PaddleOCR  # noqa: F401

    _OCR_IMPORTABLE = True
except Exception as exc:  # pragma: no cover - import guard
    _OCR_IMPORTABLE = False
    _ocr_import_error = str(exc)


def ocr_available() -> bool:
    return _OCR_IMPORTABLE


def _get_ocr_engine():
    """Lazily build the singleton PaddleOCR engine (thread-safe enough: uvicorn
    workers are single-threaded per process, and construction is idempotent)."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            use_angle_cls=True,  # handles rotated / upside-down scans
            lang=config.LANG,
            use_gpu=config.USE_GPU,
            show_log=False,
        )
    return _ocr_engine


class UnsupportedFileType(Exception):
    pass


class OcrUnavailable(Exception):
    pass


# ── Digital PDF path ────────────────────────────────────────────────────────


def _pdf_has_text_layer(pdf) -> bool:
    """A PDF counts as digital when its pages carry real extractable text.

    Scanned PDFs frequently contain a handful of stray characters (page numbers,
    producer stamps), so we require a meaningful character count per page rather
    than any text at all.
    """
    pages = pdf.pages[: config.MAX_PAGES]
    if not pages:
        return False
    total = sum(len((page.extract_text() or "").strip()) for page in pages)
    return total >= config.TEXT_LAYER_MIN_CHARS * len(pages)


def _extract_pdf_text(pdf) -> List[Dict]:
    """Words + boxes straight from the text layer. Confidence is always 1.0."""
    pages: List[Dict] = []
    for index, page in enumerate(pdf.pages[: config.MAX_PAGES]):
        tokens = [
            {
                "text": word["text"],
                "bbox": [float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"])],
                "confidence": 1.0,
            }
            for word in page.extract_words()
        ]
        tables = [{"rows": table} for table in (page.extract_tables() or [])]
        pages.append(
            {
                "index": index,
                "width": float(page.width),
                "height": float(page.height),
                "text": tokens_to_text(tokens),
                "tokens": tokens,
                "tables": tables,
            }
        )
    return pages


# ── OCR path ────────────────────────────────────────────────────────────────


def _run_ocr(image_bytes: bytes, page_index: int, size: Tuple[float, float]) -> Dict:
    if not _OCR_IMPORTABLE:
        raise OcrUnavailable(
            f"PaddleOCR is not installed in this environment ({_ocr_import_error}). "
            "Install the paddlepaddle/paddleocr requirements to process scans and images."
        )
    engine = _get_ocr_engine()
    # PaddleOCR accepts raw bytes decoded to ndarray; go through PIL so we
    # normalise WEBP/PNG-with-alpha into plain RGB.
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = engine.ocr(np.array(image), cls=True)

    tokens: List[Dict] = []
    # PaddleOCR returns [[ [box, (text, score)], ... ]] — one list per image.
    for line in (result or [[]])[0] or []:
        box, (text, score) = line[0], line[1]
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        tokens.append(
            {
                "text": text,
                "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                "confidence": float(score),
            }
        )

    return {
        "index": page_index,
        "width": float(size[0]),
        "height": float(size[1]),
        "text": tokens_to_text(tokens),
        "tokens": tokens,
        "tables": [],
    }


def _rasterise_pdf(data: bytes) -> List[Tuple[bytes, Tuple[float, float]]]:
    import fitz  # pymupdf

    zoom = config.DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    rendered: List[Tuple[bytes, Tuple[float, float]]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in list(doc)[: config.MAX_PAGES]:
            pixmap = page.get_pixmap(matrix=matrix)
            rendered.append((pixmap.tobytes("png"), (pixmap.width, pixmap.height)))
    return rendered


# ── Public entry point ──────────────────────────────────────────────────────


def extract(data: bytes, content_type: str) -> Dict:
    """Extract reading-order text + tokens from an invoice document.

    Picks the digital-PDF fast path when a text layer is present, else OCR.
    """
    started = time.monotonic()

    if content_type in config.PDF_MIME_TYPES:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if _pdf_has_text_layer(pdf):
                pages = _extract_pdf_text(pdf)
                method = "pdf-text"
            else:
                pages = None
                method = "ocr"

        if pages is None:  # scanned PDF → rasterise then OCR
            pages = [
                _run_ocr(image, index, size)
                for index, (image, size) in enumerate(_rasterise_pdf(data))
            ]

    elif content_type in config.IMAGE_MIME_TYPES:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as probe:
            size = probe.size
        pages = [_run_ocr(data, 0, size)]
        method = "ocr"

    else:
        raise UnsupportedFileType(f"Unsupported content type: {content_type}")

    return {
        "method": method,
        "pageCount": len(pages),
        "durationMs": int((time.monotonic() - started) * 1000),
        "text": "\n\n".join(page["text"] for page in pages).strip(),
        "pages": pages,
    }
