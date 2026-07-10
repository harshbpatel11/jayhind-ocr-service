"""Extraction engines: digital-PDF text layer (fast path) and PaddleOCR."""
import io
import math
import time
from statistics import median
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
    workers are single-threaded per process, and construction is idempotent).

    Model weights download on first use, not at import, so start-up stays fast.
    Doc-orientation and unwarping pre-models are disabled: invoices are already
    page-shaped, the models cost seconds per page, and they add ARM crash surface.
    Line-level rotation is still handled by `use_textline_orientation`.
    """
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang=config.LANG,
            device="gpu" if config.USE_GPU else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,  # rotated / upside-down scan lines
            text_detection_model_name=config.DET_MODEL,
            text_recognition_model_name=config.REC_MODEL,
            cpu_threads=config.CPU_THREADS,
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


def _estimate_skew(polys) -> float:
    """Dominant text angle (radians) from the top edge of each detected line box.

    Photographed invoices are never square to the camera. A 3° skew spreads one
    text line across ~70px of a 1400px-wide page — far more than a line's height
    — so grouping tokens into lines by their y coordinate falls apart unless the
    page is de-skewed first. The median is used so a few odd boxes can't drag it.
    """
    angles = []
    for poly in polys:
        (x0, y0), (x1, y1) = (float(poly[0][0]), float(poly[0][1])), (float(poly[1][0]), float(poly[1][1]))
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) < 5:  # too short to give a reliable angle
            continue
        angle = math.atan2(dy, dx)
        if abs(angle) < math.radians(15):  # ignore vertical/garbage boxes
            angles.append(angle)
    return median(angles) if angles else 0.0


def _rotate(point, angle: float, cx: float, cy: float):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y = point[0] - cx, point[1] - cy
    return (x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy)


def _polygons_to_tokens(texts, scores, polys, size: Tuple[float, float]) -> List[Dict]:
    """PaddleOCR returns a quadrilateral per text line; we need axis-aligned boxes.

    Coordinates are de-skewed about the page centre first, so downstream line
    grouping and the seller/buyer column split work on rotated scans.
    """
    skew = _estimate_skew(polys)
    cx, cy = size[0] / 2.0, size[1] / 2.0

    tokens: List[Dict] = []
    for text, score, poly in zip(texts, scores, polys):
        points = [(float(p[0]), float(p[1])) for p in poly]
        if abs(skew) > math.radians(0.3):
            points = [_rotate(p, -skew, cx, cy) for p in points]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        tokens.append(
            {
                "text": text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "confidence": float(score),
            }
        )
    return tokens


def _fit_for_ocr(image):
    """Scale an image so its longest side lands inside the band Paddle reads best.

    Text much under ~20px tall recognises poorly, so small screenshots are
    upscaled; oversized photos are downscaled because inference cost and peak
    memory grow with pixel count while accuracy does not. Aspect ratio is kept, so
    the token boxes stay proportional to the page dimensions we report.
    """
    from PIL import Image

    longest = max(image.size)
    if longest < config.MIN_IMAGE_SIDE:
        scale = config.MIN_IMAGE_SIDE / longest
    elif longest > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / longest
    else:
        return image
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(resized, Image.LANCZOS)


def _run_ocr(image_bytes: bytes, page_index: int) -> Dict:
    """OCR one page image. The page's reported size is the size actually inferred
    on, so token boxes and page dimensions always share one coordinate space."""
    if not _OCR_IMPORTABLE:
        raise OcrUnavailable(
            f"PaddleOCR is not installed in this environment ({_ocr_import_error}). "
            "Install the paddlepaddle/paddleocr requirements to process scans and images."
        )
    engine = _get_ocr_engine()

    # Normalise WEBP / PNG-with-alpha into plain RGB before handing to Paddle.
    import numpy as np
    from PIL import Image

    image = _fit_for_ocr(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    size = (float(image.width), float(image.height))
    array = np.array(image)

    # PaddleOCR 3.x: `predict()` returns one result dict per image, carrying
    # parallel `rec_texts` / `rec_scores` / `rec_polys` lists. (2.x's `ocr(cls=)`
    # returned a nested [[box, (text, score)]] structure — handled as a fallback.)
    if hasattr(engine, "predict"):
        result = engine.predict(input=array)
        if not result:
            return _empty_page(page_index, size)
        page = result[0]
        tokens = _polygons_to_tokens(page["rec_texts"], page["rec_scores"], page["rec_polys"], size)
    else:  # pragma: no cover - PaddleOCR 2.x
        legacy = engine.ocr(array, cls=True)
        rows = (legacy or [[]])[0] or []
        tokens = _polygons_to_tokens(
            [row[1][0] for row in rows], [row[1][1] for row in rows], [row[0] for row in rows], size
        )

    return {
        "index": page_index,
        "width": size[0],
        "height": size[1],
        "text": tokens_to_text(tokens),
        "tokens": tokens,
        "tables": [],
    }


def _empty_page(page_index: int, size: Tuple[float, float]) -> Dict:
    return {"index": page_index, "width": float(size[0]), "height": float(size[1]), "text": "", "tokens": [], "tables": []}


def _rasterise_pdf(data: bytes) -> List[bytes]:
    import fitz  # pymupdf

    zoom = config.DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=data, filetype="pdf") as doc:
        return [page.get_pixmap(matrix=matrix).tobytes("png") for page in list(doc)[: config.MAX_PAGES]]


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
            pages = [_run_ocr(image, index) for index, image in enumerate(_rasterise_pdf(data))]

    elif content_type in config.IMAGE_MIME_TYPES:
        pages = [_run_ocr(data, 0)]
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
