"""Extraction engines: digital-PDF text layer (fast path) and OCR.

Three OCR engines for the image / scanned-PDF path, chosen by `config.OCR_ENGINE`:
  • "onnx" (default) — RapidOCR on ONNX Runtime, bundled PP-OCR models
    (~50 MB, much faster on ARM/CPU).
  • "classic" — PaddleOCR PP-OCRv5 detection + recognition (~2 GB; fallback).
  • "vl" — PaddleOCR-VL, a local ~0.9B vision-language model. Better on hard
    scans but far slower on CPU; falls back to classic on any error so a page is
    never left unread.
All emit the same page dict, so structuring downstream is engine-agnostic.
"""
import io
import logging
import math
import time
from statistics import median
from typing import Dict, List, Optional, Tuple

from . import config
from .reading_order import tokens_to_text

logger = logging.getLogger("ocr-service.extractor")

# PaddleOCR is optional at import time: the digital-PDF path must work in a
# light install (see README). The engine is also lazily constructed so process
# start-up stays fast and model weights download on first real use.
_ocr_engine = None
_ocr_import_error: Optional[str] = None

# Presence is checked WITHOUT importing: paddlepaddle and onnxruntime segfault
# when loaded into the same process on ARM, so the unused engine must never be
# imported. The real import happens lazily in `_get_ocr_engine()`.
def _paddle_importable() -> bool:  # pragma: no cover - trivial guard
    import importlib.util

    try:
        return importlib.util.find_spec("paddleocr") is not None
    except Exception:
        return False


_OCR_IMPORTABLE = _paddle_importable()
if not _OCR_IMPORTABLE:
    _ocr_import_error = "paddleocr is not installed"


def _onnx_importable() -> bool:
    """Presence check WITHOUT importing: loading onnxruntime and paddle into the
    same process segfaults on ARM, so we never import the engine we won't use."""
    import importlib.util

    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def ocr_available() -> bool:
    """Whether the *configured* engine can run (reported by /health)."""
    if config.OCR_ENGINE == "remote":
        return bool(config.OCR_REMOTE_URL)
    if config.OCR_ENGINE == "onnx":
        return _onnx_importable()
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


def _deskew_gray(gray):
    """Rotate a grayscale image so its text lines are horizontal.

    The skew angle is the tilt of the minimum-area rectangle around the text
    pixels. Tiny (<0.3°) and implausible (>15°) angles are ignored so a clean
    page or a bad estimate is left untouched.
    """
    import cv2
    import numpy as np

    inverted = 255 - gray
    mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 50:
        return gray
    # minAreaRect's angle convention differs across OpenCV versions (it can report
    # 7° or 83° for the same 7° tilt). Fold it into (-45, 45] then correct by its
    # negation — verified to zero the residual for both signs on real pages.
    raw = cv2.minAreaRect(coords)[-1]
    angle = raw % 90
    if angle > 45:
        angle -= 90
    correction = -angle
    if abs(correction) < 0.3 or abs(correction) > 15:
        return gray
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), correction, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _preprocess_for_ocr(image):
    """Clean a photographed/scanned page before OCR: grayscale → deskew → local
    contrast (CLAHE) → gentle denoise. Returns a PIL RGB image (PaddleOCR wants
    3 channels). Best-effort: any failure returns the original image unchanged.
    """
    if not config.OCR_PREPROCESS:
        return image
    try:
        import cv2
        import numpy as np
        from PIL import Image

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray = _deskew_gray(gray)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        # Edge-preserving denoise — removes photo/scanner speckle without
        # smearing glyph strokes (bilateral keeps edges sharp).
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
        return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    except Exception as exc:  # never let preprocessing break extraction
        logger.warning("image preprocessing failed (%s); using the original image", exc)
        return image


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

    image = _fit_for_ocr(_preprocess_for_ocr(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
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


# ── ONNX Runtime path (RapidOCR — the default engine) ────────────────────────

_onnx_engine = None


def _get_onnx_engine():
    """Lazily build the RapidOCR (ONNX Runtime) engine. Bundled PP-OCR ONNX models
    — no download, no paddlepaddle."""
    global _onnx_engine
    if _onnx_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _onnx_engine = RapidOCR()
    return _onnx_engine


def _run_onnx_ocr(image_bytes: bytes, page_index: int) -> Dict:
    """OCR one page with RapidOCR/ONNX: ~8x faster than paddle on CPU and more
    robust on tilted/blurred photos (measured — see tests/ACCURACY_BASELINE.md).

    RapidOCR returns a quadrilateral per line; the page is already de-skewed by
    `_preprocess_for_ocr`, so an axis-aligned bounding box is faithful.
    """
    import numpy as np
    from PIL import Image

    image = _fit_for_ocr(_preprocess_for_ocr(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
    size = (float(image.width), float(image.height))
    result, _elapse = _get_onnx_engine()(np.array(image))
    if not result:
        return _empty_page(page_index, size)

    tokens: List[Dict] = []
    for box, text, score in result:
        if not str(text).strip():
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        tokens.append({
            "text": str(text),
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "confidence": float(score),
        })
    return {
        "index": page_index, "width": size[0], "height": size[1],
        "text": tokens_to_text(tokens), "tokens": tokens, "tables": [],
    }


# ── PaddleOCR-VL path (opt-in local vision-language model) ────────────────────

_vl_engine = None


def _get_vl_engine():
    """Lazily build the PaddleOCR-VL pipeline (weights download on first use).

    Doc-orientation/unwarping pre-models are disabled (invoices are page-shaped
    and they add cost + ARM crash surface), matching the classic engine.
    """
    global _vl_engine
    if _vl_engine is None:
        from paddleocr import PaddleOCRVL

        _vl_engine = PaddleOCRVL(use_doc_orientation_classify=False, use_doc_unwarping=False)
    return _vl_engine


def _vl_tokens_from_result(res, size: Tuple[float, float]) -> Tuple[List[Dict], str]:
    """Best-effort map a PaddleOCR-VL result into (tokens, text).

    VL returns layout blocks with bounding boxes + recognised content. We emit one
    token per block (its bbox + text) so the geometry-based structuring keeps the
    column layout; the reading-order text is derived from those tokens. Defensive:
    unknown result shapes yield ([], "") and the caller falls back to classic.
    """
    data = res.json if hasattr(res, "json") else res
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    blocks = []
    if isinstance(data, dict):
        for key in ("parsing_res_list", "layout_parsing_result", "blocks", "boxes"):
            if isinstance(data.get(key), list):
                blocks = data[key]
                break

    tokens: List[Dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bbox = block.get("block_bbox") or block.get("bbox") or block.get("layout_bbox")
        text = block.get("block_content") or block.get("content") or block.get("text") or ""
        if not (bbox and str(text).strip()):
            continue
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError, IndexError):
            continue
        # A multiline block → one token per line, stacked within the block bbox.
        lines = [ln for ln in str(text).splitlines() if ln.strip()] or [str(text)]
        step = (y1 - y0) / max(1, len(lines))
        for i, line in enumerate(lines):
            tokens.append({
                "text": line.strip(),
                "bbox": [x0, y0 + i * step, x1, y0 + (i + 1) * step],
                "confidence": float(block.get("score", 0.9) or 0.9),
            })

    if tokens:
        return tokens, tokens_to_text(tokens)
    # No geometry available — fall back to the markdown/plain text of the page.
    md = getattr(res, "markdown", None)
    text = md.get("markdown_texts") if isinstance(md, dict) else (md or "")
    return [], str(text or "").strip()


def _run_vl_ocr(image_bytes: bytes, page_index: int) -> Dict:
    """OCR one page image with PaddleOCR-VL."""
    import numpy as np
    from PIL import Image

    image = _fit_for_ocr(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    size = (float(image.width), float(image.height))
    results = _get_vl_engine().predict(input=np.array(image))
    if not results:
        return _empty_page(page_index, size)
    tokens, text = _vl_tokens_from_result(results[0], size)
    return {"index": page_index, "width": size[0], "height": size[1], "text": text, "tokens": tokens, "tables": []}


def _run_remote_ocr(image_bytes: bytes, page_index: int) -> Dict:
    """Send one page image to a remote OCR engine and adapt its reply.

    Used when `OCR_ENGINE=remote`: PaddleOCR-VL (which segfaults on this ARM box)
    runs on an x86/GPU host — typically Google Colab reached through a Cloudflare
    tunnel — and returns axis-aligned tokens in the coordinate space of the image
    it inferred on, so the local geometry-first structuring is unchanged. Any
    network/HTTP failure raises `OcrUnavailable` → 503, which the backend treats
    as retryable.
    """
    import json
    import urllib.error
    import urllib.request

    if not config.OCR_REMOTE_URL:
        raise OcrUnavailable("OCR_ENGINE=remote but OCR_REMOTE_URL is not set.")

    url = f"{config.OCR_REMOTE_URL}/ocr-page"
    headers = {"Content-Type": "image/png"}
    if config.OCR_REMOTE_KEY:
        headers["x-ocr-key"] = config.OCR_REMOTE_KEY
    request = urllib.request.Request(url, data=image_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=config.OCR_REMOTE_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")[:200]
        raise OcrUnavailable(f"Remote OCR returned HTTP {exc.code}: {body}") from exc
    except Exception as exc:  # timeout, DNS, tunnel down, bad JSON …
        raise OcrUnavailable(f"Remote OCR call to {url} failed: {exc}") from exc

    size = (float(payload.get("width") or 0.0), float(payload.get("height") or 0.0))
    tokens: List[Dict] = []
    for token in payload.get("tokens") or []:
        bbox = token.get("bbox")
        text = str(token.get("text", "")).strip()
        if not (bbox and text and len(bbox) >= 4):
            continue
        tokens.append({
            "text": text,
            "bbox": [float(v) for v in bbox[:4]],
            "confidence": float(token.get("confidence", 0.9) or 0.9),
        })
    # Prefer local reading-order over the tokens (identical to every other engine);
    # fall back to the remote's plain text only when it sent no geometry.
    text = tokens_to_text(tokens) if tokens else str(payload.get("text", "")).strip()
    return {"index": page_index, "width": size[0], "height": size[1], "text": text, "tokens": tokens, "tables": []}


def _ocr_page(image_bytes: bytes, page_index: int) -> Dict:
    """Run the configured OCR engine on one page.

    ⚠️ Engines are **mutually exclusive per process**: loading onnxruntime and
    paddlepaddle together segfaults on ARM. So `onnx` never auto-falls-back to
    `classic` in-process — switching engines is an env change (`OCR_ENGINE`), and
    a failed page surfaces as an error the backend can retry.

    ⚠️ PaddleOCR-VL also SEGFAULTS on inference on aarch64/CPU. A SIGSEGV crashes
    the worker and is not catchable, so `OCR_ENGINE=vl` is x86 / GPU only — use
    `remote` to reach a VL server on an x86/GPU host instead.
    """
    engine = config.OCR_ENGINE
    if engine == "remote":
        return _run_remote_ocr(image_bytes, page_index)
    if engine == "onnx":
        return _run_onnx_ocr(image_bytes, page_index)
    if engine == "vl":
        try:
            return _run_vl_ocr(image_bytes, page_index)
        except Exception as exc:  # download failed / unsupported result shape (NOT a segfault)
            logger.warning("PaddleOCR-VL failed on page %d (%s); falling back to classic OCR", page_index, exc)
    return _run_ocr(image_bytes, page_index)


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
            pages = [_ocr_page(image, index) for index, image in enumerate(_rasterise_pdf(data))]

    elif content_type in config.IMAGE_MIME_TYPES:
        pages = [_ocr_page(data, 0)]
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
