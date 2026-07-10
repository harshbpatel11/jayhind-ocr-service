"""Environment-driven configuration for the OCR sidecar."""
import os


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


USE_GPU: bool = _bool("OCR_USE_GPU", False)
LANG: str = os.getenv("OCR_LANG", "en")

#: PaddleOCR 3.x model names. The 3.x default (PP-OCRv6) **segfaults on
#: aarch64/ARM**; PP-OCRv5 mobile is stable everywhere, fast, and preserves word
#: spacing (v4 runs the words together). Override on x86 if you want the larger
#: `_server_` variants for a little more accuracy.
DET_MODEL: str = os.getenv("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
REC_MODEL: str = os.getenv("OCR_REC_MODEL", "PP-OCRv5_mobile_rec")

#: Paddle's multi-threaded CPU inference segfaults on ARM, so we pin one thread
#: by default. On x86 raising this (2-4) is a straight speed win.
CPU_THREADS: int = _int("OCR_CPU_THREADS", 1)

#: Rasterisation DPI for scanned PDFs. 200 balances OCR accuracy and speed.
DPI: int = _int("OCR_DPI", 200)

#: Hard cap on pages processed per document (protects the worker from a 200-page scan).
MAX_PAGES: int = _int("OCR_MAX_PAGES", 10)

#: Every image is scaled so its longest side lands inside this band before
#: inference. Below ~1400px the text on an A4 invoice is too small to recognise
#: reliably; above ~2600px inference cost and peak memory climb with no accuracy
#: gain — and a full-resolution phone photo is precisely what exhausts the worker
#: on ARM (the same place `OCR_CPU_THREADS > 1` crashes).
MIN_IMAGE_SIDE: int = _int("OCR_MIN_IMAGE_SIDE", 1400)
MAX_IMAGE_SIDE: int = _int("OCR_MAX_IMAGE_SIDE", 2600)

#: Hard cap on an uploaded document. The backend caps uploads too, but the sidecar
#: must protect itself — it is the process that would run out of memory.
MAX_UPLOAD_BYTES: int = _int("OCR_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)

#: Chars per page required before a PDF is treated as "digital" (has a text layer).
#: Scanned PDFs often carry a few stray characters (page numbers, producer marks),
#: so a small non-zero threshold is safer than `> 0`.
TEXT_LAYER_MIN_CHARS: int = _int("OCR_TEXT_LAYER_MIN_CHARS", 120)

#: Tokens whose vertical centres fall within this fraction of the median token
#: height are treated as the same visual line when reconstructing reading order.
LINE_TOLERANCE_RATIO: float = 0.6

IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp"})
PDF_MIME_TYPES = frozenset({"application/pdf"})
