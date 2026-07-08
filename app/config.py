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

#: Rasterisation DPI for scanned PDFs. 200 balances OCR accuracy and speed.
DPI: int = _int("OCR_DPI", 200)

#: Hard cap on pages processed per document (protects the worker from a 200-page scan).
MAX_PAGES: int = _int("OCR_MAX_PAGES", 10)

#: Chars per page required before a PDF is treated as "digital" (has a text layer).
#: Scanned PDFs often carry a few stray characters (page numbers, producer marks),
#: so a small non-zero threshold is safer than `> 0`.
TEXT_LAYER_MIN_CHARS: int = _int("OCR_TEXT_LAYER_MIN_CHARS", 120)

#: Tokens whose vertical centres fall within this fraction of the median token
#: height are treated as the same visual line when reconstructing reading order.
LINE_TOLERANCE_RATIO: float = 0.6

IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp"})
PDF_MIME_TYPES = frozenset({"application/pdf"})
