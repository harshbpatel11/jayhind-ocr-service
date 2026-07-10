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

#: Which extraction engine to use for images / scanned PDFs:
#:   "onnx" (default) — RapidOCR on ONNX Runtime. Bundled PP-OCR ONNX models, no
#:               paddlepaddle. Measured ~8x faster than "classic" on CPU **and**
#:               more accurate on tilted/blurred photos.
#:   "classic"  — PaddleOCR (PP-OCRv5). The fallback engine; select it if a real
#:               invoice ever trips ONNX.
#:   "vl"       — PaddleOCR-VL, a local ~0.9B vision-language model. x86/GPU only
#:               (it segfaults on aarch64/CPU).
#: ⚠️ Engines are mutually exclusive per process — onnxruntime and paddlepaddle
#: segfault when loaded together on ARM, so we never import the one we don't use.
#: Digital PDFs always take the exact text-layer path regardless of this setting.
OCR_ENGINE: str = os.getenv("OCR_ENGINE", "onnx").strip().lower()

#: Model tier for the classic engine:
#:   "fast" (default) — PP-OCRv5 **mobile** models + 200 DPI. Stable and quick on
#:                aarch64/ARM (~13-19s/page).
#:   "accurate"       — PP-OCRv5 **server** models + 300 DPI. Better recognition,
#:                but MEASURED at >2 min/page on this 4-core ARM/CPU box and heavy
#:                on RAM, so it is opt-in rather than the default here. Prefer it on
#:                x86 / GPU / many-core hosts (set OCR_MODEL_TIER=accurate).
#: NB: PaddleOCR 3.x's PP-OCRv6 default **segfaults on ARM**, so we never use it.
#: (Most invoices are digital PDFs and take the exact text-layer path — no model
#: at all — so the tier only affects photographed/scanned inputs.)
OCR_MODEL_TIER: str = os.getenv("OCR_MODEL_TIER", "fast").strip().lower()
_ACCURATE = OCR_MODEL_TIER == "accurate"

#: Explicit model overrides win over the tier default.
DET_MODEL: str = os.getenv("OCR_DET_MODEL") or ("PP-OCRv5_server_det" if _ACCURATE else "PP-OCRv5_mobile_det")
REC_MODEL: str = os.getenv("OCR_REC_MODEL") or ("PP-OCRv5_server_rec" if _ACCURATE else "PP-OCRv5_mobile_rec")

#: Paddle's multi-threaded CPU inference segfaults on ARM, so we pin one thread
#: by default. On x86 raising this (2-4) is a straight speed win.
CPU_THREADS: int = _int("OCR_CPU_THREADS", 1)

#: Rasterisation DPI for scanned PDFs. 300 on the accurate tier (sharper glyphs),
#: 200 on the fast tier.
DPI: int = _int("OCR_DPI", 300 if _ACCURATE else 200)

#: PaddleOCR-VL model name (used only when OCR_ENGINE="vl").
VL_REC_MODEL: str = os.getenv("OCR_VL_MODEL", "PaddleOCR-VL-0.9B")

#: Clean up photographed / scanned images before OCR (deskew + adaptive contrast
#: + gentle denoise via OpenCV). Helps recognition on phone photos and faxed
#: paper; harmless on clean scans. Applies only to the OCR path, never digital
#: PDFs. Set OCR_PREPROCESS=0 to disable.
OCR_PREPROCESS: bool = _bool("OCR_PREPROCESS", True)

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
