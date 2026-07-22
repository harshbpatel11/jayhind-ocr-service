"""OpenCV image cleanup: deskew, denoise, contrast, adaptive threshold, resize.

Each step is a small pure(-ish) function over a NumPy array and is individually
toggleable via :class:`~app.config.Settings`, so the pipeline can tune what a
given reader wants (a VL model likes clean greyscale; a classic OCR backend likes
a hard binary image). Every step degrades gracefully — if a transform fails or
OpenCV is unavailable it returns the input unchanged rather than dropping a page.

Order matters and follows document-imaging best practice:
    resize-if-low-DPI → grayscale → contrast (CLAHE) → denoise → deskew
    → [optional] adaptive threshold.
"""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.domain.interfaces import ImagePreprocessor
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:  # OpenCV is optional at import time so the module is testable without it.
    import cv2

    _HAVE_CV2 = True
except Exception:  # pragma: no cover - environment guard
    cv2 = None  # type: ignore[assignment]
    _HAVE_CV2 = False


class ImagePreprocessorImpl(ImagePreprocessor):
    """Configurable OpenCV preprocessing chain."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def preprocess(self, image: np.ndarray, dpi: int) -> np.ndarray:
        if not self._s.preprocess_enabled or not _HAVE_CV2:
            return image
        try:
            work = self._resize_for_dpi(image, dpi)
            gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
            if self._s.contrast_enabled:
                gray = self._enhance_contrast(gray)
            if self._s.denoise_enabled:
                gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
            if self._s.deskew_enabled:
                gray = self._deskew(gray)
            if self._s.adaptive_threshold_enabled:
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
                )
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        except Exception as exc:  # never fail a page on a cosmetic transform
            logger.warning("preprocess failed, using raw image: %s", exc)
            return image

    # -- steps ----------------------------------------------------------------
    def _resize_for_dpi(self, image: np.ndarray, dpi: int) -> np.ndarray:
        """Upscale low-DPI pages, downscale oversized ones (memory + speed)."""
        h, w = image.shape[:2]
        short_edge = min(h, w)
        long_edge = max(h, w)
        scale = 1.0
        if short_edge < self._s.min_short_edge_px:
            scale = self._s.min_short_edge_px / float(short_edge)
        if long_edge * scale > self._s.max_long_edge_px:
            scale = self._s.max_long_edge_px / float(long_edge)
        if abs(scale - 1.0) < 0.02:
            return image
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interp)

    @staticmethod
    def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """CLAHE — local contrast that survives uneven phone-photo lighting."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """Estimate the dominant text angle and rotate the page level.

        The skew angle is the median orientation of long near-horizontal line
        segments found by a probabilistic Hough transform on the page edges —
        robust to sparse invoices where a min-area-rect over all ink would be
        dominated by a logo or a stamp.
        """
        angle = _estimate_skew_angle(gray)
        if angle is None or abs(angle) < 0.3:
            return gray
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(
            gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )


def _estimate_skew_angle(gray: np.ndarray) -> float | None:
    """Median skew angle in degrees (positive = counter-clockwise), or ``None``."""
    if not _HAVE_CV2:
        return None
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=200, minLineLength=gray.shape[1] // 3, maxLineGap=20
    )
    if lines is None:
        return None
    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        deg = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if -45.0 < deg < 45.0:  # near-horizontal text lines only
            angles.append(deg)
    if not angles:
        return None
    return float(np.median(angles))
