"""Unit tests for the OpenCV preprocessing chain."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.preprocessing.image_preprocessor import ImagePreprocessorImpl, _estimate_skew_angle

cv2 = pytest.importorskip("cv2")


def _page_with_text(angle: float = 0.0) -> np.ndarray:
    img = np.full((400, 800, 3), 255, dtype=np.uint8)
    for y in range(80, 320, 40):  # horizontal text-like bars
        cv2.rectangle(img, (60, y), (740, y + 12), (0, 0, 0), -1)
    if angle:
        matrix = cv2.getRotationMatrix2D((400, 200), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (800, 400), borderValue=(255, 255, 255))
    return img


def test_preprocess_returns_rgb_same_shape_family():
    out = ImagePreprocessorImpl(Settings()).preprocess(_page_with_text(), dpi=200)
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8


def test_low_dpi_image_is_upscaled():
    small = np.full((200, 300, 3), 255, dtype=np.uint8)
    out = ImagePreprocessorImpl(Settings(min_short_edge_px=1000)).preprocess(small, dpi=72)
    assert min(out.shape[:2]) >= 1000


def test_skew_angle_detected():
    angle = _estimate_skew_angle(cv2.cvtColor(_page_with_text(angle=5.0), cv2.COLOR_RGB2GRAY))
    assert angle is not None
    assert abs(abs(angle) - 5.0) < 2.0


def test_disabled_preprocess_is_passthrough():
    img = _page_with_text()
    out = ImagePreprocessorImpl(Settings(preprocess_enabled=False)).preprocess(img, dpi=200)
    assert np.array_equal(out, img)
