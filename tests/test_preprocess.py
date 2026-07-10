"""Image preprocessing (OpenCV) — deskew / contrast / denoise before OCR."""
import pathlib

import cv2
import numpy as np
import pytest
from PIL import Image

import app.config as config
from app.extractor import _deskew_gray, _preprocess_for_ocr

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _skew_magnitude(gray) -> float:
    """Residual page tilt in degrees, folded into (-45, 45] so it is robust to
    OpenCV's version-dependent minAreaRect convention."""
    mask = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    raw = cv2.minAreaRect(np.column_stack(np.where(mask > 0)))[-1]
    angle = raw % 90
    if angle > 45:
        angle -= 90
    return abs(angle)


def test_preprocess_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(config, "OCR_PREPROCESS", False)
    img = Image.new("RGB", (100, 100), "white")
    assert _preprocess_for_ocr(img) is img


def test_preprocess_returns_rgb(monkeypatch):
    monkeypatch.setattr(config, "OCR_PREPROCESS", True)
    out = _preprocess_for_ocr(Image.new("RGB", (200, 200), "white"))
    assert out.mode == "RGB"


def test_importing_extractor_loads_neither_ocr_engine():
    """onnxruntime and paddlepaddle SEGFAULT when loaded into the same process on
    ARM. Availability must therefore be probed with `find_spec`, and the real
    import deferred until an engine is actually used."""
    import subprocess
    import sys

    probe = (
        "import sys; import app.extractor;"
        "print('paddleocr' in sys.modules, 'onnxruntime' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    ).stdout.strip().splitlines()[-1]
    assert out == "False False", f"an OCR engine was imported at module load: {out}"


@pytest.mark.parametrize("tilt", [6.0, -6.0])
def test_deskew_corrects_a_real_page(tilt):
    """A real rendered invoice tilted by ±6° must come back near-horizontal —
    catches the sign/convention bug where the skew was doubled instead of fixed."""
    base = cv2.cvtColor(np.array(Image.open(FIXTURES / "clean_scan.png").convert("RGB")), cv2.COLOR_RGB2GRAY)
    m = cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), tilt, 1.0)
    tilted = cv2.warpAffine(base, m, (base.shape[1], base.shape[0]), borderValue=255)
    corrected = _deskew_gray(tilted)
    assert _skew_magnitude(corrected) < 1.5 < _skew_magnitude(tilted)
