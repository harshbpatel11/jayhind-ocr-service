"""End-to-end checks against the FastAPI app.

The digital-PDF fixtures run everywhere. Image/scan fixtures need PaddleOCR, so
those tests skip automatically in a light install.
"""
import pathlib

import pytest
from fastapi.testclient import TestClient

from app.extractor import ocr_available
from app.main import app

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
client = TestClient(app)

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "purchase_digital.pdf").exists(),
    reason="run `python tests/make_fixtures.py` first",
)


def post(name: str, content_type: str):
    data = (FIXTURES / name).read_bytes()
    return client.post("/extract", files={"file": (name, data, content_type)})


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["ocr_available"], bool)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("purchase_digital.pdf", "24AAHCV3778L1ZQ"),   # supplier GSTIN
        ("sales_digital.pdf", "24AAACU6278M1ZV"),     # customer GSTIN
        ("interstate_igst.pdf", "29AAACL1745Q1Z0"),   # Karnataka seller
    ],
)
def test_digital_pdf_uses_text_layer_and_finds_gstin(name, expected):
    """Digital PDFs must take the no-OCR fast path and reproduce exact characters."""
    response = post(name, "application/pdf")
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "pdf-text"
    assert body["pageCount"] >= 1
    assert expected in body["text"].replace(" ", "")
    assert all(token["confidence"] == 1.0 for token in body["pages"][0]["tokens"])


def test_digital_pdf_preserves_reading_order():
    body = post("purchase_digital.pdf", "application/pdf").json()
    text = body["text"]
    # Header must come before the line items, which come before the totals.
    assert text.index("TAX INVOICE") < text.index("Logitech MX Master")
    assert text.index("Logitech MX Master") < text.index("Grand Total")


def test_tokens_carry_bounding_boxes():
    body = post("purchase_digital.pdf", "application/pdf").json()
    token = body["pages"][0]["tokens"][0]
    assert len(token["bbox"]) == 4
    x1, y1, x2, y2 = token["bbox"]
    assert x2 > x1 and y2 > y1


def test_unsupported_content_type_is_415():
    response = client.post("/extract", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_empty_file_is_400():
    response = client.post("/extract", files={"file": ("a.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_corrupt_pdf_is_400():
    response = client.post("/extract", files={"file": ("a.pdf", b"not really a pdf", "application/pdf")})
    assert response.status_code == 400


@pytest.mark.skipif(not ocr_available(), reason="PaddleOCR not installed (light install)")
@pytest.mark.parametrize("name", ["clean_scan.png", "low_quality_photo.png"])
def test_images_take_the_ocr_path(name):
    if not (FIXTURES / name).exists():
        pytest.skip(f"{name} fixture not generated")
    body = post(name, "image/png").json()
    assert body["method"] == "ocr"
    assert body["text"]
    assert all(0.0 <= t["confidence"] <= 1.0 for t in body["pages"][0]["tokens"])


def test_image_without_ocr_engine_returns_503():
    if ocr_available():
        pytest.skip("PaddleOCR installed — 503 path not reachable")
    if not (FIXTURES / "clean_scan.png").exists():
        pytest.skip("clean_scan.png fixture not generated")
    assert post("clean_scan.png", "image/png").status_code == 503
