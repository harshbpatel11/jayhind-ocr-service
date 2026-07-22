"""Unit tests for document ingestion (PDF text layer, images, bad input)."""

from __future__ import annotations

import io

import numpy as np
import pytest

from app.config import Settings
from app.domain.pipeline_types import SourceKind
from app.ingestion.document_loader import DocumentLoaderImpl, UnsupportedDocument


def _loader() -> DocumentLoaderImpl:
    return DocumentLoaderImpl(Settings())


def test_digital_pdf_uses_text_layer():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 60), "TAX INVOICE\nGSTIN: 24AJGPP6816J1ZY\nGrand Total 59000", fontsize=11)
    data = doc.tobytes()
    doc.close()

    loaded = _loader().load(data, "application/pdf", "invoice.pdf")
    assert loaded.method == "pdf-text"
    assert loaded.page_count == 1
    assert loaded.pages[0].has_text_layer
    assert loaded.pages[0].source == SourceKind.PDF_TEXT


def test_png_image_is_single_ocr_page():
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((120, 200, 3), 255, dtype=np.uint8)).save(buffer, "PNG")
    loaded = _loader().load(buffer.getvalue(), "image/png", "scan.png")
    assert loaded.method == "ocr"
    assert loaded.page_count == 1
    assert loaded.pages[0].source == SourceKind.IMAGE


def test_pdf_sniffed_by_magic_bytes_without_extension():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((40, 60), "hello", fontsize=11)
    data = doc.tobytes()
    doc.close()
    loaded = _loader().load(data, "application/octet-stream", "file.bin")
    assert loaded.page_count == 1


def test_garbage_raises_terminal():
    with pytest.raises(UnsupportedDocument):
        _loader().load(b"not an image or pdf", "application/octet-stream", "x.bin")


def test_empty_raises():
    with pytest.raises(UnsupportedDocument):
        _loader().load(b"", "image/png", "empty.png")
