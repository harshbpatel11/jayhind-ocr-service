"""Decode an upload into per-page RGB images.

Handles every input the spec requires — PDF (digital or scanned), PNG, JPG,
TIFF (incl. multi-page), and phone photos. For a **digital** PDF page the
embedded text layer is carried alongside the raster so the pipeline can skip OCR
on pages that already have selectable text (the ``pdf-text`` fast path).

The only framework dependency is PyMuPDF (PDF raster + text) and Pillow (image
decode); everything else is NumPy. Nothing here touches the network.
"""

from __future__ import annotations

import io

import numpy as np

from app.config import Settings
from app.domain.interfaces import DocumentLoader
from app.domain.pipeline_types import LoadedDocument, PageImage, SourceKind
from app.utils.logging import get_logger

logger = get_logger(__name__)


class UnsupportedDocument(Exception):
    """The upload is neither a readable image nor a PDF (terminal, HTTP 4xx)."""


class DocumentLoaderImpl(DocumentLoader):
    """Default loader: PyMuPDF for PDFs, Pillow for images."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # -- public API -----------------------------------------------------------
    def load(self, data: bytes, content_type: str, filename: str) -> LoadedDocument:
        if not data:
            raise UnsupportedDocument("Empty file")
        if self._is_pdf(data, content_type, filename):
            return self._load_pdf(data)
        return self._load_image(data)

    # -- type sniffing --------------------------------------------------------
    @staticmethod
    def _is_pdf(data: bytes, content_type: str, filename: str) -> bool:
        """PDF if the magic bytes, the content type, or the extension say so."""
        if data[:5] == b"%PDF-":
            return True
        ct = (content_type or "").lower()
        name = (filename or "").lower()
        return "pdf" in ct or name.endswith(".pdf")

    # -- PDF ------------------------------------------------------------------
    def _load_pdf(self, data: bytes) -> LoadedDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - environment guard
            raise UnsupportedDocument(f"PDF support unavailable: {exc}") from exc

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise UnsupportedDocument(f"Could not open PDF: {exc}") from exc

        pages: list[PageImage] = []
        all_have_text = True
        try:
            dpi = self._settings.pdf_dpi
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for index, page in enumerate(doc):
                if index >= self._settings.max_pages:
                    logger.warning("PDF truncated at %d pages", self._settings.max_pages)
                    break
                text_layer = (page.get_text("text") or "").strip()
                if len(text_layer) < self._settings.pdf_text_min_chars:
                    all_have_text = False
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = _pixmap_to_rgb(pixmap)
                pages.append(
                    PageImage(
                        index=index,
                        image=image,
                        dpi=dpi,
                        source=SourceKind.PDF_TEXT if text_layer else SourceKind.PDF_SCAN,
                        text_layer=text_layer,
                    )
                )
        finally:
            doc.close()

        if not pages:
            raise UnsupportedDocument("PDF has no pages")
        method = "pdf-text" if all_have_text else "ocr"
        return LoadedDocument(pages=pages, method=method)

    # -- images ---------------------------------------------------------------
    def _load_image(self, data: bytes) -> LoadedDocument:
        try:
            from PIL import Image, ImageOps, ImageSequence
        except ImportError as exc:  # pragma: no cover - environment guard
            raise UnsupportedDocument(f"Image support unavailable: {exc}") from exc

        try:
            pil = Image.open(io.BytesIO(data))
        except Exception as exc:
            raise UnsupportedDocument(f"Not a readable image or PDF: {exc}") from exc

        pages: list[PageImage] = []
        for index, frame in enumerate(ImageSequence.Iterator(pil)):
            if index >= self._settings.max_pages:
                break
            # Honour EXIF orientation (phone photos) then flatten to RGB.
            rgb = ImageOps.exif_transpose(frame).convert("RGB")
            pages.append(
                PageImage(
                    index=index,
                    image=np.asarray(rgb, dtype=np.uint8),
                    dpi=_guess_dpi(pil),
                    source=SourceKind.IMAGE,
                )
            )
        if not pages:
            raise UnsupportedDocument("Image had no decodable frames")
        return LoadedDocument(pages=pages, method="ocr")


def _pixmap_to_rgb(pixmap) -> np.ndarray:
    """PyMuPDF pixmap → contiguous (H, W, 3) uint8 RGB array."""
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    array = array.reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:  # RGBA → RGB
        array = array[:, :, :3]
    elif pixmap.n == 1:  # grey → RGB
        array = np.repeat(array, 3, axis=2)
    return np.ascontiguousarray(array)


def _guess_dpi(pil) -> int:
    """Best-effort DPI from image metadata; 0 means 'unknown' (upscale later)."""
    dpi = pil.info.get("dpi")
    if isinstance(dpi, (tuple, list)) and dpi:
        try:
            return int(round(float(dpi[0])))
        except (TypeError, ValueError):
            return 0
    return 0
