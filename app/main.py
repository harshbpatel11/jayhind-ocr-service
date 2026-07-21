"""FastAPI entry point for the invoice OCR sidecar.

The NestJS backend POSTs an invoice file here and receives reading-order text
plus per-token boxes/confidences. Nothing about invoices (GSTINs, totals,
parties) is understood at this layer — structuring happens in the backend so the
extraction engine stays swappable. See INVOICE_SCANNING_PLAN.md §3.1.
"""
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config
from .extractor import OcrUnavailable, UnsupportedFileType, extract, ocr_available
from .models import ExtractResponse, HealthResponse, ParseResponse
from .structuring import parse_invoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr-service")

app = FastAPI(title="Invoice OCR Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ocr_available=ocr_available(),
        gpu=config.USE_GPU,
        engine=config.OCR_ENGINE,
        remote=config.OCR_REMOTE_URL if config.OCR_ENGINE == "remote" else "",
    )


async def _read_and_extract(file: UploadFile) -> dict:
    """Shared read + validate + OCR/text extraction for /extract and /parse."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > config.MAX_UPLOAD_BYTES:
        # 413 is a 4xx, so the backend treats it as terminal for this document
        # rather than retrying a file that will never fit.
        raise HTTPException(
            status_code=413,
            detail=f"Document is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    try:
        return extract(data, (file.content_type or "").lower())
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except OcrUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # unreadable/corrupt document
        logger.exception("extraction failed for %s", file.filename)
        raise HTTPException(status_code=400, detail=f"Could not read document: {exc}") from exc


@app.post("/extract", response_model=ExtractResponse)
async def extract_document(file: UploadFile = File(...)) -> ExtractResponse:
    """Raw reading-order text + token boxes (the extraction seam)."""
    result = await _read_and_extract(file)
    logger.info(
        "extracted %s via %s: %d page(s), %d chars in %dms",
        file.filename, result["method"], result["pageCount"], len(result["text"]), result["durationMs"],
    )
    return ExtractResponse(**result)


@app.post("/parse", response_model=ParseResponse)
async def parse_document(file: UploadFile = File(...)) -> ParseResponse:
    """Extraction + structuring: returns the structured `ExtractedInvoice` the
    NestJS matching layer consumes. All offline — nothing leaves this host."""
    started = time.monotonic()
    ocr = await _read_and_extract(file)
    invoice = parse_invoice(ocr)
    duration = int((time.monotonic() - started) * 1000)
    logger.info(
        "parsed %s via %s: seller=%r %d line(s), grand=%s in %dms",
        file.filename, ocr["method"], invoice["seller"]["name"],
        len(invoice["lineItems"]), invoice["totals"]["grandTotal"], duration,
    )
    return ParseResponse(
        method=ocr["method"],
        structuringMethod="rules",
        pageCount=ocr["pageCount"],
        durationMs=duration,
        text=ocr["text"],
        invoice=invoice,
    )
