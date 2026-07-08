"""FastAPI entry point for the invoice OCR sidecar.

The NestJS backend POSTs an invoice file here and receives reading-order text
plus per-token boxes/confidences. Nothing about invoices (GSTINs, totals,
parties) is understood at this layer — structuring happens in the backend so the
extraction engine stays swappable. See INVOICE_SCANNING_PLAN.md §3.1.
"""
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config
from .extractor import OcrUnavailable, UnsupportedFileType, extract, ocr_available
from .models import ExtractResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr-service")

app = FastAPI(title="Invoice OCR Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", ocr_available=ocr_available(), gpu=config.USE_GPU)


@app.post("/extract", response_model=ExtractResponse)
async def extract_document(file: UploadFile = File(...)) -> ExtractResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    content_type = (file.content_type or "").lower()
    try:
        result = extract(data, content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except OcrUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # unreadable/corrupt document
        logger.exception("extraction failed for %s", file.filename)
        raise HTTPException(status_code=400, detail=f"Could not read document: {exc}") from exc

    logger.info(
        "extracted %s via %s: %d page(s), %d chars in %dms",
        file.filename, result["method"], result["pageCount"], len(result["text"]), result["durationMs"],
    )
    return ExtractResponse(**result)
