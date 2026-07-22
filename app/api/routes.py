"""HTTP routes: ``GET /health`` and ``POST /parse`` (+ ``/extract`` for QA).

``/parse`` returns the ERP contract as **plain JSON** (not a stream): this sidecar
is loopback-only, so there is no proxy idle-timeout to work around, and the hub's
proxy consumes ``response.json()`` directly. The 4xx-terminal / 5xx-retryable
split the hub relies on is carried by the real HTTP status via
:class:`PipelineError`.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile

from app import __version__
from app.api.schemas import HealthResponse
from app.container import Container
from app.pipeline.errors import PipelineError
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def get_container(request: Request) -> Container:
    """FastAPI dependency: the singleton container built at startup."""
    return request.app.state.container


def _check_auth(container: Container, authorization: str) -> None:
    """Enforce the bearer key only when one is configured (loopback ⇒ off)."""
    key = container.settings.api_key
    if not key:
        return
    expected = f"Bearer {key}"
    if not hmac.compare_digest(authorization or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _is_ready(component: object) -> bool:
    checker = getattr(component, "is_ready", None)
    return bool(checker()) if callable(checker) else True


@router.get("/health", response_model=HealthResponse)
def health(container: Container = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine=container.reader.engine,
        extractor=container.extractor.name,
        reader_ready=_is_ready(container.reader),
        extractor_ready=_is_ready(container.extractor),
        version=__version__,
        gpu=False,
    )


@router.post("/parse")
async def parse(
    request: Request,
    file: UploadFile = File(...),
    authorization: str = Header(default=""),
    container: Container = Depends(get_container),
) -> dict:
    _check_auth(container, authorization)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > container.settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Document is larger than {container.settings.max_upload_mb} MB",
        )
    try:
        result = await container.pipeline.run(data, file.content_type or "", file.filename or "upload")
    except PipelineError as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected → retryable
        logger.exception("unexpected pipeline error")
        raise HTTPException(status_code=503, detail=f"OCR service error: {exc}") from exc
    return result.model_dump(by_alias=True)


@router.post("/extract")
async def extract(
    file: UploadFile = File(...),
    authorization: str = Header(default=""),
    container: Container = Depends(get_container),
) -> dict:
    """Raw reader passthrough (reading-order text + markdown) — for QA/debugging."""
    _check_auth(container, authorization)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    import time

    def _run() -> dict:
        started = time.monotonic()
        document = container.loader.load(data, file.content_type or "", file.filename or "upload")
        for page in document.pages:
            if not page.has_text_layer:
                page.image = container.preprocessor.preprocess(page.image, page.dpi)
        reader_output = container.reader.read(document.pages)
        return {
            "method": document.method,
            "engine": reader_output.engine,
            "pageCount": document.page_count,
            "durationMs": int((time.monotonic() - started) * 1000),
            "text": reader_output.text,
            "markdown": reader_output.markdown,
        }

    try:
        import anyio

        return await anyio.to_thread.run_sync(_run)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"OCR service error: {exc}") from exc
