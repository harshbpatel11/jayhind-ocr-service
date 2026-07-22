"""FastAPI application factory + server entrypoint.

The composition root (:class:`~app.container.Container`) is built once and stored
on ``app.state``; a lifespan hook warms the heavy models on startup so the first
request is not penalised. Run it with::

    python -m app.main            # or: uvicorn app.main:app --host 127.0.0.1 --port 8100
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.api.routes import router
from app.config import load_settings
from app.container import Container
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: Container = app.state.container
    if os.getenv("OCR_WARM_UP", "true").lower() in ("1", "true", "yes"):
        logger.info("warming up models (reader=%s, extractor=%s) ...",
                    container.reader.engine, container.extractor.name)
        try:
            container.pipeline.warm_up()
            logger.info("warm-up complete.")
        except Exception as exc:
            # Don't crash the server: /health will report not-ready and the first
            # /parse will surface the real error with the right HTTP status.
            logger.error("warm-up failed (service will start degraded): %s", exc)
    yield


def create_app(container: Container | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a ``container`` to inject fakes in tests."""
    configure_logging()
    app = FastAPI(title="Jayhind Invoice-OCR", version=__version__, lifespan=_lifespan)
    app.state.container = container or Container()
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Console entrypoint: serve on the configured host/port."""
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("OCR_LOG_LEVEL", "info").lower(),
        workers=1,  # single worker: models are loaded once, in-process
    )


if __name__ == "__main__":
    main()
