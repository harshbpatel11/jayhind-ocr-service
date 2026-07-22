"""Small API response schemas that sit alongside the domain contract."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """``GET /health`` payload."""

    status: str = "ok"
    engine: str
    extractor: str
    reader_ready: bool
    extractor_ready: bool
    version: str
    gpu: bool = False
