"""Typed pipeline errors carrying the HTTP status the ERP hub depends on.

The hub's OCR proxy preserves a strict split end-to-end:

    4xx  → the DOCUMENT is unreadable → terminal (the child fails the scan)
    5xx  → an ENGINE / infra error    → retryable

so the pipeline must classify every failure into exactly one of these two.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base pipeline error with an attached HTTP status."""

    http_status: int = 500

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        if http_status is not None:
            self.http_status = http_status


class TerminalDocumentError(PipelineError):
    """The document itself cannot be read — never retry (HTTP 4xx)."""

    http_status = 400


class RetryableEngineError(PipelineError):
    """An engine/infrastructure failure — retrying may succeed (HTTP 5xx)."""

    http_status = 503
