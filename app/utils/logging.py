"""Minimal structured logging setup.

One place configures the root logger so every module can just call
``logging.getLogger(__name__)``. Kept dependency-free (stdlib ``logging``) — no
Winston/rotate equipment is needed for a single loopback sidecar.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotently configure root logging (safe to call from every entrypoint)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("OCR_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, lvl, logging.INFO))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    configure_logging()
    return logging.getLogger(name)
