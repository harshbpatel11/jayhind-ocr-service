"""Artifact repositories (repository pattern).

Two interchangeable implementations of :class:`ArtifactRepository`:

  * :class:`NullArtifactRepository` — the default no-op (nothing is stored).
  * :class:`FileArtifactRepository` — writes each request's page images, reader
    markdown and raw model output under ``artifacts/<request_id>/`` for
    debugging and accuracy work.

Because pipeline stages depend only on the ``ArtifactRepository`` Protocol, an
operator flips ``OCR_SAVE_ARTIFACTS=true`` to start archiving without any code
change — the composition root swaps the implementation.
"""

from __future__ import annotations

import os

from app.domain.interfaces import ArtifactRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class NullArtifactRepository(ArtifactRepository):
    """Stores nothing (default)."""

    def save_text(self, request_id: str, name: str, text: str) -> None:
        return None

    def save_bytes(self, request_id: str, name: str, data: bytes) -> None:
        return None


class FileArtifactRepository(ArtifactRepository):
    """Writes artifacts under ``<base_dir>/<request_id>/``."""

    def __init__(self, base_dir: str) -> None:
        self._base = base_dir

    def _dir(self, request_id: str) -> str:
        path = os.path.join(self._base, request_id)
        os.makedirs(path, exist_ok=True)
        return path

    def save_text(self, request_id: str, name: str, text: str) -> None:
        try:
            with open(os.path.join(self._dir(request_id), name), "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:  # never fail a request over a debug artifact
            logger.warning("could not save artifact %s/%s: %s", request_id, name, exc)

    def save_bytes(self, request_id: str, name: str, data: bytes) -> None:
        try:
            with open(os.path.join(self._dir(request_id), name), "wb") as handle:
                handle.write(data)
        except OSError as exc:
            logger.warning("could not save artifact %s/%s: %s", request_id, name, exc)
