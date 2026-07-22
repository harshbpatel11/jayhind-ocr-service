"""Typed configuration (12-factor, env-driven).

Every tunable is an environment variable with a sensible default, validated once
at startup via ``pydantic-settings``. Grouped by pipeline stage so the knobs read
top-to-bottom in the same order the document flows.

The service is loopback-only by design (the ERP hub calls it on
``127.0.0.1:8100``), so auth is OFF unless ``OCR_API_KEY`` is set — matching the
hub proxy, which sends a bearer token only when ``OCR_SERVICE_KEY`` is configured.
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── server ────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8100
    #: Blank ⇒ no auth (loopback deployment). Set to require ``Bearer <key>``.
    api_key: str = ""
    max_upload_mb: int = 25
    #: Worker threads for the blocking pipeline; 1 keeps peak RAM predictable when
    #: both PaddleOCR-VL and the 8B model are resident.
    pipeline_workers: int = 1

    # ── ingestion ─────────────────────────────────────────────────────────────
    #: DPI to rasterise PDF pages at. 200 balances legibility vs. memory; low-DPI
    #: pages are upscaled by the preprocessor.
    pdf_dpi: int = 200
    #: A digital-PDF page with at least this many embedded characters skips OCR.
    pdf_text_min_chars: int = 40
    max_pages: int = 100

    # ── preprocessing ─────────────────────────────────────────────────────────
    preprocess_enabled: bool = True
    deskew_enabled: bool = True
    denoise_enabled: bool = True
    #: Adaptive-threshold (binarisation) can hurt VL models that expect greyscale;
    #: off by default, on for classic OCR backends.
    adaptive_threshold_enabled: bool = False
    contrast_enabled: bool = True
    #: Upscale so the shorter document edge is at least this many pixels (~150 DPI
    #: on an A4 page) — recovers detail from low-DPI phone photos.
    min_short_edge_px: int = 1000
    max_long_edge_px: int = 4000

    # ── reader (OCR + layout) ─────────────────────────────────────────────────
    #: ``rapidocr`` (default — PP-OCR via ONNX Runtime, the CPU-stable reader on
    #: ARM64), ``paddleocr-vl`` / ``paddleocr`` (PaddlePaddle-native; accurate but
    #: segfault on this aarch64 CPU build — use where paddle inference works), or
    #: ``null`` (deterministic stub for tests / digital-PDF text only).
    reader_engine: str = "rapidocr"
    #: Document-orientation classifier (rotates a sideways/upside-down page level).
    #: Textline-angle handling is internal to PaddleOCR-VL.
    use_doc_orientation: bool = True
    #: Cap the CPU threads Paddle uses so a busy box stays responsive.
    reader_cpu_threads: int = 4

    # ── extractor (LLM) ───────────────────────────────────────────────────────
    #: ``qwen`` (default, local Qwen3-8B via llama.cpp) or ``rules`` (deterministic
    #: fallback that structures purely from the rule hints — no model needed).
    extractor_engine: str = "qwen"
    #: Local GGUF path (downloaded by scripts/download_models.sh).
    llm_model_path: str = "models/Qwen3-8B-Q4_K_M.gguf"
    llm_context_tokens: int = 8192
    llm_max_output_tokens: int = 3072
    #: 0.0 = greedy/deterministic — the right choice for structured extraction.
    llm_temperature: float = 0.0
    llm_threads: int = 4
    #: Constrain generation to valid JSON with a GBNF grammar (llama.cpp) so the
    #: model *cannot* emit prose. Falls back to prompt-only if unsupported.
    llm_json_grammar: bool = True
    #: Characters of reader markdown fed to the model (keeps the prompt in-context).
    llm_input_char_budget: int = 14000

    # ── validation / scoring ──────────────────────────────────────────────────
    #: Rupee tolerance when checking that extracted amounts foot (mirrors the ERP).
    amount_tolerance: float = 1.0

    # ── artifacts / debugging ─────────────────────────────────────────────────
    save_artifacts: bool = False
    artifacts_dir: str = "artifacts"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolved_model_path(self) -> str:
        """Absolute path to the GGUF weights (relative paths are repo-rooted)."""
        if os.path.isabs(self.llm_model_path):
            return self.llm_model_path
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(repo_root, self.llm_model_path)


def load_settings() -> Settings:
    """Build a :class:`Settings` instance (call once at startup)."""
    return Settings()
