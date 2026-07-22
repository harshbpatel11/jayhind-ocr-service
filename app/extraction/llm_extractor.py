"""Qwen3-8B local-CPU extractor via llama.cpp (GGUF).

llama.cpp is the right CPU runtime for an 8B model on ARM64: quantised GGUF
weights (Q4_K_M ≈ 4.7 GB) fit comfortably in 24 GB RAM, it uses ARM NEON, and it
needs neither torch nor a GPU. Generation is greedy (``temperature=0``) and, when
``llm_json_grammar`` is on, constrained by a JSON-Schema-derived grammar so the
model can only emit a valid contract-shaped object.

Accuracy over speed, as required: on a 4-vCPU box an 8B model runs at only a few
tokens/second, so a full extraction takes tens of seconds to minutes. That is the
deliberate trade-off — see the README's performance note.
"""

from __future__ import annotations

import os

from app.config import Settings
from app.domain.interfaces import InvoiceExtractor
from app.domain.pipeline_types import ExtractionContext
from app.extraction.json_utils import parse_json_object
from app.extraction.prompt import (
    SYSTEM_PROMPT,
    build_json_schema,
    build_user_prompt,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LlmExtractorError(RuntimeError):
    """Raised when the model cannot be loaded or fails to produce JSON."""


class QwenLlmExtractor(InvoiceExtractor):
    """Local Qwen3-8B-Instruct extractor backed by llama-cpp-python."""

    name = "qwen3-8b-instruct"

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._llm = None
        self._schema = build_json_schema()

    # -- lifecycle ------------------------------------------------------------
    def warm_up(self) -> None:
        if self._llm is not None:
            return
        model_path = self._s.resolved_model_path()
        if not os.path.exists(model_path):
            raise LlmExtractorError(
                f"GGUF model not found at {model_path}. "
                "Run scripts/download_models.sh or set OCR_LLM_MODEL_PATH."
            )
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment guard
            raise LlmExtractorError(f"llama-cpp-python is not installed: {exc}") from exc

        logger.info("loading Qwen GGUF (%s, %d threads) ...", model_path, self._s.llm_threads)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=self._s.llm_context_tokens,
            n_threads=self._s.llm_threads,
            n_batch=256,
            verbose=False,
        )
        logger.info("Qwen extractor ready.")

    def is_ready(self) -> bool:
        return self._llm is not None

    def _ensure(self):
        if self._llm is None:
            self.warm_up()
        return self._llm

    # -- extraction -----------------------------------------------------------
    def extract(self, context: ExtractionContext) -> dict:
        llm = self._ensure()
        user_prompt = build_user_prompt(
            context.reader.markdown or context.reader.text,
            context.hints,
            self._s.llm_input_char_budget,
        )
        response_format = {"type": "json_object"}
        if self._s.llm_json_grammar:
            response_format["schema"] = self._schema

        try:
            completion = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._s.llm_temperature,
                max_tokens=self._s.llm_max_output_tokens,
                response_format=response_format,
            )
        except Exception as exc:
            raise LlmExtractorError(f"LLM generation failed: {exc}") from exc

        text = completion["choices"][0]["message"]["content"] or ""
        try:
            return parse_json_object(text)
        except ValueError as exc:
            raise LlmExtractorError(f"model did not return valid JSON: {exc}") from exc
