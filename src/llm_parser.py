"""
llm_parser.py — Ollama LLM integration with prompt management and retry logic.

Architecture:
    PromptBuilder   — loads and formats the prompt template from disk
    OllamaClient    — thin wrapper around ollama.chat() with timeout handling
    LLMParser       — orchestrates prompt → LLM call → JSON parse → retry loop

Design decisions:
    - Prompts are stored in a plain text file (not hardcoded) so they can be
      updated without touching Python source.
    - The retry loop retries up to `settings.llm_retry_limit` times with
      exponential back-off before writing the record to errors.csv.
    - Temperature is set to 0.0 by default to maximise determinism.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import ollama

from src.config import settings
from src.logger import get_logger
from src.utils import safe_json_parse

log = get_logger(__name__)


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Loads a plain-text prompt template and injects runtime variables.

    The template uses Python's ``str.format_map()`` syntax, e.g.::

        RAW DESCRIPTION:
        {raw_description}

    Attributes:
        template_path: Path to the ``.txt`` prompt file.
        _template:     Cached template string (loaded on first use).
    """

    def __init__(self, template_path: Path | str | None = None) -> None:
        self.template_path = Path(template_path or settings.prompt_path)
        self._template: str | None = None

    def _load(self) -> str:
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {self.template_path}"
            )
        return self.template_path.read_text(encoding="utf-8")

    @property
    def template(self) -> str:
        if self._template is None:
            self._template = self._load()
        return self._template

    def build(self, raw_description: str) -> str:
        """
        Substitute ``{raw_description}`` in the template.

        Args:
            raw_description: Cleaned product text from the preprocessor.

        Returns:
            Ready-to-send prompt string.
        """
        return self.template.replace("{raw_description}", raw_description)

    def reload(self) -> None:
        """Force a reload of the template from disk (useful in hot-reload scenarios)."""
        self._template = self._load()
        log.debug("Prompt template reloaded from disk")


# ── Ollama Client ─────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around the Ollama Python SDK.

    Centralises model name, host, and generation parameters so they
    can be changed from a single location (settings).

    Args:
        model:  Ollama model identifier, e.g. ``"llama3.2"``.
        host:   Ollama server URL, e.g. ``"http://localhost:11434"``.
    """

    def __init__(
        self,
        model: str = settings.ollama_model,
        host: str = settings.ollama_host,
    ) -> None:
        self.model = model
        self.client = ollama.Client(host=host)
        log.info(f"OllamaClient initialised — model={model!r}, host={host!r}")

    def chat(self, prompt: str) -> str:
        """
        Send *prompt* as a user message and return the assistant's reply.

        Args:
            prompt: Full prompt text (system instructions + product description).

        Returns:
            Raw string response from the model.

        Raises:
            ollama.ResponseError: On API / model errors.
            Exception: Any other unexpected failure.
        """
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        )
        return response["message"]["content"]

    def is_available(self) -> bool:
        """
        Ping Ollama and check the target model is installed.

        Returns:
            ``True`` if the model is available, ``False`` otherwise.
        """
        try:
            models = self.client.list()
            names = [m["name"] for m in models.get("models", [])]
            available = any(self.model in n for n in names)
            if not available:
                log.warning(
                    f"Model {self.model!r} not found in Ollama. "
                    f"Available models: {names}"
                )
            return available
        except Exception as exc:  # noqa: BLE001
            log.error(f"Could not connect to Ollama: {exc}")
            return False


# ── LLM Parser ────────────────────────────────────────────────────────────────

class LLMParser:
    """
    Orchestrates the full LLM inference pipeline for a single product:

    1. Build the prompt via :class:`PromptBuilder`.
    2. Send it to Ollama via :class:`OllamaClient`.
    3. Extract JSON from the raw LLM response via ``safe_json_parse``.
    4. Retry up to ``settings.llm_retry_limit`` times on failure.
    5. Return the parsed dict or ``None`` if all retries exhausted.

    Args:
        client:  :class:`OllamaClient` instance (injected for testability).
        builder: :class:`PromptBuilder` instance (injected for testability).
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        builder: PromptBuilder | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.builder = builder or PromptBuilder()

    def parse(
        self, raw_description: str, product_id: int = 0
    ) -> tuple[dict[str, Any] | None, int, str | None]:
        """
        Parse a single product description with automatic retry.

        Args:
            raw_description: Cleaned text from the preprocessor.
            product_id:      Record ID used only for log messages.

        Returns:
            A 3-tuple:
                - ``parsed``: dict on success, ``None`` on failure
                - ``retry_count``: number of retry attempts made
                - ``llm_raw_output``: last raw LLM response (for error CSV)
        """
        prompt = self.builder.build(raw_description)
        retry_count = 0
        last_raw: str | None = None

        for attempt in range(1, settings.llm_retry_limit + 1):
            log.debug(f"[ID={product_id}] LLM attempt {attempt}/{settings.llm_retry_limit}")
            try:
                raw_output = self.client.chat(prompt)
                last_raw = raw_output
                parsed = safe_json_parse(raw_output)

                if parsed is not None:
                    log.debug(f"[ID={product_id}] JSON parsed successfully on attempt {attempt}")
                    return parsed, retry_count, last_raw

                log.warning(
                    f"[ID={product_id}] Attempt {attempt}: LLM returned non-JSON output"
                )

            except Exception as exc:  # noqa: BLE001
                log.error(f"[ID={product_id}] Attempt {attempt}: LLM call failed — {exc}")

            retry_count += 1
            if attempt < settings.llm_retry_limit:
                backoff = 2 ** (attempt - 1)  # 1s, 2s
                log.info(f"[ID={product_id}] Retrying in {backoff}s…")
                time.sleep(backoff)

        log.error(
            f"[ID={product_id}] All {settings.llm_retry_limit} attempts failed. "
            "Record will be written to errors.csv."
        )
        return None, retry_count, last_raw
