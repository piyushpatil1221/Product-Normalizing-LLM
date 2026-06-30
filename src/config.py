"""
config.py — Centralized configuration for the LLM Product Normalizer pipeline.

Uses Pydantic BaseSettings to support environment variable overrides,
making the pipeline portable across local, Docker, and CI environments.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ── Resolve project root (src/ lives one level inside project root) ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    All runtime settings for the pipeline.

    Can be overridden via environment variables or a .env file placed
    at the project root.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    ollama_model: str = Field(default="llama3.2", description="Ollama model identifier")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server base URL")
    llm_temperature: float = Field(default=0.0, description="LLM sampling temperature (0 = deterministic)")
    llm_max_tokens: int = Field(default=512, description="Maximum tokens the LLM may generate")
    llm_retry_limit: int = Field(default=3, description="Number of retry attempts on bad LLM output")

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="Directory for CSV files")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", description="Directory for log files")
    prompt_path: Path = Field(
        default=PROJECT_ROOT / "src" / "prompts" / "normalization_prompt.txt",
        description="Path to the system prompt template",
    )

    # ── Files ─────────────────────────────────────────────────────────────────
    input_file: str = Field(default="messy_products.csv", description="Raw input CSV filename")
    output_file: str = Field(default="clean_products.csv", description="Cleaned output CSV filename")
    error_file: str = Field(default="errors.csv", description="Error log CSV filename")
    log_file: str = Field(default="pipeline.log", description="Pipeline log filename")

    # ── Processing ────────────────────────────────────────────────────────────
    batch_size: int = Field(default=10, description="Records per processing batch (future use)")
    max_workers: int = Field(default=1, description="Parallel workers (keep 1 for Ollama stability)")
    confidence_threshold: float = Field(default=0.5, description="Minimum LLM confidence to accept record")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI bind port")

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def input_path(self) -> Path:
        return self.data_dir / self.input_file

    @property
    def output_path(self) -> Path:
        return self.data_dir / self.output_file

    @property
    def error_path(self) -> Path:
        return self.data_dir / self.error_file

    @property
    def log_path(self) -> Path:
        return self.log_dir / self.log_file


# ── Singleton instance used throughout the project ───────────────────────────
settings = Settings()

# Ensure required directories exist at import time
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
