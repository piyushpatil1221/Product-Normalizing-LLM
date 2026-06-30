"""
logger.py — Structured logging configuration for the pipeline.

Sets up both file and console handlers with colourised output
so engineers get clear, readable logs in the terminal while a
persistent log file is written to logs/pipeline.log.
"""

import logging
import sys
from pathlib import Path

from src.config import settings

# ── ANSI colour codes for console output ─────────────────────────────────────
_RESET = "\033[0m"
_COLOURS: dict[int, str] = {
    logging.DEBUG: "\033[36m",    # Cyan
    logging.INFO: "\033[32m",     # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",    # Red
    logging.CRITICAL: "\033[35m", # Magenta
}


class ColouredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI colour codes to console log output."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, _RESET)
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


def get_logger(name: str = "pipeline") -> logging.Logger:
    """
    Return a named logger pre-configured with file and console handlers.

    Args:
        name: Logger name (typically the calling module's __name__).

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── File Handler ──────────────────────────────────────────────────────────
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # ── Console Handler ───────────────────────────────────────────────────────
    console_formatter = ColouredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ── Module-level default logger ───────────────────────────────────────────────
log = get_logger("pipeline")
