"""
exporter.py — CSV export module for the normalization pipeline.

Writes:
    - clean_products.csv  : successfully normalized records
    - errors.csv          : records that failed LLM parsing or validation

Uses pandas for reliable UTF-8 CSV generation with consistent column ordering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.logger import get_logger
from src.schemas import ErrorRecord, NormalizedProduct

log = get_logger(__name__)

# ── Column ordering for the output CSV ────────────────────────────────────────
CLEAN_COLUMNS: list[str] = [
    "id",
    "product_name",
    "brand",
    "category",
    "price",
    "currency",
    "offer",
    "availability",
    "delivery",
    "seller",
    "confidence",
    "processing_status",
    "raw_description",
]

ERROR_COLUMNS: list[str] = [
    "id",
    "raw_description",
    "error_type",
    "error_detail",
    "retry_count",
    "llm_raw_output",
]


def products_to_dataframe(products: list[NormalizedProduct]) -> pd.DataFrame:
    """
    Convert a list of :class:`NormalizedProduct` objects to a DataFrame.

    Args:
        products: Successfully validated and post-processed records.

    Returns:
        DataFrame with columns in :data:`CLEAN_COLUMNS` order.
    """
    if not products:
        return pd.DataFrame(columns=CLEAN_COLUMNS)
    rows = [p.model_dump() for p in products]
    df = pd.DataFrame(rows)
    # Ensure all expected columns are present (fill missing with None)
    for col in CLEAN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[CLEAN_COLUMNS]


def errors_to_dataframe(errors: list[ErrorRecord]) -> pd.DataFrame:
    """
    Convert a list of :class:`ErrorRecord` objects to a DataFrame.

    Args:
        errors: Records that failed during LLM parsing or validation.

    Returns:
        DataFrame with columns in :data:`ERROR_COLUMNS` order.
    """
    if not errors:
        return pd.DataFrame(columns=ERROR_COLUMNS)
    rows = [e.model_dump() for e in errors]
    df = pd.DataFrame(rows)
    for col in ERROR_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[ERROR_COLUMNS]


def export_clean_csv(
    products: list[NormalizedProduct],
    output_path: Path | str | None = None,
) -> Path:
    """
    Write successfully normalized products to ``clean_products.csv``.

    Args:
        products:    List of :class:`NormalizedProduct` instances.
        output_path: Override default output path from settings.

    Returns:
        Absolute path to the written file.
    """
    path = Path(output_path or settings.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = products_to_dataframe(products)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat

    log.info(f"Exported {len(products)} clean records -> {path}")
    return path


def export_error_csv(
    errors: list[ErrorRecord],
    error_path: Path | str | None = None,
) -> Path:
    """
    Write failed records to ``errors.csv``.

    Args:
        errors:     List of :class:`ErrorRecord` instances.
        error_path: Override default error path from settings.

    Returns:
        Absolute path to the written file.
    """
    path = Path(error_path or settings.error_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = errors_to_dataframe(errors)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    log.info(f"Exported {len(errors)} error records -> {path}")
    return path


def export_results(
    products: list[NormalizedProduct],
    errors: list[ErrorRecord],
    output_path: Path | str | None = None,
    error_path: Path | str | None = None,
) -> dict[str, Path]:
    """
    Export both clean and error CSVs in one call.

    Returns:
        Dict with keys ``"clean"`` and ``"errors"`` pointing to output paths.
    """
    clean_path = export_clean_csv(products, output_path)
    err_path = export_error_csv(errors, error_path)

    total = len(products) + len(errors)
    success_rate = (len(products) / total * 100) if total else 0.0
    log.info(
        f"Pipeline complete — "
        f"Total: {total} | Success: {len(products)} | "
        f"Errors: {len(errors)} | Rate: {success_rate:.1f}%"
    )
    return {"clean": clean_path, "errors": err_path}
