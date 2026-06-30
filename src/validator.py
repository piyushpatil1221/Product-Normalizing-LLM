"""
validator.py — Pydantic validation layer between the LLM and the output CSV.

Responsibilities:
    - Accept the raw dict returned by the LLM parser
    - Attempt to create a validated :class:`ProductSchema` instance
    - Log and return :class:`ErrorRecord` objects for any failed validations
    - Enforce business rules (e.g., confidence threshold)
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.config import settings
from src.logger import get_logger
from src.schemas import ErrorRecord, NormalizedProduct, ProductSchema, RawProduct

log = get_logger(__name__)


def validate_product(
    raw: RawProduct,
    llm_data: dict[str, Any],
    retry_count: int = 0,
    llm_raw_output: str | None = None,
) -> NormalizedProduct | ErrorRecord:
    """
    Validate LLM output and return either a :class:`NormalizedProduct`
    or an :class:`ErrorRecord`.

    Validation steps:
        1. Pydantic schema validation (types, ranges, required fields).
        2. Confidence threshold check (must exceed ``settings.confidence_threshold``).

    Args:
        raw:            Original :class:`RawProduct` (holds id + raw text).
        llm_data:       Dict parsed from the LLM's JSON response.
        retry_count:    Number of LLM retries that occurred.
        llm_raw_output: Last raw string from the LLM (stored in error CSV).

    Returns:
        :class:`NormalizedProduct` on success, :class:`ErrorRecord` on failure.
    """
    try:
        product = ProductSchema(**llm_data)
    except ValidationError as exc:
        log.warning(
            f"[ID={raw.id}] Pydantic validation failed — "
            f"{exc.error_count()} error(s): {exc.errors(include_url=False)}"
        )
        return ErrorRecord(
            id=raw.id,
            raw_description=raw.raw_description,
            error_type="validation_error",
            error_detail=str(exc.errors(include_url=False)),
            retry_count=retry_count,
            llm_raw_output=llm_raw_output,
        )

    # Business rule: reject low-confidence extractions
    if product.confidence < settings.confidence_threshold:
        log.warning(
            f"[ID={raw.id}] Confidence {product.confidence:.2f} below "
            f"threshold {settings.confidence_threshold:.2f} — routing to errors"
        )
        return ErrorRecord(
            id=raw.id,
            raw_description=raw.raw_description,
            error_type="low_confidence",
            error_detail=(
                f"Confidence {product.confidence:.2f} < "
                f"threshold {settings.confidence_threshold:.2f}"
            ),
            retry_count=retry_count,
            llm_raw_output=llm_raw_output,
        )

    log.debug(f"[ID={raw.id}] Validation passed — confidence={product.confidence:.2f}")

    return NormalizedProduct(
        id=raw.id,
        raw_description=raw.raw_description,
        processing_status="success",
        **product.model_dump(),
    )


def validate_batch(
    raw_products: list[RawProduct],
    llm_results: list[tuple[dict[str, Any] | None, int, str | None]],
) -> tuple[list[NormalizedProduct], list[ErrorRecord]]:
    """
    Validate a batch of LLM results against their originating raw products.

    Args:
        raw_products: Source :class:`RawProduct` list (same length as llm_results).
        llm_results:  List of ``(parsed_dict, retry_count, llm_raw_output)`` tuples.

    Returns:
        2-tuple of (successes, failures).
    """
    successes: list[NormalizedProduct] = []
    failures: list[ErrorRecord] = []

    for raw, (llm_data, retry_count, llm_raw_output) in zip(raw_products, llm_results):
        if llm_data is None:
            # LLM never returned parseable JSON after all retries
            failures.append(
                ErrorRecord(
                    id=raw.id,
                    raw_description=raw.raw_description,
                    error_type="json_parse_error",
                    error_detail="LLM did not return valid JSON after all retries",
                    retry_count=retry_count,
                    llm_raw_output=llm_raw_output,
                )
            )
            continue

        result = validate_product(raw, llm_data, retry_count, llm_raw_output)
        if isinstance(result, NormalizedProduct):
            successes.append(result)
        else:
            failures.append(result)

    return successes, failures
