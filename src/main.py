"""
main.py — CLI entry point for the LLM Product Normalization Pipeline.

Usage:
    python -m src.main                          # use default input path
    python -m src.main --input data/custom.csv  # custom input
    python -m src.main --limit 50               # process only 50 records

The pipeline runs sequentially:
    1. Preprocessing  — clean and deduplicate raw descriptions
    2. LLM Parsing    — extract structured data via Ollama
    3. Validation     — enforce Pydantic schema and business rules
    4. Post-processing — normalise brands, categories, availability
    5. Export         — write clean_products.csv and errors.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.config import settings
from src.exporter import export_results
from src.llm_parser import LLMParser, OllamaClient
from src.logger import get_logger
from src.postprocess import postprocess_batch
from src.preprocess import run_preprocessing
from src.schemas import ErrorRecord, NormalizedProduct, RawProduct
from src.validator import validate_product

log = get_logger("pipeline.main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-Powered Product Data Normalization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.input_path,
        help="Path to the raw messy products CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.output_path,
        help="Path for the clean output CSV",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=settings.error_path,
        help="Path for the errors CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records (for debugging)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.ollama_model,
        help="Ollama model identifier",
    )
    return parser.parse_args()


def run_pipeline(
    input_path: Path,
    output_path: Path,
    error_path: Path,
    limit: int | None = None,
    model: str = settings.ollama_model,
) -> dict:
    """
    Execute the full normalization pipeline end-to-end.

    Args:
        input_path:  Raw CSV path.
        output_path: Clean output CSV path.
        error_path:  Error CSV path.
        limit:       Max records to process (None = all).
        model:       Ollama model name.

    Returns:
        Summary dict with keys: total, success, errors, duration_seconds.
    """
    pipeline_start = time.perf_counter()
    log.info("=" * 60)
    log.info("LLM Product Normalization Pipeline — STARTING")
    log.info(f"  Input   : {input_path}")
    log.info(f"  Output  : {output_path}")
    log.info(f"  Errors  : {error_path}")
    log.info(f"  Model   : {model}")
    log.info("=" * 60)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    log.info("[Step 1/5] Preprocessing raw data…")
    raw_products: list[RawProduct] = run_preprocessing(input_path)

    if limit:
        raw_products = raw_products[:limit]
        log.info(f"[Step 1/5] Limiting to {limit} records as requested")

    log.info(f"[Step 1/5] {len(raw_products)} records ready for LLM")

    # ── Step 2: LLM Parsing ───────────────────────────────────────────────────
    log.info("[Step 2/5] Initialising Ollama LLM parser…")
    client = OllamaClient(model=model)

    if not client.is_available():
        log.error(
            f"Model '{model}' is not available in Ollama. "
            f"Run: ollama pull {model}"
        )
        sys.exit(1)

    parser = LLMParser(client=client)

    successes: list[NormalizedProduct] = []
    failures: list[ErrorRecord] = []

    log.info("[Step 2/5] Running LLM inference…")
    with tqdm(
        total=len(raw_products),
        desc="Normalizing",
        unit="product",
        colour="green",
        dynamic_ncols=True,
    ) as pbar:
        for raw in raw_products:
            log.debug(f"Processing ID={raw.id}: {raw.raw_description[:60]}…")
            llm_data, retry_count, llm_raw = parser.parse(
                raw.raw_description, product_id=raw.id
            )

            # ── Step 3: Validation ────────────────────────────────────────────
            result = validate_product(raw, llm_data or {}, retry_count, llm_raw)

            if isinstance(result, NormalizedProduct):
                successes.append(result)
            else:
                # llm_data was None — route directly to errors
                if llm_data is None:
                    from src.schemas import ErrorRecord as ER
                    failures.append(
                        ER(
                            id=raw.id,
                            raw_description=raw.raw_description,
                            error_type="json_parse_error",
                            error_detail="LLM returned no valid JSON after all retries",
                            retry_count=retry_count,
                            llm_raw_output=llm_raw,
                        )
                    )
                else:
                    failures.append(result)

            pbar.set_postfix(
                success=len(successes),
                errors=len(failures),
                refresh=False,
            )
            pbar.update(1)

    log.info(f"[Step 3/5] Validation complete — {len(successes)} passed, {len(failures)} failed")

    # ── Step 4: Post-processing ────────────────────────────────────────────────
    log.info("[Step 4/5] Post-processing normalized records…")
    successes = postprocess_batch(successes)

    # ── Step 5: Export ─────────────────────────────────────────────────────────
    log.info("[Step 5/5] Exporting CSVs…")
    export_results(successes, failures, output_path, error_path)

    duration = time.perf_counter() - pipeline_start
    total = len(raw_products)
    success_rate = (len(successes) / total * 100) if total else 0.0

    summary = {
        "total": total,
        "success": len(successes),
        "errors": len(failures),
        "success_rate": round(success_rate, 2),
        "duration_seconds": round(duration, 2),
    }

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Total processed : {total}")
    log.info(f"  Successful      : {len(successes)}")
    log.info(f"  Failed          : {len(failures)}")
    log.info(f"  Success rate    : {success_rate:.1f}%")
    log.info(f"  Duration        : {duration:.1f}s")
    log.info("=" * 60)

    return summary


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        error_path=args.errors,
        limit=args.limit,
        model=args.model,
    )


if __name__ == "__main__":
    main()
