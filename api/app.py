"""
api/app.py — FastAPI backend for the LLM Product Normalization Pipeline.

Endpoints:
    GET  /health           — liveness check; confirms Ollama is reachable
    POST /normalize        — normalize a single raw product description
    POST /upload           — upload a CSV/JSON/JSONL/Excel file, normalize all rows

All endpoints return JSON. Errors are returned as standard HTTP error
responses with descriptive detail messages.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import settings
from src.exporter import errors_to_dataframe, products_to_dataframe
from src.llm_parser import LLMParser, OllamaClient
from src.logger import get_logger
from src.postprocess import postprocess_batch, postprocess_product
from src.preprocess import (
    build_description_from_columns,
    clean_description,
    detect_description_columns,
    preprocess_dataframe,
)
from src.schemas import ErrorRecord, NormalizedProduct, RawProduct
from src.validator import validate_product

log = get_logger("api")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Product Normalizer API",
    description=(
        "Offline LLM-powered API that converts messy e-commerce product "
        "descriptions into clean, structured JSON using Ollama + llama3.2."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared LLM parser instance (created once at startup) ──────────────────────
_parser: LLMParser | None = None


def get_parser() -> LLMParser:
    global _parser
    if _parser is None:
        _parser = LLMParser()
    return _parser


# ── Request / Response schemas ─────────────────────────────────────────────────

class NormalizeRequest(BaseModel):
    """Request body for the /normalize endpoint."""
    raw_description: str
    product_id: int = 1


class NormalizeResponse(BaseModel):
    """Successful normalization result."""
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    retry_count: int = 0
    processing_time_ms: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    ollama_available: bool
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    """Bulk CSV upload response."""
    total: int
    success: int
    errors: int
    success_rate: float
    results: list[dict[str, Any]]
    error_records: list[dict[str, Any]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Liveness + readiness check.

    Returns 200 if the service is up. ``ollama_available`` reflects
    whether the configured model is reachable.
    """
    client = OllamaClient()
    available = client.is_available()
    log.info(f"Health check — ollama_available={available}")
    return HealthResponse(
        status="ok" if available else "degraded",
        model=settings.ollama_model,
        ollama_available=available,
    )


@app.post("/normalize", response_model=NormalizeResponse, tags=["Normalization"])
async def normalize_single(request: NormalizeRequest) -> NormalizeResponse:
    """
    Normalize a single raw product description.

    Cleans the input text, sends it to the local LLM, validates and
    post-processes the response, then returns the structured product data.

    - **raw_description**: Messy product text scraped from an e-commerce site.
    - **product_id**: Optional identifier for logging purposes.
    """
    if not request.raw_description.strip():
        raise HTTPException(status_code=422, detail="raw_description must not be empty")

    start = time.perf_counter()
    log.info(f"[/normalize] ID={request.product_id} — processing…")

    # Clean the input
    cleaned = clean_description(request.raw_description)
    raw = RawProduct(id=request.product_id, raw_description=cleaned)

    # LLM parse
    parser = get_parser()
    llm_data, retry_count, llm_raw = parser.parse(cleaned, product_id=request.product_id)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if llm_data is None:
        log.warning(f"[/normalize] ID={request.product_id} — LLM parse failed")
        return NormalizeResponse(
            success=False,
            error="LLM failed to return valid JSON after all retries",
            retry_count=retry_count,
            processing_time_ms=round(elapsed_ms, 2),
        )

    # Validate
    result = validate_product(raw, llm_data, retry_count, llm_raw)

    if isinstance(result, NormalizedProduct):
        normed = postprocess_product(result)
        log.info(f"[/normalize] ID={request.product_id} — success")
        return NormalizeResponse(
            success=True,
            data=normed.model_dump(),
            retry_count=retry_count,
            processing_time_ms=round(elapsed_ms, 2),
        )
    else:
        log.warning(f"[/normalize] ID={request.product_id} — validation failed")
        return NormalizeResponse(
            success=False,
            error=result.error_detail,
            retry_count=retry_count,
            processing_time_ms=round(elapsed_ms, 2),
        )


@app.post("/upload", response_model=UploadResponse, tags=["Normalization"])
async def upload_file(
    file: UploadFile = File(...),
    columns: str | None = None,
) -> UploadResponse:
    """
    Upload a product data file and normalize all rows.

    Accepts **CSV, JSON (array of objects), JSONL, or Excel (.xlsx/.xls)**.

    - If the file has a single `raw_description` column it is used directly.
    - Otherwise, all non-empty columns are merged into a labeled description
      (e.g. `Title: Apple iPhone 15\\nPrice: ₹79,999`) and sent to the LLM.
    - Use the optional **columns** query parameter to specify exactly which
      columns to merge, e.g. `?columns=title,price,availability`.

    Returns a JSON payload with:
    - **results**: successfully normalized records
    - **error_records**: records that failed
    - **success_rate**: percentage of successfully normalized rows
    """
    SUPPORTED = {".csv", ".json", ".jsonl", ".xlsx", ".xls"}
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(SUPPORTED))}",
        )

    log.info(f"[/upload] Received file: {filename} ({ext})")

    content = await file.read()
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
        elif ext == ".json":
            df = pd.read_json(io.BytesIO(content), orient="records").astype(str).fillna("")
        elif ext == ".jsonl":
            df = pd.read_json(io.BytesIO(content), lines=True).astype(str).fillna("")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content), dtype=str).fillna("")
        else:
            df = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="File has no usable data")

    # Resolve which columns to merge
    if columns:
        selected_cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if not selected_cols:
            raise HTTPException(
                status_code=400,
                detail=f"None of the requested columns exist. Available: {list(df.columns)}",
            )
    else:
        selected_cols = detect_description_columns(df)

    # Build raw_description
    df["raw_description"] = df.apply(
        lambda row: build_description_from_columns(row, selected_cols),
        axis=1,
    )

    df = preprocess_dataframe(df)
    df = df.head(100)  # Cap at 100 rows for API safety

    log.info(f"[/upload] Processing {len(df)} rows using columns: {selected_cols}…")

    parser = get_parser()
    successes: list[NormalizedProduct] = []
    failures: list[ErrorRecord] = []

    for _, row in df.iterrows():
        raw = RawProduct(id=int(row["id"]), raw_description=str(row["raw_description"]))
        llm_data, retry_count, llm_raw = parser.parse(raw.raw_description, raw.id)

        if llm_data is None:
            failures.append(
                ErrorRecord(
                    id=raw.id,
                    raw_description=raw.raw_description,
                    error_type="json_parse_error",
                    error_detail="LLM returned no valid JSON",
                    retry_count=retry_count,
                    llm_raw_output=llm_raw,
                )
            )
            continue

        result = validate_product(raw, llm_data, retry_count, llm_raw)
        if isinstance(result, NormalizedProduct):
            successes.append(postprocess_product(result))
        else:
            failures.append(result)

    total = len(successes) + len(failures)
    rate = round(len(successes) / total * 100, 2) if total else 0.0
    log.info(f"[/upload] Done — {len(successes)}/{total} succeeded")

    return UploadResponse(
        total=total,
        success=len(successes),
        errors=len(failures),
        success_rate=rate,
        results=[p.model_dump() for p in successes],
        error_records=[e.model_dump() for e in failures],
    )


@app.get("/download/clean", tags=["Export"])
async def download_clean_csv():
    """
    Stream the latest clean_products.csv back to the client.
    Raises 404 if the pipeline has not been run yet.
    """
    path = settings.output_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="clean_products.csv not found. Run the pipeline first.",
        )
    return StreamingResponse(
        open(path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=clean_products.csv"},
    )


@app.get("/download/errors", tags=["Export"])
async def download_error_csv():
    """Stream the latest errors.csv back to the client."""
    path = settings.error_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="errors.csv not found.")
    return StreamingResponse(
        open(path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=errors.csv"},
    )
