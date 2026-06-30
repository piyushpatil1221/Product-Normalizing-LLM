"""
schemas.py — Pydantic models for the LLM Product Normalizer pipeline.

Defines:
    - ProductSchema    : validated output from the LLM
    - RawProduct       : raw input row before LLM processing
    - NormalizedProduct: final cleaned & enriched record written to CSV
    - ErrorRecord      : row that failed validation or LLM parsing
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class CurrencyEnum(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    UNKNOWN = "UNKNOWN"


class AvailabilityEnum(str, Enum):
    IN_STOCK = "In Stock"
    OUT_OF_STOCK = "Out of Stock"
    COMING_SOON = "Coming Soon"
    LIMITED = "Limited Stock"
    UNKNOWN = "Unknown"


class CategoryEnum(str, Enum):
    MOBILE = "Mobile"
    LAPTOP = "Laptop"
    HEADPHONES = "Headphones"
    TV = "TV"
    SHOES = "Shoes"
    FASHION = "Fashion"
    KITCHEN = "Kitchen Appliance"
    GAMING = "Gaming"
    BOOK = "Book"
    FURNITURE = "Furniture"
    UNKNOWN = "Unknown"


# ── LLM Output Schema ─────────────────────────────────────────────────────────

class ProductSchema(BaseModel):
    """
    Schema that the LLM is expected to return as a JSON object.
    Strict validation ensures data quality before the record reaches the CSV.
    """

    product_name: str = Field(..., min_length=2, description="Clean product name without offers/noise")
    brand: str = Field(..., min_length=1, description="Brand / manufacturer name")
    category: str = Field(..., description="Product category")
    price: Optional[int] = Field(None, ge=0, description="Numeric price (integer, INR)")
    currency: str = Field(default="INR", description="ISO-4217 currency code")
    offer: Optional[str] = Field(None, description="Active offer text, if any")
    availability: str = Field(default="Unknown", description="Stock availability string")
    delivery: Optional[str] = Field(None, description="Delivery timeline, if mentioned")
    seller: Optional[str] = Field(None, description="Seller name, if mentioned")
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM confidence score 0–1")

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("price", mode="before")
    @classmethod
    def coerce_price(cls, v):
        """Accept string prices like '99,999' or '₹ 12000' and convert to int."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            # Strip currency symbols and commas
            cleaned = v.replace("₹", "").replace("Rs", "").replace("INR", "")
            cleaned = cleaned.replace(",", "").strip()
            try:
                return int(float(cleaned))
            except ValueError:
                return None
        return None

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v):
        """Normalize currency strings to ISO-4217."""
        mapping = {
            "inr": "INR",
            "₹": "INR",
            "rs": "INR",
            "rupee": "INR",
            "rupees": "INR",
            "usd": "USD",
            "$": "USD",
            "eur": "EUR",
            "€": "EUR",
        }
        return mapping.get(str(v).lower().strip(), "INR")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        """Clamp confidence to [0, 1] regardless of LLM output."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("product_name", "brand", "category", mode="before")
    @classmethod
    def strip_strings(cls, v):
        return str(v).strip() if v else v

    @model_validator(mode="after")
    def check_required_name_and_brand(self) -> "ProductSchema":
        if not self.product_name:
            raise ValueError("product_name must not be empty")
        if not self.brand:
            raise ValueError("brand must not be empty")
        return self


# ── Pipeline Data Models ──────────────────────────────────────────────────────

class RawProduct(BaseModel):
    """Represents a single raw row from the input CSV."""

    id: int
    raw_description: str


class NormalizedProduct(ProductSchema):
    """ProductSchema enriched with pipeline metadata for the output CSV."""

    id: int
    raw_description: str
    processing_status: str = "success"


class ErrorRecord(BaseModel):
    """Row that failed LLM parsing or Pydantic validation."""

    id: int
    raw_description: str
    error_type: str       # e.g. "json_parse_error", "validation_error", "llm_timeout"
    error_detail: str
    retry_count: int = 0
    llm_raw_output: Optional[str] = None
