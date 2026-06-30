"""
tests/test_validator.py — Unit tests for the validation layer.

Tests cover:
    - Successful validation of well-formed LLM data
    - Pydantic validation errors (missing fields, wrong types)
    - Confidence threshold enforcement
    - Batch validation routing (successes vs errors)
    - Price coercion (string → int)
    - Currency normalization
"""

import pytest

from src.schemas import (
    ErrorRecord,
    NormalizedProduct,
    ProductSchema,
    RawProduct,
)
from src.validator import validate_batch, validate_product

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_raw() -> RawProduct:
    return RawProduct(id=1, raw_description="Apple iPhone 15 128GB ₹79999 In Stock")


@pytest.fixture
def valid_llm_data() -> dict:
    return {
        "product_name": "Apple iPhone 15 128GB",
        "brand": "Apple",
        "category": "Mobile",
        "price": 79999,
        "currency": "INR",
        "offer": "Flat Rs.5000 off with HDFC Card",
        "availability": "In Stock",
        "delivery": "Ships Tomorrow",
        "seller": "Appario Retail",
        "confidence": 0.95,
    }


# ── ProductSchema unit tests ───────────────────────────────────────────────────

class TestProductSchema:
    def test_valid_product_schema(self, valid_llm_data):
        """A fully valid dict should produce a ProductSchema without errors."""
        schema = ProductSchema(**valid_llm_data)
        assert schema.product_name == "Apple iPhone 15 128GB"
        assert schema.brand == "Apple"
        assert schema.price == 79999
        assert schema.currency == "INR"
        assert schema.confidence == 0.95

    def test_price_string_coercion(self):
        """Price given as a comma-formatted string should be coerced to int."""
        schema = ProductSchema(
            product_name="Test Product",
            brand="TestBrand",
            category="Mobile",
            price="79,999",
            currency="INR",
            availability="In Stock",
            confidence=0.8,
        )
        assert schema.price == 79999

    def test_price_with_rupee_symbol(self):
        """Price with ₹ prefix should be cleaned and coerced."""
        schema = ProductSchema(
            product_name="Test",
            brand="Brand",
            category="Mobile",
            price="₹ 12,500",
            currency="INR",
            availability="In Stock",
            confidence=0.7,
        )
        assert schema.price == 12500

    def test_price_none_is_allowed(self):
        """Price can be None if the LLM couldn't extract it."""
        schema = ProductSchema(
            product_name="Book Title",
            brand="Penguin",
            category="Book",
            price=None,
            currency="INR",
            availability="In Stock",
            confidence=0.6,
        )
        assert schema.price is None

    def test_currency_normalization_rs(self):
        """'rs' should normalize to 'INR'."""
        schema = ProductSchema(
            product_name="Test",
            brand="Brand",
            category="Mobile",
            currency="rs",
            availability="In Stock",
            confidence=0.8,
        )
        assert schema.currency == "INR"

    def test_currency_normalization_symbol(self):
        """Rupee symbol should normalize to 'INR'."""
        schema = ProductSchema(
            product_name="Test",
            brand="Brand",
            category="Mobile",
            currency="₹",
            availability="In Stock",
            confidence=0.8,
        )
        assert schema.currency == "INR"

    def test_confidence_clamped_above_one(self):
        """Confidence > 1.0 should be clamped to 1.0."""
        schema = ProductSchema(
            product_name="Test",
            brand="Brand",
            category="Mobile",
            confidence=1.5,
            availability="In Stock",
        )
        assert schema.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        """Confidence < 0 should be clamped to 0.0."""
        schema = ProductSchema(
            product_name="Test",
            brand="Brand",
            category="Mobile",
            confidence=-0.1,
            availability="In Stock",
        )
        assert schema.confidence == 0.0

    def test_missing_product_name_raises(self):
        """Missing product_name should raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProductSchema(
                product_name="",
                brand="Brand",
                category="Mobile",
                confidence=0.9,
                availability="In Stock",
            )

    def test_negative_price_rejected(self):
        """Negative integer price should raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProductSchema(
                product_name="Test",
                brand="Brand",
                category="Mobile",
                price=-500,
                availability="In Stock",
                confidence=0.8,
            )


# ── validate_product tests ─────────────────────────────────────────────────────

class TestValidateProduct:
    def test_valid_returns_normalized_product(self, valid_raw, valid_llm_data):
        """Valid data should return a NormalizedProduct."""
        result = validate_product(valid_raw, valid_llm_data)
        assert isinstance(result, NormalizedProduct)
        assert result.id == 1
        assert result.processing_status == "success"

    def test_invalid_data_returns_error_record(self, valid_raw):
        """Data missing required fields should return an ErrorRecord."""
        bad_data = {"brand": "Apple"}  # missing product_name and confidence
        result = validate_product(valid_raw, bad_data)
        assert isinstance(result, ErrorRecord)
        assert result.error_type == "validation_error"

    def test_low_confidence_returns_error(self, valid_raw, valid_llm_data):
        """Confidence below threshold should produce an ErrorRecord."""
        valid_llm_data["confidence"] = 0.01  # very low
        result = validate_product(valid_raw, valid_llm_data)
        assert isinstance(result, ErrorRecord)
        assert result.error_type == "low_confidence"

    def test_error_record_preserves_raw_description(self, valid_raw):
        """ErrorRecord must carry the original raw_description for debugging."""
        result = validate_product(valid_raw, {})
        assert isinstance(result, ErrorRecord)
        assert result.raw_description == valid_raw.raw_description

    def test_retry_count_is_recorded(self, valid_raw):
        """retry_count from the LLM parser should be preserved in ErrorRecord."""
        result = validate_product(valid_raw, {}, retry_count=2)
        assert isinstance(result, ErrorRecord)
        assert result.retry_count == 2


# ── validate_batch tests ───────────────────────────────────────────────────────

class TestValidateBatch:
    def test_batch_splits_correctly(self, valid_raw, valid_llm_data):
        """Batch with one good and one bad record should split 1:1."""
        raw_list = [
            valid_raw,
            RawProduct(id=2, raw_description="bad record"),
        ]
        llm_results = [
            (valid_llm_data, 0, None),
            (None, 3, "LLM timeout"),  # failed
        ]
        successes, failures = validate_batch(raw_list, llm_results)
        assert len(successes) == 1
        assert len(failures) == 1

    def test_none_llm_data_goes_to_errors(self, valid_raw):
        """None llm_data should route directly to errors as json_parse_error."""
        raw_list = [valid_raw]
        llm_results = [(None, 3, "garbage output")]
        successes, failures = validate_batch(raw_list, llm_results)
        assert len(successes) == 0
        assert len(failures) == 1
        assert failures[0].error_type == "json_parse_error"

    def test_all_valid_batch(self, valid_raw, valid_llm_data):
        """All valid records → all in successes, none in failures."""
        raw_list = [valid_raw, RawProduct(id=2, raw_description="product 2")]
        llm_results = [(valid_llm_data.copy(), 0, None), (valid_llm_data.copy(), 0, None)]
        # Update id for second product
        llm_results[1][0]["product_name"] = "Samsung Galaxy S25"
        successes, failures = validate_batch(raw_list, llm_results)
        assert len(successes) == 2
        assert len(failures) == 0
