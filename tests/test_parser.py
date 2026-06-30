"""
tests/test_parser.py — Unit tests for the LLM parser and utilities.

Tests cover:
    - safe_json_parse: markdown fences, bare objects, clean JSON, garbage
    - PromptBuilder: template loading, variable injection
    - LLMParser: retry logic, success path, failure path (mocked LLM)
    - Preprocessing functions
    - Postprocessing normalization maps
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_parser import LLMParser, OllamaClient, PromptBuilder
from src.postprocess import (
    normalize_availability,
    normalize_brand,
    normalize_category,
    postprocess_product,
)
from src.preprocess import (
    clean_description,
    fix_encoding,
    normalize_unicode,
    normalize_whitespace,
    remove_html,
)
from src.schemas import NormalizedProduct
from src.utils import chunk_list, format_price, safe_json_parse


# ═══════════════════════════════════════════════════════════════════════════════
# safe_json_parse
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeJsonParse:
    def test_clean_json(self):
        """Should parse a clean JSON string directly."""
        result = safe_json_parse('{"brand": "Apple", "price": 79999}')
        assert result == {"brand": "Apple", "price": 79999}

    def test_json_in_markdown_fence(self):
        """Should extract JSON from triple-backtick fence."""
        text = '```json\n{"brand": "Samsung", "price": 49999}\n```'
        result = safe_json_parse(text)
        assert result["brand"] == "Samsung"

    def test_json_in_unmarked_fence(self):
        """Should extract JSON from fence without language tag."""
        text = '```\n{"brand": "OnePlus"}\n```'
        result = safe_json_parse(text)
        assert result["brand"] == "OnePlus"

    def test_json_embedded_in_text(self):
        """Should find JSON object embedded in surrounding text."""
        text = 'Here is the result: {"product_name": "Galaxy S25"} — done.'
        result = safe_json_parse(text)
        assert result["product_name"] == "Galaxy S25"

    def test_garbage_returns_none(self):
        """Complete garbage should return None."""
        assert safe_json_parse("Sorry, I cannot help with that.") is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert safe_json_parse("") is None

    def test_none_input_returns_none(self):
        """None input should return None."""
        assert safe_json_parse(None) is None  # type: ignore[arg-type]

    def test_partial_json_returns_none(self):
        """Truncated JSON should return None."""
        assert safe_json_parse('{"brand": "Apple", "price":') is None

    def test_nested_json(self):
        """Should handle nested JSON structures."""
        text = '{"product": {"name": "Laptop"}, "price": 55000}'
        result = safe_json_parse(text)
        assert result["price"] == 55000


# ═══════════════════════════════════════════════════════════════════════════════
# PromptBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:
    def test_build_injects_description(self, tmp_path):
        """PromptBuilder.build should replace {raw_description} placeholder."""
        template_file = tmp_path / "prompt.txt"
        template_file.write_text("Parse this: {raw_description}", encoding="utf-8")

        builder = PromptBuilder(template_path=template_file)
        result = builder.build("iPhone 15 128GB ₹79999")
        assert "iPhone 15 128GB" in result
        assert "{raw_description}" not in result

    def test_missing_template_raises(self, tmp_path):
        """FileNotFoundError should be raised for non-existent templates."""
        builder = PromptBuilder(template_path=tmp_path / "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            _ = builder.template

    def test_reload_updates_template(self, tmp_path):
        """reload() should pick up changes written after initial load."""
        template_file = tmp_path / "prompt.txt"
        template_file.write_text("Version 1: {raw_description}", encoding="utf-8")
        builder = PromptBuilder(template_path=template_file)
        _ = builder.template  # loads v1

        template_file.write_text("Version 2: {raw_description}", encoding="utf-8")
        builder.reload()
        assert "Version 2" in builder.template


# ═══════════════════════════════════════════════════════════════════════════════
# LLMParser (mocked OllamaClient)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMParser:
    def _make_parser(self, responses: list[str]) -> LLMParser:
        """Build a LLMParser with a mocked client returning given responses."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.chat.side_effect = responses

        # Use the real prompt template if available, else a stub
        try:
            builder = PromptBuilder()
        except Exception:
            builder = MagicMock(spec=PromptBuilder)
            builder.build.return_value = "stub prompt"

        return LLMParser(client=mock_client, builder=builder)

    def test_success_on_first_attempt(self):
        """Parser should return parsed dict and retry_count=0 on first success."""
        json_response = '{"product_name":"iPhone 15","brand":"Apple","category":"Mobile","price":79999,"currency":"INR","offer":null,"availability":"In Stock","delivery":"Ships Tomorrow","seller":null,"confidence":0.95}'
        parser = self._make_parser([json_response])
        result, retries, _ = parser.parse("iPhone 15 ₹79999", product_id=1)
        assert result is not None
        assert result["brand"] == "Apple"
        assert retries == 0

    def test_retry_on_bad_json_then_success(self):
        """Parser should retry after bad JSON and succeed on second attempt."""
        bad = "Sorry, I cannot parse this."
        good = '{"product_name":"Galaxy S25","brand":"Samsung","category":"Mobile","price":49999,"currency":"INR","offer":null,"availability":"In Stock","delivery":null,"seller":null,"confidence":0.88}'
        parser = self._make_parser([bad, good])
        result, retries, _ = parser.parse("Samsung Galaxy S25", product_id=2)
        assert result is not None
        assert result["brand"] == "Samsung"
        assert retries == 1

    def test_all_retries_fail_returns_none(self):
        """If all retries return bad JSON, parse() should return (None, max_retries, ...)."""
        from src.config import settings

        bad_responses = ["not json"] * settings.llm_retry_limit
        parser = self._make_parser(bad_responses)
        result, retries, last_raw = parser.parse("garbage input", product_id=99)
        assert result is None
        assert retries == settings.llm_retry_limit
        assert last_raw == "not json"

    def test_exception_during_llm_call_retries(self):
        """Network/timeout exceptions should be caught and trigger retry."""
        good = '{"product_name":"OnePlus 12","brand":"OnePlus","category":"Mobile","price":64999,"currency":"INR","offer":null,"availability":"In Stock","delivery":null,"seller":null,"confidence":0.9}'
        parser = self._make_parser([Exception("Connection refused"), good])
        result, retries, _ = parser.parse("OnePlus 12", product_id=3)
        assert result is not None
        assert retries == 1

    def test_json_with_markdown_fence_is_parsed(self):
        """Parser should handle LLM output wrapped in markdown code fence."""
        fenced = '```json\n{"product_name":"Pixel 8","brand":"Google","category":"Mobile","price":59999,"currency":"INR","offer":null,"availability":"In Stock","delivery":null,"seller":null,"confidence":0.85}\n```'
        parser = self._make_parser([fenced])
        result, retries, _ = parser.parse("Google Pixel 8", product_id=4)
        assert result is not None
        assert result["brand"] == "Google"


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_remove_html_tags(self):
        result = remove_html("<b>Apple</b> <i>iPhone</i> &amp; more")
        assert "<b>" not in result
        assert "&amp;" not in result
        assert "Apple" in result

    def test_normalize_whitespace_collapses_spaces(self):
        result = normalize_whitespace("Apple  iPhone   15   128GB")
        assert "  " not in result
        assert result == "Apple iPhone 15 128GB"

    def test_normalize_whitespace_strips_edges(self):
        result = normalize_whitespace("  Apple iPhone 15  ")
        assert result == "Apple iPhone 15"

    def test_clean_description_full_pipeline(self):
        messy = "  <b>APPLE</b>\n\niPhone&nbsp;15\t\t128GB\n\n\n"
        result = clean_description(messy)
        assert "<b>" not in result
        assert "APPLE" in result or "Apple" in result
        assert "\t" not in result

    def test_clean_description_empty_string(self):
        result = clean_description("")
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Postprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostprocessing:
    def test_normalize_brand_apple(self):
        assert normalize_brand("apple") == "Apple"
        assert normalize_brand("APPLE") == "Apple"

    def test_normalize_brand_samsung(self):
        assert normalize_brand("samsung") == "Samsung"

    def test_normalize_brand_unknown_titlecases(self):
        result = normalize_brand("somenewbrand")
        assert result == "Somenewbrand"

    def test_normalize_category_mobile_variants(self):
        assert normalize_category("smartphone") == "Mobile"
        assert normalize_category("Cellphone") == "Mobile"
        assert normalize_category("phone") == "Mobile"

    def test_normalize_category_headphones_variants(self):
        assert normalize_category("earbuds") == "Headphones"
        assert normalize_category("TWS") == "Headphones"
        assert normalize_category("Headset") == "Headphones"

    def test_normalize_availability_in_stock(self):
        assert normalize_availability("available") == "In Stock"
        assert normalize_availability("YES") == "In Stock"

    def test_normalize_availability_out_of_stock(self):
        assert normalize_availability("sold out") == "Out of Stock"

    def test_normalize_availability_limited_pattern(self):
        assert normalize_availability("only 5 left") == "Limited Stock"

    def test_normalize_availability_coming_soon(self):
        assert normalize_availability("pre-order") == "Coming Soon"


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtils:
    def test_chunk_list_even(self):
        chunks = list(chunk_list([1, 2, 3, 4, 5, 6], 2))
        assert chunks == [[1, 2], [3, 4], [5, 6]]

    def test_chunk_list_odd(self):
        chunks = list(chunk_list([1, 2, 3, 4, 5], 2))
        assert len(chunks) == 3
        assert chunks[-1] == [5]

    def test_chunk_list_empty(self):
        assert list(chunk_list([], 3)) == []

    def test_format_price_inr(self):
        assert format_price(99999, "INR") == "₹99,999"

    def test_format_price_usd(self):
        assert format_price(1000, "USD") == "$1,000"

    def test_format_price_none(self):
        assert format_price(None) == "N/A"
