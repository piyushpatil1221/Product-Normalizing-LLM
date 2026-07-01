"""
utils.py — Shared utility functions used across the pipeline.

Provides:
    - safe_json_parse     : extract JSON from noisy LLM output
    - extract_price_regex : regex-based fallback price extractor
    - fix_null_price      : patch a product record when LLM missed the price
    - sanitize_text       : light text cleanup
    - format_price        : format integer price for display
    - chunk_list          : split a list into fixed-size batches
    - timer               : context manager measuring elapsed time
    - write_json_lines    : append records to a JSONL file
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from typing import Any, Generator, Iterable, TypeVar

T = TypeVar("T")

# ── JSON extraction ───────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_BARE_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def safe_json_parse(text: str) -> dict[str, Any] | None:
    """
    Attempt to extract a JSON object from LLM output.

    Handles three common failure modes:
    1. LLM wraps JSON in a markdown code fence (```json … ```)
    2. LLM prepends/appends explanatory text around a bare ``{…}`` block
    3. LLM returns clean JSON directly

    Args:
        text: Raw string returned by the LLM.

    Returns:
        Parsed dict, or ``None`` if no valid JSON is found.
    """
    if not text:
        return None

    # Case 1 — markdown fences
    fence_match = _JSON_BLOCK_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Case 2 — bare JSON object embedded in text
    obj_match = _BARE_OBJECT_RE.search(text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # Case 3 — entire string is valid JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ── Regex price extractor (fallback for when LLM misses the price) ────────────

# All patterns that indicate a price in the text (ordered most-specific first)
_PRICE_PATTERNS = [
    # Sale price: Sale Rs/₹ 7996  —  explicit sale label FIRST (highest priority)
    re.compile(
        r"(?:sale|offer|discounted|now)[:\s]+(?:price)?[:\s]*"
        r"(?:Rs\.?|INR|₹)?\s*([\d][\d,]*)",
        re.IGNORECASE,
    ),
    # @ Rs.1,499  /  @ ₹1499
    re.compile(r"@\s*(?:Rs\.?|INR|₹)\s*([\d][\d,]*)", re.IGNORECASE),
    # ₹79,999  /  ₹79999
    re.compile(r"₹\s*([\d][\d,]+)"),
    # Rs.1,499  /  Rs 62990  /  Rs1499
    re.compile(r"Rs\.?\s*([\d][\d,]*)", re.IGNORECASE),
    # INR 24990
    re.compile(r"INR\s*([\d][\d,]*)", re.IGNORECASE),
    # Price: 99999  /  Selling Price: 79999
    re.compile(
        r"(?:selling\s+)?price[:\s]+[₹$€]?\s*([\d][\d,]*)",
        re.IGNORECASE,
    ),
    # MRP: ₹11,995  —  LOWEST priority
    re.compile(r"MRP[:\s]*[₹$€]?\s*([\d][\d,]*)", re.IGNORECASE),
]

# Amount patterns that should be EXCLUDED (discounts, not prices)
_DISCOUNT_PATTERNS = [
    # "Flat Rs5000 OFF", "Flat ₹5,000 OFF", "Extra ₹2000 discount"
    re.compile(
        r"(?:flat|upto|up to|extra|additional|get|save|cashback|off|avail)[\s₹$€,Rs.]*([\d][\d,]*)\s*(?:off|discount|cashback|back|rupees)",
        re.IGNORECASE,
    ),
    # "X OFF" where X is directly labelled as discount
    re.compile(
        r"([\d][\d,]*)\s*(?:off|discount)",
        re.IGNORECASE,
    ),
]


def extract_price_regex(text: str) -> int | None:
    """
    Fallback price extractor using regex patterns.

    Tries each price pattern in priority order. Discount amounts
    ("₹5000 off", "Flat ₹2000 cashback") are excluded.

    Args:
        text: Raw product description text.

    Returns:
        Integer price, or None if not found.
    """
    if not text:
        return None

    # Collect all discount amounts to exclude
    excluded: set[int] = set()
    for pat in _DISCOUNT_PATTERNS:
        for m in pat.finditer(text):
            try:
                excluded.add(int(m.group(1).replace(",", "")))
            except (ValueError, IndexError):
                pass

    # Try each price pattern
    for pat in _PRICE_PATTERNS:
        for m in pat.finditer(text):
            try:
                candidate = int(m.group(1).replace(",", ""))
                if candidate > 0 and candidate not in excluded:
                    return candidate
            except (ValueError, IndexError):
                pass

    return None


def fix_null_price(llm_data: dict[str, Any], raw_text: str) -> dict[str, Any]:
    """
    If the LLM returned price=null but the raw text contains a detectable
    price, patch the dict using the regex extractor.

    Args:
        llm_data: Parsed dict from the LLM.
        raw_text: Original raw description text.

    Returns:
        Possibly-patched llm_data dict (new copy if patched, same object otherwise).
    """
    if llm_data.get("price") is not None:
        return llm_data  # Already has a price — nothing to do

    found = extract_price_regex(raw_text)
    if found is not None:
        llm_data = dict(llm_data)  # shallow copy to avoid mutating caller's dict
        llm_data["price"] = found
    return llm_data


# ── Text helpers ──────────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """
    Light sanitisation: strip leading/trailing whitespace and collapse
    internal runs of whitespace to a single space.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def format_price(price: int | None, currency: str = "INR") -> str:
    """
    Format a numeric price for human-readable display.

    Examples:
        >>> format_price(99999, "INR")
        '₹99,999'
        >>> format_price(None)
        'N/A'
    """
    if price is None:
        return "N/A"
    symbol = {"INR": "₹", "USD": "$", "EUR": "€"}.get(currency, currency)
    return f"{symbol}{price:,}"


# ── Batch processing ──────────────────────────────────────────────────────────

def chunk_list(items: list[T], size: int) -> Generator[list[T], None, None]:
    """
    Yield successive fixed-size chunks from *items*.

    Args:
        items: Source list.
        size:  Maximum chunk length (last chunk may be smaller).

    Yields:
        Consecutive sublists of *items*.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Timing ────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """
    Context manager that prints elapsed wall-clock time.

    Usage::

        with timer("LLM inference"):
            result = llm.call(prompt)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] finished in {elapsed:.3f}s")


# ── JSONL writer ──────────────────────────────────────────────────────────────

def write_json_lines(path: str, records: Iterable[dict[str, Any]]) -> None:
    """
    Append *records* to a JSONL (newline-delimited JSON) file.

    Creates the file if it does not exist.

    Args:
        path:    Destination file path.
        records: Iterable of dicts to serialise.
    """
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
