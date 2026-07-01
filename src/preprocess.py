"""
preprocess.py — Text preprocessing module for the normalization pipeline.

Responsibilities:
    - Load any web-scraped input file (CSV, JSON, JSONL, Excel)
    - Merge selected columns into a labeled description for the LLM
    - Remove HTML entities, control characters, and encoding artifacts
    - Normalize unicode, punctuation, and whitespace
    - Drop exact duplicates and empty descriptions
    - Return a clean list of RawProduct objects ready for the LLM

All operations are pure functions (no side effects) to make them
easy to test in isolation.
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.logger import get_logger
from src.schemas import RawProduct

log = get_logger(__name__)

# ── Compiled regex patterns (compile once, reuse many times) ──────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BULLET_RE = re.compile(r"[•·▪▸►◦‣⁃]")


# ── Individual cleaning steps ─────────────────────────────────────────────────

def remove_html(text: str) -> str:
    """Unescape HTML entities and strip all HTML tags."""
    text = html.unescape(text)
    return _HTML_TAG_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """
    Normalise to NFC (canonical decomposition, then canonical composition).
    Converts lookalike characters (e.g., curly quotes) to ASCII equivalents.
    """
    return unicodedata.normalize("NFC", text)


def fix_encoding(text: str) -> str:
    """
    Attempt to fix common mojibake patterns produced by latin-1 / utf-8
    double-encoding (e.g., â‚¹ → ₹).
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def remove_control_chars(text: str) -> str:
    """Strip invisible control characters that confuse the LLM."""
    return _CONTROL_CHAR_RE.sub("", text)


def normalize_bullets(text: str) -> str:
    """Replace typographic bullet symbols with a plain newline."""
    return _BULLET_RE.sub("\n", text)


def normalize_punctuation(text: str) -> str:
    """
    Standardize punctuation:
        - Curly quotes → straight quotes
        - Em/en dashes → hyphens
        - Ellipsis character → '...'
    """
    mapping = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2014": "-",
            "\u2013": "-",
            "\u2026": "...",
        }
    )
    return text.translate(mapping)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs; limit consecutive newlines to two."""
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_description(text: str) -> str:
    """
    Apply all cleaning steps in sequence.

    Pipeline order matters:
    1. fix_encoding   — must run before unicode normalisation
    2. remove_html    — strips tags/entities
    3. normalize_unicode
    4. remove_control_chars
    5. normalize_bullets
    6. normalize_punctuation
    7. normalize_whitespace  — must be last
    """
    text = fix_encoding(text)
    text = remove_html(text)
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_bullets(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ── DataFrame-level operations ────────────────────────────────────────────────

# Supported file extensions
_SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls"}


def load_input_file(path: Path | str) -> pd.DataFrame:
    """
    Load a web-scraped data file into a DataFrame.

    Supports CSV, JSON (array of objects), JSONL (newline-delimited JSON),
    and Excel (.xlsx / .xls) formats. All values are coerced to strings.

    Args:
        path: Absolute path to the input file.

    Returns:
        Raw DataFrame with all original columns preserved.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8", dtype=str)
    elif ext == ".json":
        df = pd.read_json(path, orient="records", dtype=str)
        df = df.astype(str)
    elif ext == ".jsonl":
        df = pd.read_json(path, lines=True, dtype=str)
        df = df.astype(str)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, encoding="utf-8", dtype=str)

    # Replace literal 'nan' strings (from dtype=str conversion) with empty string
    df = df.fillna("")

    log.info(f"Loaded {len(df)} rows from '{path.name}' ({ext} format)")
    return df


def detect_description_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect which columns are likely to carry useful product information.

    Heuristic: prefer columns whose names suggest product text, or fall back
    to all non-numeric-looking columns. A column named 'raw_description' is
    always returned alone (backward compatibility).

    Args:
        df: DataFrame from any supported input format.

    Returns:
        Ordered list of column names to use as description source.
    """
    cols = list(df.columns)

    # Exact backward-compat shortcut
    if "raw_description" in cols:
        return ["raw_description"]

    # Prefer columns with description-like names first
    PRIORITY_KEYWORDS = [
        "description", "title", "name", "product", "item",
        "brand", "price", "offer", "discount", "availability",
        "stock", "delivery", "seller", "category", "badge",
        "detail", "spec", "feature",
    ]

    priority = [
        c for c in cols
        if any(kw in c.lower() for kw in PRIORITY_KEYWORDS)
    ]
    rest = [c for c in cols if c not in priority]

    ordered = priority + rest

    # Filter out columns that are entirely empty
    non_empty = [c for c in ordered if df[c].str.strip().any()]

    return non_empty if non_empty else ordered


def build_description_from_columns(
    row: "pd.Series",
    columns: List[str],
    labeled: bool = True,
) -> str:
    """
    Build a single LLM-ready description string from multiple DataFrame columns.

    When *labeled* is True (recommended), each field is prefixed with its
    column name so the LLM understands the semantic meaning of each value::

        Title: Apple iPhone 15 128GB
        Price: ₹79,999
        Availability: Only 2 Left

    Args:
        row:     A single DataFrame row (pd.Series).
        columns: List of column names to include.
        labeled: If True, prefix each value with its column name.

    Returns:
        A multi-line string ready to be injected into the LLM prompt.
    """
    parts = []
    for col in columns:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            if labeled:
                label = col.replace("_", " ").title()
                parts.append(f"{label}: {val}")
            else:
                parts.append(val)
    return "\n".join(parts)


def load_raw_csv(path: Path | str) -> pd.DataFrame:
    """
    Legacy loader — kept for backward compatibility.

    Loads a CSV and renames the first column that contains 'description'
    (case-insensitive) to 'raw_description'. Falls back to column 1.

    New code should use :func:`load_input_file` instead.
    """
    df = load_input_file(path)

    # Identify the description column (backward-compat behaviour)
    desc_col = next(
        (c for c in df.columns if "description" in c.lower()),
        df.columns[0],
    )
    if desc_col != "raw_description":
        df = df.rename(columns={desc_col: "raw_description"})

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all cleaning operations on the DataFrame in place.

    Steps:
        1. Drop rows with missing/empty descriptions
        2. Apply clean_description to every row
        3. Drop exact duplicates (by cleaned text)
        4. Reset index and assign sequential IDs

    Args:
        df: Raw DataFrame with 'raw_description' column.

    Returns:
        Cleaned DataFrame ready for LLM processing.
    """
    original_count = len(df)

    # Step 1 — drop missing
    df = df.dropna(subset=["raw_description"])
    df = df[df["raw_description"].str.strip().astype(bool)]

    # Step 2 — clean each description
    df["raw_description"] = df["raw_description"].apply(clean_description)

    # Step 3 — drop duplicates
    df = df.drop_duplicates(subset=["raw_description"])

    # Step 4 — reset index + assign ID
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    removed = original_count - len(df)
    log.info(
        f"Preprocessing complete: {len(df)} records remain "
        f"({removed} removed as duplicates/empty)"
    )
    return df


def dataframe_to_raw_products(df: pd.DataFrame) -> list[RawProduct]:
    """
    Convert a preprocessed DataFrame into a list of :class:`RawProduct` objects.

    Args:
        df: Cleaned DataFrame with 'id' and 'raw_description' columns.

    Returns:
        List of validated :class:`RawProduct` instances.
    """
    products: list[RawProduct] = []
    for _, row in df.iterrows():
        products.append(
            RawProduct(
                id=int(row["id"]),
                raw_description=str(row["raw_description"]),
            )
        )
    log.debug(f"Converted {len(products)} rows to RawProduct objects")
    return products


def run_preprocessing(input_path: Path | str) -> list[RawProduct]:
    """
    Convenience wrapper: load CSV → preprocess → return RawProduct list.

    Args:
        input_path: Path to the raw CSV.

    Returns:
        List of :class:`RawProduct` objects ready for the LLM stage.
    """
    df = load_raw_csv(input_path)
    df = preprocess_dataframe(df)
    return dataframe_to_raw_products(df)
