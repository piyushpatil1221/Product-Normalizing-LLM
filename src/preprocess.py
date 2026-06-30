"""
preprocess.py — Text preprocessing module for the normalization pipeline.

Responsibilities:
    - Load the raw CSV from disk
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

def load_raw_csv(path: Path | str) -> pd.DataFrame:
    """
    Load the raw product CSV.

    Expects at least one column whose name contains 'description' (case-insensitive).
    Falls back to using the first column if no match is found.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        DataFrame with a guaranteed 'raw_description' column.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8", dtype=str)
    log.info(f"Loaded {len(df)} rows from {path.name}")

    # Identify the description column
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
