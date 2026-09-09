"""Corpus loading.

Single source of truth: data/processed/processed_content_data.csv.
The DataFrame is the in-process representation of the corpus.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd

from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def corpus_csv_path() -> str:
    """Return the absolute path to the processed-content CSV."""
    return str(PROCESSED_DATA_DIR / "processed_content_data.csv")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known columns to the right dtypes and ensure required schema."""
    expected_cols = [
        "S No.", "URL", "Transcript", "Year", "Names", "Title", "preprocessed_content"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Corpus CSV is missing required columns: {missing}. Run scripts/preprocess_data.py first.")

    if "rating" not in df.columns:
        df["rating"] = pd.NA

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["preprocessed_content"] = df["preprocessed_content"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def load_corpus() -> pd.DataFrame:
    """Load the corpus CSV. Cached in-process for the lifetime of the interpreter."""
    path = corpus_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus CSV not found at {path}. Run scripts/preprocess_data.py first."
        )
    logger.info("Loading corpus from %s", path)
    df = pd.read_csv(path)
    return _normalize_columns(df)


def get_corpus_size() -> int:
    """Number of documents in the corpus."""
    return len(load_corpus())
