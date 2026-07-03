"""Corpus loading.

Single source of truth: data/processed/processed_content_data.csv.
The DataFrame is the in-process representation of the corpus.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)


def corpus_csv_path() -> str:
    """Resolve the processed-content CSV path relative to the project root."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return os.path.join(project_root, "data", "processed", "processed_content_data.csv")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known columns to the right dtypes; tolerate missing ones."""
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "preprocessed_content" not in df.columns:
        raise ValueError(
            "Corpus CSV is missing 'preprocessed_content'. "
            "Run scripts/preprocess_data.py first."
        )
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
    """Number of specials in the corpus."""
    return len(load_corpus())
