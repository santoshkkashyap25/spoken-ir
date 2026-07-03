"""Preprocess scraped transcripts into the corpus CSV.

Reads data/raw/scraped_and_cleaned_content_data.csv, runs the NLP
pipeline on each Transcript, and writes data/processed/processed_content_data.csv.
The corpus CSV is the single source of truth for both the AI layer
and the API.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

# Allow running directly: `python scripts/preprocess_data.py`.
sys.path.insert(0, ".")

import pandas as pd

from ai.nlp import preprocess
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocess_data")


@lru_cache(maxsize=1)
def _pipeline():
    """Touch the AI layer once so NLTK + spaCy are warmed before the loop."""
    return preprocess("warmup")


def preprocess_data() -> pd.DataFrame | None:
    """Run the full preprocessing pipeline. Idempotent given identical input."""
    scraped_csv = os.path.join(RAW_DATA_DIR, "scraped_and_cleaned_content_data.csv")
    if not os.path.exists(scraped_csv):
        logger.error("Scraped data not found at %s. Run scrape_data.py first.", scraped_csv)
        return None

    df = pd.read_csv(scraped_csv)
    if "Transcript" not in df.columns or df["Transcript"].empty:
        logger.error("No 'Transcript' column in scraped data.")
        return None

    # Warm up the pipeline (downloads NLTK data, loads spaCy) before the loop.
    _pipeline()

    logger.info("Applying NLP pipeline to %d transcripts...", len(df))
    df["preprocessed_content"] = df["Transcript"].apply(
        lambda x: preprocess(x) if isinstance(x, str) and x.strip() else ""
    )

    # Coerce rating to numeric; NaN if missing.
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Drop rows with no preprocessed content (corpus needs vectorizable text).
    before = len(df)
    df = df[df["preprocessed_content"].str.strip() != ""]
    if len(df) < before:
        logger.info("Dropped %d rows with empty preprocessed content", before - len(df))

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "processed_content_data.csv")
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d records to %s", len(df), out_path)
    return df


if __name__ == "__main__":
    preprocess_data()
