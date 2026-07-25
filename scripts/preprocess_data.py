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
from pathlib import Path

# Ensure project root is importable when running directly (no pip install -e)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.nlp import preprocess, preprocess_batch
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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
    expected_cols = ["S No.", "URL", "Transcript", "Year", "Names", "Title"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.error("Scraped data missing required columns: %s", missing)
        return None

    # Warm up the pipeline (downloads NLTK data, loads spaCy) before the loop.
    _pipeline()

    logger.info("Applying NLP pipeline to %d transcripts...", len(df))
    texts = df["Transcript"].tolist()
    df["preprocessed_content"] = preprocess_batch(texts)



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
