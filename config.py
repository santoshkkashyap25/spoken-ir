"""Project-level configuration.

Single source of truth for paths and shared constants. Imported by both
the scripts/ (top-level) and the ai/ + backend/ packages (via sys.path).
"""

from __future__ import annotations

import os

APP_NAME = "TransNLP — Stand-up Similarity"

# Data and model paths.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DATA_DIR = os.path.join(_DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(_DATA_DIR, "processed")
MODELS_DIR = os.path.join(_DATA_DIR, "models")

# Scraping.
SCRAPING_BASE_URL = "https://scrapsfromtheloft.com/stand-up-comedy-scripts/"
TRANSCRIPTS_RAW_DIR = os.path.join(RAW_DATA_DIR, "transcripts")
