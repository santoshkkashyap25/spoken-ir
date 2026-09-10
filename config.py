"""Project-level configuration.

Single source of truth for paths and shared constants. Imported by both
the scripts/ (top-level) and the ai/ + backend/ packages (via sys.path).

All paths are derived from the location of this file (config.py), which
lives at the project root. This makes the project fully relocatable — it
works regardless of where on disk the repo is cloned, on any OS (Windows,
macOS, Linux, Docker).
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure fast local model loading without network latency/timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

APP_NAME = "Spoken-IR — Semantic Search & Information Retrieval Engine"

# ── Project root ───────────────────────────────────────────────────────────────
# config.py lives at the project root, so its parent IS the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ── Data directories ───────────────────────────────────────────────────────────
DATA_DIR: Path          = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path      = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path        = DATA_DIR / "models"
AI_DATA_DIR: Path       = DATA_DIR / "ai"

# ── Scraping ───────────────────────────────────────────────────────────────────
SCRAPING_BASE_URL: str    = "https://scrapsfromtheloft.com/stand-up-comedy-scripts/"
TRANSCRIPTS_RAW_DIR: Path = RAW_DATA_DIR / "transcripts"

# ── Information Retrieval & Semantic Search ────────────────────────────────────
DENSE_MODEL_NAME: str     = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_EMBEDDINGS_PATH: Path = AI_DATA_DIR / "corpus_dense_embeddings.npy"
BM25_INDEX_PATH: Path       = AI_DATA_DIR / "corpus_bm25.pkl"
DEFAULT_SEARCH_MODE: str    = "hybrid"
DEFAULT_HYBRID_ALPHA: float = 0.6  # 60% Dense Semantic, 40% Sparse BM25


