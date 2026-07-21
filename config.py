"""Project-level configuration.

Single source of truth for paths and shared constants. Imported by both
the scripts/ (top-level) and the ai/ + backend/ packages (via sys.path).

All paths are derived from the location of this file (config.py), which
lives at the project root. This makes the project fully relocatable — it
works regardless of where on disk the repo is cloned, on any OS (Windows,
macOS, Linux, Docker).
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "TransNLP — Stand-up Similarity"

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

