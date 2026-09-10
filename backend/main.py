"""FastAPI entry point for the TransNLP API.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is importable when running directly (no pip install -e)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.corpus import load_corpus
from ai.embed import (
    load_corpus_bm25_index,
    load_corpus_dense_embeddings,
    load_corpus_embeddings,
)
from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("spoken_ir.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load corpus retrieval artifacts once at startup. Log a clear error if any are missing."""
    logger.info("Initializing Spoken-IR backend...")
    try:
        corpus = load_corpus()
        dense_embs = load_corpus_dense_embeddings()
        bm25_idx = load_corpus_bm25_index()
        try:
            tfidf_embs = load_corpus_embeddings()
            tfidf_shape = tfidf_embs.shape
        except Exception:
            tfidf_shape = "N/A"
    except FileNotFoundError as e:
        logger.error(
            "Startup failed: %s. "
            "Run scripts/preprocess_data.py then scripts/build_corpus_embeddings.py first.",
            e,
        )
        raise

    logger.info(
        "Corpus preloaded successfully: %d transcripts. Dense shape=%s, BM25 docs=%d, TF-IDF shape=%s",
        len(corpus),
        dense_embs.shape,
        len(corpus),
        tfidf_shape,
    )
    yield
    logger.info("Shutting down.")



app = FastAPI(
    title="Spoken-IR API",
    description="Stand-Up Comedy Semantic Search & Hybrid Information Retrieval Engine.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — origins are configurable via SPOKEN_IR_CORS_ORIGINS or legacy TRANSNLP_CORS_ORIGINS.
# Defaults to localhost for local development; override in docker-compose / k8s.
_raw_origins = os.environ.get(
    "SPOKEN_IR_CORS_ORIGINS",
    os.environ.get("TRANSNLP_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"),
)
_cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
