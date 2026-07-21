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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.corpus import load_corpus
from ai.embed import load_corpus_embeddings, load_corpus_topic_vectors
from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("transnlp.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load corpus artifacts once at startup. Log a clear error if any are missing."""
    logger.info("Loading corpus and embeddings at startup...")
    try:
        corpus = load_corpus()
        embeddings = load_corpus_embeddings()
        topic_vecs = load_corpus_topic_vectors()
    except FileNotFoundError as e:
        logger.error(
            "Startup failed: %s. "
            "Run scripts/preprocess_data.py then scripts/build_corpus_embeddings.py first.",
            e,
        )
        raise
    logger.info(
        "Startup complete: corpus=%d rows, tfidf=%s, topics=%s",
        len(corpus),
        embeddings.shape,
        topic_vecs.shape,
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="TransNLP API",
    description="Find existing specials a draft stand-up transcript most resembles.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — origins are configurable via TRANSNLP_CORS_ORIGINS (comma-separated).
# Defaults to localhost for local development; override in docker-compose / k8s.
_raw_origins = os.environ.get(
    "TRANSNLP_CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
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
