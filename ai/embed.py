"""Embeddings and indices for the corpus and queries.

Supports:
1. Dense bi-encoder embeddings via sentence-transformers (all-MiniLM-L6-v2, 384-d)
2. BM25 lexical index via rank_bm25 (BM25Okapi)
3. Classical TF-IDF vectorizer (retained for backward compatibility)
"""

from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from ai import nltk_setup  # noqa: F401
from ai.corpus import load_corpus
from ai.nlp import identity_analyzer, identity_tokenizer, preprocess
from config import (
    AI_DATA_DIR,
    BM25_INDEX_PATH,
    DENSE_EMBEDDINGS_PATH,
    DENSE_MODEL_NAME,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def embeddings_path() -> str:
    AI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(AI_DATA_DIR / "corpus_embeddings.npy")


def dense_embeddings_path() -> str:
    AI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DENSE_EMBEDDINGS_PATH)


def bm25_index_path() -> str:
    AI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(BM25_INDEX_PATH)


# ── Dense Model & Embeddings (sentence-transformers) ───────────────────────────

@lru_cache(maxsize=1)
def load_dense_model():
    """Load the sentence-transformers bi-encoder model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading dense model: %s", DENSE_MODEL_NAME)
    return SentenceTransformer(DENSE_MODEL_NAME)


def embed_dense_query(text: str) -> np.ndarray:
    """Encode a single text query into a normalized 384-d dense vector."""
    if not isinstance(text, str) or not text.strip():
        return np.zeros((1, 384), dtype=np.float32)
    model = load_dense_model()
    # Normalize embeddings so dot-product equals cosine similarity
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32)


def build_corpus_dense_embeddings(corpus: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Build the (N, 384) dense embedding matrix using sentence-transformers."""
    df = corpus if corpus is not None else load_corpus()
    model = load_dense_model()

    # We use the raw Transcript when available, truncated to the first 4000 characters
    # (or preprocessed content) to give the bi-encoder high-quality semantic context.
    texts: list[str] = []
    for _, row in df.iterrows():
        raw = row.get("Transcript")
        prep = row.get("preprocessed_content")
        if isinstance(raw, str) and raw.strip():
            # First 2000 chars captures key themes without exceeding token budget
            texts.append(raw[:2000].strip())
        elif isinstance(prep, str) and prep.strip():
            texts.append(prep[:1500].strip())
        else:
            texts.append("")

    logger.info("Encoding %d corpus documents with %s...", len(texts), DENSE_MODEL_NAME)
    matrix = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dense = np.asarray(matrix, dtype=np.float32)
    np.save(dense_embeddings_path(), dense)
    logger.info("Saved dense corpus embeddings: %s shape=%s", dense_embeddings_path(), dense.shape)
    return dense


@lru_cache(maxsize=1)
def load_corpus_dense_embeddings() -> np.ndarray:
    """Load the persisted dense corpus embeddings matrix."""
    path = dense_embeddings_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dense corpus embeddings not found at {path}. "
            "Run scripts/build_corpus_embeddings.py first."
        )
    arr = np.load(path)
    logger.info("Loaded dense corpus embeddings: shape=%s", arr.shape)
    return arr


# ── BM25 Lexical Index ─────────────────────────────────────────────────────────

def build_corpus_bm25_index(corpus: Optional[pd.DataFrame] = None) -> BM25Okapi:
    """Build and serialize the BM25Okapi index over preprocessed corpus tokens."""
    df = corpus if corpus is not None else load_corpus()

    tokenized_corpus: list[list[str]] = []
    for text in df["preprocessed_content"].tolist():
        if isinstance(text, str) and text.strip():
            tokenized_corpus.append(text.split())
        else:
            tokenized_corpus.append([])

    bm25 = BM25Okapi(tokenized_corpus)
    path = bm25_index_path()
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("Saved BM25 index to %s (documents=%d)", path, len(tokenized_corpus))
    return bm25


@lru_cache(maxsize=1)
def load_corpus_bm25_index() -> BM25Okapi:
    """Load the persisted BM25 index."""
    path = bm25_index_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BM25 index not found at {path}. "
            "Run scripts/build_corpus_embeddings.py first."
        )
    with open(path, "rb") as f:
        bm25 = pickle.load(f)
    logger.info("Loaded BM25 index from %s", path)
    return bm25


# ── Classical TF-IDF (Backward compatibility) ─────────────────────────────────

def _ensure_legacy_pickle_stubs():
    """Ensure modules referenced by legacy pickle files are importable in any environment."""
    import sys
    import types
    from ai.nlp import identity_tokenizer, identity_analyzer

    if "src.utils.helpers" not in sys.modules:
        if "src" not in sys.modules:
            sys.modules["src"] = types.ModuleType("src")
        if "src.utils" not in sys.modules:
            sys.modules["src.utils"] = types.ModuleType("src.utils")
            sys.modules["src"].utils = sys.modules["src.utils"]
        stub = types.ModuleType("src.utils.helpers")
        stub.identity_tokenizer = identity_tokenizer
        stub.identity_analyzer = identity_analyzer
        sys.modules["src.utils.helpers"] = stub
        sys.modules["src.utils"].helpers = stub


@lru_cache(maxsize=1)
def load_vectorizer():
    _ensure_legacy_pickle_stubs()
    path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_query(text: str) -> np.ndarray:
    """Vectorize a single preprocessed string with TF-IDF. Returns shape (1, V) dense."""
    vectorizer = load_vectorizer()
    processed = preprocess(text)
    tokens = processed.split() if processed else []
    if not tokens:
        return np.zeros((1, len(vectorizer.vocabulary_)), dtype=np.float32)
    vec = vectorizer.transform([tokens])
    return vec.toarray()


def build_corpus_embeddings(corpus: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Build the (N, V) dense TF-IDF matrix for the corpus and persist it."""
    df = corpus if corpus is not None else load_corpus()
    vectorizer = load_vectorizer()

    tokenized: list[list[str]] = []
    for text in df["preprocessed_content"].tolist():
        if not isinstance(text, str) or not text.strip():
            tokenized.append([])
            continue
        tokenized.append(text.split())

    matrix = vectorizer.transform(tokenized)
    dense = np.asarray(matrix.toarray(), dtype=np.float32)
    
    # Pre-normalize for faster cosine similarity at runtime
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense_normalized = dense / norms
    
    np.save(embeddings_path(), dense_normalized)
    logger.info("Saved corpus TF-IDF embeddings: %s shape=%s", embeddings_path(), dense_normalized.shape)
    return dense_normalized


@lru_cache(maxsize=1)
def load_corpus_embeddings() -> np.ndarray:
    """Load the persisted TF-IDF corpus embeddings matrix."""
    path = embeddings_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus embeddings not found at {path}. "
            "Run scripts/build_corpus_embeddings.py first."
        )
    arr = np.load(path)
    logger.info("Loaded corpus embeddings: shape=%s", arr.shape)
    return arr