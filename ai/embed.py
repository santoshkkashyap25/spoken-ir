"""Embeddings — TF-IDF vectors for the corpus and ad-hoc queries.

The corpus embedding matrix is a (N, V) sparse matrix where V is the TF-IDF
vocabulary (max 1000 features as configured when the vectorizer was trained).
It's persisted to data/ai/corpus_embeddings.npy as a dense array for fast
loading. Re-generate with scripts/build_corpus_embeddings.py after
preprocess_data.py.

Also precomputes a (N, num_topics) LDA topic-probability matrix and
persists it to data/ai/corpus_topic_vectors.npy.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import pickle

from ai import nltk_setup  # noqa: F401
from ai.corpus import load_corpus
from ai.nlp import identity_analyzer, identity_tokenizer, preprocess
from config import AI_DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


def embeddings_path() -> str:
    AI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(AI_DATA_DIR / "corpus_embeddings.npy")


def topic_vectors_path() -> str:
    AI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(AI_DATA_DIR / "corpus_topic_vectors.npy")


@lru_cache(maxsize=1)
def load_vectorizer():
    path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_query(text: str) -> np.ndarray:
    """Vectorize a single preprocessed string. Returns shape (1, V) dense."""
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
    logger.info("Saved corpus embeddings: %s shape=%s", embeddings_path(), dense_normalized.shape)
    return dense_normalized


@lru_cache(maxsize=1)
def load_corpus_embeddings() -> np.ndarray:
    """Load the persisted corpus embeddings matrix."""
    path = embeddings_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus embeddings not found at {path}. "
            "Run scripts/build_corpus_embeddings.py first."
        )
    arr = np.load(path)
    logger.info("Loaded corpus embeddings: shape=%s", arr.shape)
    return arr


def build_corpus_topic_vectors(corpus: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Build the (N, num_topics) LDA topic-probability matrix for the corpus."""
    from ai.topics import load_lda_dictionary, load_lda_model, TOPIC_LABELS

    df = corpus if corpus is not None else load_corpus()
    lda = load_lda_model()
    dictionary = load_lda_dictionary()

    rows: list[list[float]] = []
    for text in df["preprocessed_content"].tolist():
        tokens = text.split() if isinstance(text, str) and text.strip() else []
        if not tokens:
            rows.append([0.0] * len(TOPIC_LABELS))
            continue
        bow = dictionary.doc2bow(tokens)
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        vec = np.zeros(len(TOPIC_LABELS))
        for topic_id, p in dist:
            if topic_id < len(TOPIC_LABELS):
                vec[topic_id] = p
        rows.append(vec.tolist())

    arr = np.asarray(rows, dtype=np.float32)
    np.save(topic_vectors_path(), arr)
    logger.info("Saved corpus topic vectors: %s shape=%s", topic_vectors_path(), arr.shape)
    return arr


@lru_cache(maxsize=1)
def load_corpus_topic_vectors() -> np.ndarray:
    """Load the persisted corpus topic vector matrix."""
    path = topic_vectors_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus topic vectors not found at {path}. "
            "Run scripts/build_corpus_embeddings.py first."
        )
    return np.load(path)