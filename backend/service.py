"""Service layer: orchestration over the `ai/` retrieval and topic layers.

Translates between AI primitives and Pydantic response models.
Supports:
- Dense semantic search (MiniLM bi-encoder)
- Sparse lexical search (BM25)
- Hybrid search (convex fusion of Dense + BM25)
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from ai import similarity
from ai.corpus import load_corpus
from ai.embed import (
    embed_dense_query,
    embed_query,
    load_corpus_bm25_index,
    load_corpus_dense_embeddings,
    load_corpus_embeddings,
)
from ai.nlp import preprocess
from ai.similarity import Neighbor, match_strength_label

from backend.schemas import MatchItem

logger = logging.getLogger(__name__)


def _safe_float(x) -> Optional[float]:
    """Convert a possibly-NaN value to a JSON-friendly float, or None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _safe_str(x) -> Optional[str]:
    """Convert a value to a JSON-friendly string, or None for NaN/missing."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None


def get_health() -> dict:
    """Build the /health response body."""
    return {
        "status": "ok",
        "corpus_size": len(load_corpus()),
    }


@lru_cache(maxsize=128)
def get_match(
    text: str,
    k: int = 5,
    search_mode: str = "hybrid",
    alpha: float = 0.6,
) -> dict:
    """Perform hybrid, dense, or sparse lexical retrieval across the transcript corpus."""
    corpus = load_corpus()

    search_mode = search_mode.lower().strip()
    if search_mode not in ("hybrid", "dense", "sparse"):
        search_mode = "hybrid"

    dense_corpus: Optional[np.ndarray] = None
    query_dense_vec: Optional[np.ndarray] = None
    dense_scores: Optional[np.ndarray] = None

    # Compute dense scores if needed
    if search_mode in ("hybrid", "dense"):
        try:
            dense_corpus = load_corpus_dense_embeddings()
            query_dense_vec = embed_dense_query(text)
            q_norm = float(np.linalg.norm(query_dense_vec))
            if q_norm > 0:
                q = query_dense_vec / q_norm
                dense_scores = (dense_corpus @ q.T).ravel()
            else:
                dense_scores = np.zeros(len(dense_corpus), dtype=np.float32)
        except Exception as e:
            logger.warning("Dense retrieval unavailable, falling back to TF-IDF: %s", e)
            dense_scores = None

    # Compute sparse BM25 scores if needed
    sparse_scores: Optional[np.ndarray] = None
    if search_mode in ("hybrid", "sparse"):
        try:
            bm25_index = load_corpus_bm25_index()
            processed_text = preprocess(text)
            tokens = processed_text.split() if processed_text else []
            if tokens:
                raw_bm25 = np.asarray(bm25_index.get_scores(tokens), dtype=np.float32)
                sparse_scores = similarity.normalize_scores(raw_bm25)
            else:
                sparse_scores = np.zeros(len(corpus), dtype=np.float32)
        except Exception as e:
            logger.warning("BM25 index unavailable, falling back to TF-IDF: %s", e)
            sparse_scores = None

    # Select retrieval method based on mode and availability
    neighbors: list[Neighbor] = []
    confidence_score: float = 0.0

    if search_mode == "hybrid" and dense_scores is not None and sparse_scores is not None:
        neighbors = similarity.hybrid_top_k(dense_scores, sparse_scores, alpha=alpha, k=k)
        confidence_score = float(np.mean([n.score for n in neighbors])) if neighbors else 0.0

    elif search_mode == "dense" and dense_scores is not None:
        d_norm = similarity.normalize_scores(dense_scores)
        k_eff = min(k, len(d_norm))
        cand_idx = np.argpartition(-d_norm, k_eff - 1)[:k_eff]
        ordered = cand_idx[np.argsort(-d_norm[cand_idx])]
        neighbors = [
            Neighbor(
                index=int(i),
                score=float(d_norm[i]),
                dense_score=float(d_norm[i]),
            )
            for i in ordered
        ]
        confidence_score = float(np.mean([n.score for n in neighbors])) if neighbors else 0.0

    elif search_mode == "sparse" and sparse_scores is not None:
        k_eff = min(k, len(sparse_scores))
        cand_idx = np.argpartition(-sparse_scores, k_eff - 1)[:k_eff]
        ordered = cand_idx[np.argsort(-sparse_scores[cand_idx])]
        neighbors = [
            Neighbor(
                index=int(i),
                score=float(sparse_scores[i]),
                sparse_score=float(sparse_scores[i]),
            )
            for i in ordered
        ]
        confidence_score = float(np.mean([n.score for n in neighbors])) if neighbors else 0.0

    else:
        # Fallback to TF-IDF cosine similarity
        corpus_vecs = load_corpus_embeddings()
        query_vec = embed_query(text)
        neighbors = similarity.cosine_top_k(query_vec, corpus_vecs, k=k)
        confidence_score = similarity.ood_score(query_vec, corpus_vecs, k=k)

    strength = match_strength_label(confidence_score)

    items: list[MatchItem] = []
    for n in neighbors:
        row = corpus.iloc[n.index]
        items.append(
            MatchItem(
                index=n.index,
                title=_safe_str(row.get("Title")),
                names=_safe_str(row.get("Names")),
                year=_safe_float(row.get("Year")),
                rating=_safe_float(row.get("rating")),
                similarity=round(n.score, 4),
                dense_score=round(n.dense_score, 4) if n.dense_score is not None else None,
                sparse_score=round(n.sparse_score, 4) if n.sparse_score is not None else None,
            )
        )

    return {
        "matches": items,
        "search_mode": search_mode,
        "alpha": alpha,
        "ood_score": round(confidence_score, 4),
        "match_strength": strength,
        "corpus_size": len(corpus),
    }

