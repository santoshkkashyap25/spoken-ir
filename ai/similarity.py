"""Similarity search and hybrid information retrieval over the transcript corpus.

Provides:
- Dense semantic retrieval (cosine similarity over dense embeddings)
- Sparse lexical retrieval (BM25 scoring over tokenized documents)
- Hybrid retrieval (convex score fusion of dense semantic + sparse BM25)
- Reciprocal Rank Fusion (RRF) for rank-based hybrid retrieval
- Out-of-distribution / retrieval confidence scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neighbor:
    index: int
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


def cosine_top_k(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    k: int = 5,
) -> list[Neighbor]:
    """Return the top-k corpus rows by cosine similarity to `query_vec`.

    `query_vec` may be shape (V,) or (1, V). `corpus_matrix` is (N, V).
    Computes exact cosine similarity, handling both normalized and unnormalized inputs.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if corpus_matrix.shape[0] == 0:
        return []

    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        k_eff = min(k, corpus_matrix.shape[0])
        return [Neighbor(index=int(i), score=0.0) for i in range(k_eff)]
    q = query_vec / q_norm

    # Dot product
    raw_scores = (corpus_matrix @ q.T).ravel()

    # Compute corpus norms for exact cosine similarity
    c_norms = np.linalg.norm(corpus_matrix, axis=1)
    c_norms[c_norms == 0.0] = 1.0
    scores = raw_scores / c_norms

    k_eff = min(k, scores.shape[0])
    candidate_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
    ordered = candidate_idx[np.argsort(-scores[candidate_idx])]
    return [Neighbor(index=int(i), score=float(scores[i])) for i in ordered]


def dense_top_k(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    k: int = 5,
) -> list[Neighbor]:
    """Top-k search using dense embeddings (e.g., all-MiniLM-L6-v2)."""
    return cosine_top_k(query_vec, corpus_matrix, k=k)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize raw scores into [0, 1] range using min-max scaling."""
    if len(scores) == 0:
        return scores
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max - s_min < 1e-9:
        if s_max > 0:
            return np.ones_like(scores, dtype=np.float32)
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - s_min) / (s_max - s_min)


def bm25_top_k(
    query_tokens: list[str],
    bm25_index,
    k: int = 5,
) -> list[Neighbor]:
    """Return top-k corpus items scored by BM25, normalized to [0, 1]."""
    if not query_tokens or bm25_index is None:
        return []
    raw_scores = np.asarray(bm25_index.get_scores(query_tokens), dtype=np.float32)
    norm_scores = normalize_scores(raw_scores)

    k_eff = min(k, len(norm_scores))
    if k_eff == 0:
        return []

    candidate_idx = np.argpartition(-norm_scores, k_eff - 1)[:k_eff]
    ordered = candidate_idx[np.argsort(-norm_scores[candidate_idx])]
    return [
        Neighbor(
            index=int(i),
            score=float(norm_scores[i]),
            sparse_score=float(norm_scores[i]),
        )
        for i in ordered
    ]


def hybrid_top_k(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    alpha: float = 0.6,
    k: int = 5,
) -> list[Neighbor]:
    """Combine dense semantic scores and sparse lexical scores via convex combination.

    Score(d) = alpha * Dense(d) + (1 - alpha) * Sparse(d)
    where alpha in [0, 1]. 1.0 = 100% dense, 0.0 = 100% sparse.
    """
    n = len(dense_scores)
    if n == 0 or len(sparse_scores) != n:
        return []

    alpha = max(0.0, min(1.0, float(alpha)))
    d_norm = normalize_scores(dense_scores)
    s_norm = normalize_scores(sparse_scores)

    combined = alpha * d_norm + (1.0 - alpha) * s_norm
    k_eff = min(k, n)
    candidate_idx = np.argpartition(-combined, k_eff - 1)[:k_eff]
    ordered = candidate_idx[np.argsort(-combined[candidate_idx])]

    return [
        Neighbor(
            index=int(i),
            score=float(combined[i]),
            dense_score=float(d_norm[i]),
            sparse_score=float(s_norm[i]),
        )
        for i in ordered
    ]


def reciprocal_rank_fusion(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    c: int = 60,
    k: int = 5,
) -> list[Neighbor]:
    """Rank-based fusion (RRF): RRF(d) = 1/(c + rank_dense(d)) + 1/(c + rank_sparse(d))."""
    n = len(dense_scores)
    if n == 0 or len(sparse_scores) != n:
        return []

    dense_ranks = np.empty(n, dtype=int)
    dense_ranks[np.argsort(-dense_scores)] = np.arange(1, n + 1)

    sparse_ranks = np.empty(n, dtype=int)
    sparse_ranks[np.argsort(-sparse_scores)] = np.arange(1, n + 1)

    rrf_scores = (1.0 / (c + dense_ranks)) + (1.0 / (c + sparse_ranks))
    norm_rrf = normalize_scores(rrf_scores)

    k_eff = min(k, n)
    candidate_idx = np.argpartition(-norm_rrf, k_eff - 1)[:k_eff]
    ordered = candidate_idx[np.argsort(-norm_rrf[candidate_idx])]

    d_norm = normalize_scores(dense_scores)
    s_norm = normalize_scores(sparse_scores)

    return [
        Neighbor(
            index=int(i),
            score=float(norm_rrf[i]),
            dense_score=float(d_norm[i]),
            sparse_score=float(s_norm[i]),
        )
        for i in ordered
    ]


def ood_score(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    k: int = 5,
) -> float:
    """Mean of the top-k cosine similarities — measures how well the query fits the corpus."""
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if np.linalg.norm(query_vec) == 0.0 or corpus_matrix.shape[0] == 0:
        return 0.0
    neighbors = cosine_top_k(query_vec, corpus_matrix, k=k)
    if not neighbors:
        return 0.0
    return float(np.mean([n.score for n in neighbors]))


def match_strength_label(score: float) -> str:
    """Map a retrieval similarity score to a human-readable match strength."""
    if score >= 0.45:
        return "strong"
    if score >= 0.25:
        return "moderate"
    if score >= 0.10:
        return "weak"
    return "very_weak"
