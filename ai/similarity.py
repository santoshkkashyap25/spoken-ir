"""Similarity search over the corpus.

Cosine similarity between a query TF-IDF vector and every row in the
precomputed corpus matrix. Returns top-k neighbors with their row index
and similarity score, plus an OOD (out-of-distribution) score = mean of
the top-k similarities — a single number that says "how close is this
draft to the corpus at all?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neighbor:
    index: int
    score: float





def cosine_top_k(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    k: int = 5,
) -> list[Neighbor]:
    """Return the top-k corpus rows by cosine similarity to `query_vec`.

    `query_vec` may be shape (V,) or (1, V). `corpus_matrix` is (N, V).
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if corpus_matrix.shape[0] == 0:
        return []

    # Normalize the query, corpus is already pre-normalized
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0.0:
        # Empty query — return zero scores instead of NaNs.
        k_eff = min(k, corpus_matrix.shape[0])
        return [Neighbor(index=int(i), score=0.0) for i in range(k_eff)]
    q = query_vec / q_norm

    scores = (corpus_matrix @ q.T).ravel()
    k_eff = min(k, scores.shape[0])
    # argpartition is O(n); the sort over the k_eff winners is O(k log k).
    candidate_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
    ordered = candidate_idx[np.argsort(-scores[candidate_idx])]
    return [Neighbor(index=int(i), score=float(scores[i])) for i in ordered]


def ood_score(
    query_vec: np.ndarray,
    corpus_matrix: np.ndarray,
    k: int = 5,
) -> float:
    """Mean of the top-k cosine similarities — a 'how in-distribution is this?' number.

    Returns 0.0 for an empty query. The score is in [0, 1] because cosine
    similarity over non-negative TF-IDF vectors is non-negative.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if np.linalg.norm(query_vec) == 0.0 or corpus_matrix.shape[0] == 0:
        return 0.0
    neighbors = cosine_top_k(query_vec, corpus_matrix, k=k)
    if not neighbors:
        return 0.0
    return float(np.mean([n.score for n in neighbors]))


def match_strength_label(score: float) -> str:
    """Map an OOD score to a human-readable strength label."""
    if score >= 0.45:
        return "strong"
    if score >= 0.25:
        return "moderate"
    if score >= 0.10:
        return "weak"
    return "very_weak"
