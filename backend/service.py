"""Service layer: thin orchestration over the `ai/` layer.

No ML logic here. No HTTP here. The only job is to translate between
the AI layer's primitives and the Pydantic response shapes.
"""

from __future__ import annotations

import logging
import math
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd

from ai import similarity, topics
from ai.corpus import load_corpus
from ai.embed import embed_query, load_corpus_embeddings, load_corpus_topic_vectors
from ai.similarity import Neighbor, match_strength_label

from backend.schemas import MatchItem, SpecialInfo, TopicInfo

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
def get_match(text: str, k: int) -> dict:
    """Find the k corpus specials most similar to `text`."""
    corpus = load_corpus()
    corpus_vecs = load_corpus_embeddings()
    topic_vecs = load_corpus_topic_vectors()
    query_vec = embed_query(text)
    neighbors = similarity.cosine_top_k(query_vec, corpus_vecs, k=k)
    ood = similarity.ood_score(query_vec, corpus_vecs, k=k)
    strength = match_strength_label(ood)

    items: list[MatchItem] = []
    for n in neighbors:
        row = corpus.iloc[n.index]
        if n.index < len(topic_vecs):
            topic_mix = {
                label: float(topic_vecs[n.index, i])
                for i, label in enumerate(topics.TOPIC_LABELS)
            }
        else:
            topic_mix = {label: 0.0 for label in topics.TOPIC_LABELS}
        items.append(
            MatchItem(
                index=n.index,
                title=_safe_str(row.get("Title")),
                names=_safe_str(row.get("Names")),
                year=_safe_float(row.get("Year")),
                rating=_safe_float(row.get("rating")),
                similarity=n.score,
                topic_mix=topic_mix,
            )
        )

    return {
        "matches": items,
        "ood_score": ood,
        "match_strength": strength,
        "corpus_size": len(corpus),
    }


def get_topics() -> dict:
    """Build the /topics response body."""
    corpus = load_corpus()
    topic_vecs = load_corpus_topic_vectors()
    top_words = topics.get_top_words_per_topic(n=15)

    # Special count = specials where this topic is the DOMINANT one
    # (highest probability), not just any non-zero. This is what
    # 'specials in this topic' should mean to a user.
    dominant_idx = topic_vecs.argmax(axis=1)
    counts = [int((dominant_idx == i).sum()) for i in range(len(topics.TOPIC_LABELS))]

    shares = topic_vecs.mean(axis=0)
    total_share = float(shares.sum()) or 1.0

    dominant_ratings: list[list[float]] = [[] for _ in topics.TOPIC_LABELS]
    ratings = pd.to_numeric(corpus["rating"], errors="coerce").tolist()
    for i, t in enumerate(dominant_idx):
        r = ratings[i]
        if r is not None and not (isinstance(r, float) and math.isnan(r)):
            dominant_ratings[int(t)].append(float(r))

    out: list[TopicInfo] = []
    for i, label in enumerate(topics.TOPIC_LABELS):
        avg = sum(dominant_ratings[i]) / len(dominant_ratings[i]) if dominant_ratings[i] else None
        out.append(
            TopicInfo(
                name=label,
                top_words=top_words.get(label, []),
                special_count=counts[i],
                avg_rating=avg,
                corpus_share_pct=round(100.0 * float(shares[i]) / total_share, 2),
            )
        )
    return {"topics": out}


def get_specials(topic: str, limit: int = 20) -> dict:
    """Build the /specials response body for a single topic."""
    corpus = load_corpus()
    topic_vecs = load_corpus_topic_vectors()
    if topic not in topics.TOPIC_LABELS:
        return {"topic": topic, "specials": []}

    idx = topics.TOPIC_LABELS.index(topic)
    weights = topic_vecs[:, idx]
    order = np.argsort(-weights)[:limit]

    items: list[SpecialInfo] = []
    for i in order:
        row = corpus.iloc[int(i)]
        items.append(
            SpecialInfo(
                title=_safe_str(row.get("Title")),
                names=_safe_str(row.get("Names")),
                year=_safe_float(row.get("Year")),
                rating=_safe_float(row.get("rating")),
                topic_weight=float(weights[int(i)]),
            )
        )
    return {"topic": topic, "specials": items}
