"""Topic constants and topic-level queries.

The 7 topic labels were learned via LDA on the corpus and are stable
across runs.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import nltk
import numpy as np
import pandas as pd
import pickle

from ai import nltk_setup  # noqa: F401
from ai.corpus import load_corpus
from config import MODELS_DIR

logger = logging.getLogger(__name__)


TOPIC_LABELS: list[str] = [
    "Culture",
    "UK",
    "Crimes",
    "Situational",
    "Immigrants",
    "Relationships",
    "Politics",
]


@lru_cache(maxsize=1)
def load_lda_model():
    path = MODELS_DIR / "lda_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_lda_dictionary():
    path = MODELS_DIR / "lda_model_dict.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_topic_labels() -> list[str]:
    return list(TOPIC_LABELS)


def get_corpus_topic_distribution(corpus: Optional[pd.DataFrame] = None) -> dict[str, float]:
    """Mean of each topic column over the corpus, expressed as percentages."""
    df = corpus if corpus is not None else load_corpus()
    available = [c for c in TOPIC_LABELS if c in df.columns]
    if not available:
        return {label: 0.0 for label in TOPIC_LABELS}
    means = df[available].apply(pd.to_numeric, errors="coerce").mean()
    return {label: float(means.get(label, 0.0)) for label in TOPIC_LABELS}


def get_top_words_per_topic(n: int = 15) -> dict[str, list[str]]:
    """For each topic, return its top-n most probable words from the LDA model."""
    lda = load_lda_model()
    topics = lda.show_topics(num_topics=len(TOPIC_LABELS), num_words=n, formatted=False)
    out: dict[str, list[str]] = {}
    for topic_id, words in topics:
        label = TOPIC_LABELS[topic_id] if topic_id < len(TOPIC_LABELS) else f"topic_{topic_id}"
        out[label] = [w for w, _p in words]
    return out


def get_topic_distribution_for_text(text: str) -> list[float]:
    """Return a 7-element topic probability vector for an arbitrary string."""
    from ai.nlp import preprocess

    lda = load_lda_model()
    dictionary = load_lda_dictionary()
    processed = preprocess(text)
    if not processed.strip():
        return [0.0] * len(TOPIC_LABELS)
    bow = dictionary.doc2bow(processed.split())
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    vec = np.zeros(len(TOPIC_LABELS))
    for topic_id, p in dist:
        if topic_id < len(TOPIC_LABELS):
            vec[topic_id] = p
    return vec.tolist()


def specials_in_topic(
    topic: str,
    corpus: Optional[pd.DataFrame] = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Specials where `topic` is the dominant topic, ordered by topic weight."""
    if topic not in TOPIC_LABELS:
        return pd.DataFrame()
    df = corpus if corpus is not None else load_corpus()
    if topic not in df.columns:
        return pd.DataFrame()
    work = df[["Title", "Names", "Year", "rating", topic]].copy()
    work[topic] = pd.to_numeric(work[topic], errors="coerce").fillna(0.0)
    work = work.sort_values(topic, ascending=False).head(limit)
    return work.reset_index(drop=True)


def avg_rating_for_topic(topic: str, corpus: Optional[pd.DataFrame] = None) -> Optional[float]:
    """Mean IMDb rating across specials whose dominant topic is `topic`."""
    if topic not in TOPIC_LABELS:
        return None
    df = corpus if corpus is not None else load_corpus()
    if "rating" not in df.columns:
        return None
    # Only use topic columns that actually exist in the corpus.
    available_topics = [t for t in TOPIC_LABELS if t in df.columns]
    if not available_topics:
        return None
    dominant = df[available_topics].apply(pd.to_numeric, errors="coerce").idxmax(axis=1)
    ratings = pd.to_numeric(df.loc[dominant == topic, "rating"], errors="coerce").dropna()
    if ratings.empty:
        return None
    return float(ratings.mean())
