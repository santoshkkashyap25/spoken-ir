"""Unit tests for ai/similarity.py."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Make project root importable when running `pytest` from the project dir.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.similarity import cosine_top_k, ood_score, match_strength_label, Neighbor


@pytest.fixture
def small_corpus() -> np.ndarray:
    """A 4-row toy corpus with clear separation between two themes."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # theme A
            [0.95, 0.0, 0.0, 0.0],  # theme A
            [0.0, 1.0, 0.0, 0.0],  # theme B
            [0.0, 0.95, 0.0, 0.0],  # theme B
        ],
        dtype=np.float32,
    )


def test_cosine_top_k_returns_k_items(small_corpus: np.ndarray) -> None:
    query = np.array([[1.0, 0.0, 0.0, 0.0]])
    neighbors = cosine_top_k(query, small_corpus, k=2)
    assert len(neighbors) == 2
    assert all(isinstance(n, Neighbor) for n in neighbors)


def test_cosine_top_k_picks_closest_first(small_corpus: np.ndarray) -> None:
    query = np.array([[1.0, 0.0, 0.0, 0.0]])
    neighbors = cosine_top_k(query, small_corpus, k=2)
    indices = [n.index for n in neighbors]
    assert indices == [0, 1]


def test_cosine_top_k_handles_1d_query(small_corpus: np.ndarray) -> None:
    query = np.array([0.0, 1.0, 0.0, 0.0])
    neighbors = cosine_top_k(query, small_corpus, k=1)
    assert neighbors[0].index == 2


def test_cosine_top_k_zero_query_returns_zeros(small_corpus: np.ndarray) -> None:
    query = np.array([[0.0, 0.0, 0.0, 0.0]])
    neighbors = cosine_top_k(query, small_corpus, k=3)
    assert len(neighbors) == 3
    assert all(n.score == 0.0 for n in neighbors)


def test_cosine_top_k_k_larger_than_corpus(small_corpus: np.ndarray) -> None:
    query = np.array([[1.0, 0.0, 0.0, 0.0]])
    neighbors = cosine_top_k(query, small_corpus, k=99)
    assert len(neighbors) == small_corpus.shape[0]


def test_cosine_top_k_empty_corpus() -> None:
    query = np.array([[1.0, 0.0]])
    assert cosine_top_k(query, np.zeros((0, 2)), k=5) == []


def test_ood_score_in_distribution(small_corpus: np.ndarray) -> None:
    query = np.array([[1.0, 0.0, 0.0, 0.0]])
    score = ood_score(query, small_corpus, k=2)
    # Both top-2 rows are theme A. With the row-norm floor of 1.0, both
    # get unit length, so both cosines are exactly 1.0.
    assert score == pytest.approx(1.0, abs=1e-6)


def test_ood_score_out_of_distribution(small_corpus: np.ndarray) -> None:
    query = np.array([[0.0, 0.0, 1.0, 0.0]])  # novel third dimension
    score = ood_score(query, small_corpus, k=2)
    # No overlap with either theme → low score.
    assert score < 0.05


def test_ood_score_zero_query() -> None:
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    query = np.array([[0.0, 0.0]])
    assert ood_score(query, corpus) == 0.0


def test_match_strength_label_thresholds() -> None:
    assert match_strength_label(0.6) == "strong"
    assert match_strength_label(0.45) == "strong"
    assert match_strength_label(0.30) == "moderate"
    assert match_strength_label(0.25) == "moderate"
    assert match_strength_label(0.15) == "weak"
    assert match_strength_label(0.10) == "weak"
    assert match_strength_label(0.05) == "very_weak"
    assert match_strength_label(0.0) == "very_weak"
