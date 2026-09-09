"""Integration tests for the FastAPI app.

These tests hit the live corpus on disk. They assume:
  - data/processed/processed_content_data.csv exists
  - data/ai/corpus_embeddings.npy exists
  - data/models/{lda_model,lda_model_dict,tfidf_vectorizer}.pkl exist

If any of those are missing, the tests are skipped with a clear message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CORPUS_CSV = ROOT / "data" / "processed" / "processed_content_data.csv"
EMBEDDINGS = ROOT / "data" / "ai" / "corpus_embeddings.npy"
TFIDF = ROOT / "data" / "models" / "tfidf_vectorizer.pkl"

skip_reason = "Required corpus artifacts not on disk; run preprocess_data + build_corpus_embeddings first."
needs_artifacts = pytest.mark.skipif(
    not (CORPUS_CSV.exists() and EMBEDDINGS.exists() and TFIDF.exists()),
    reason=skip_reason,
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


@needs_artifacts
def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["corpus_size"], int)
    assert body["corpus_size"] > 0


@needs_artifacts
def test_match_returns_top_k_with_schema(client) -> None:
    payload = {
        "text": "Politics is a circus and everyone in Washington is performing. "
        "I watched the news for an hour and laughed the whole time. "
        "The president, the senators, the talking heads — all of them are doing bits.",
        "k": 5,
        "search_mode": "hybrid",
        "alpha": 0.6,
    }
    r = client.post("/match", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["matches"]) == 5
    assert body["search_mode"] == "hybrid"
    assert all(m["similarity"] >= 0.0 for m in body["matches"])
    # Top match should be at least as similar as the last one.
    sims = [m["similarity"] for m in body["matches"]]
    assert sims == sorted(sims, reverse=True)
    assert body["corpus_size"] > 0
    assert body["match_strength"] in {"strong", "moderate", "weak", "very_weak"}


@needs_artifacts
def test_match_dense_and_sparse_modes(client) -> None:
    text = "Crime rates in metropolitan areas have sparked fierce policy debates."
    # Dense mode
    r_dense = client.post("/match", json={"text": text, "k": 3, "search_mode": "dense"})
    assert r_dense.status_code == 200
    assert r_dense.json()["search_mode"] == "dense"
    assert len(r_dense.json()["matches"]) == 3

    # Sparse mode
    r_sparse = client.post("/match", json={"text": text, "k": 3, "search_mode": "sparse"})
    assert r_sparse.status_code == 200
    assert r_sparse.json()["search_mode"] == "sparse"
    assert len(r_sparse.json()["matches"]) == 3


@needs_artifacts
def test_match_rejects_too_short_text(client) -> None:
    r = client.post("/match", json={"text": "hi", "k": 5})
    assert r.status_code == 422


@needs_artifacts
def test_match_handles_empty_query_gracefully(client) -> None:
    # 10+ chars of whitespace — minimum validator passes but content is empty.
    r = client.post("/match", json={"text": " " * 20, "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["ood_score"] == 0.0
    assert all(m["similarity"] == 0.0 for m in body["matches"])

