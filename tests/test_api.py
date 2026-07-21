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
LDA_MODEL = ROOT / "data" / "models" / "lda_model.pkl"
LDA_DICT = ROOT / "data" / "models" / "lda_model_dict.pkl"
TFIDF = ROOT / "data" / "models" / "tfidf_vectorizer.pkl"

skip_reason = "Required corpus artifacts not on disk; run preprocess_data + build_corpus_embeddings first."
needs_artifacts = pytest.mark.skipif(
    not (CORPUS_CSV.exists() and EMBEDDINGS.exists() and LDA_MODEL.exists() and LDA_DICT.exists() and TFIDF.exists()),
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
        "The president, the senators, the talking heads — all of them are doing bits. "
        "Stand-up comedy has nothing on a State of the Union address.",
        "k": 5,
    }
    r = client.post("/match", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["matches"]) == 5
    assert all(m["similarity"] >= 0.0 for m in body["matches"])
    # Top match should be at least as similar as the last one.
    sims = [m["similarity"] for m in body["matches"]]
    assert sims == sorted(sims, reverse=True)
    assert body["corpus_size"] > 0
    assert body["match_strength"] in {"strong", "moderate", "weak", "very_weak"}


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
    # Empty query → OOD score of 0, all similarities 0.
    assert body["ood_score"] == 0.0
    assert all(m["similarity"] == 0.0 for m in body["matches"])


@needs_artifacts
def test_topics_endpoint(client) -> None:
    r = client.get("/topics")
    assert r.status_code == 200
    topics = r.json()["topics"]
    assert len(topics) == 7
    expected = {"Culture", "UK", "Crimes", "Situational", "Immigrants", "Relationships", "Politics"}
    assert {t["name"] for t in topics} == expected
    for t in topics:
        assert len(t["top_words"]) > 0
        assert isinstance(t["special_count"], int)


@needs_artifacts
def test_specials_endpoint(client) -> None:
    r = client.get("/specials", params={"topic": "Politics", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] == "Politics"
    assert len(body["specials"]) <= 5
    for s in body["specials"]:
        assert 0.0 <= s["topic_weight"] <= 1.0


@needs_artifacts
def test_specials_unknown_topic_returns_empty(client) -> None:
    r = client.get("/specials", params={"topic": "BogusTopic", "limit": 5})
    assert r.status_code == 200
    assert r.json()["specials"] == []
