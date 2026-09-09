"""Pydantic schemas for the API surface."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Query transcript or text passage")
    k: int = Field(5, ge=1, le=20, description="Number of neighbors to return")
    search_mode: Literal["hybrid", "dense", "sparse"] = Field(
        "hybrid",
        description="Retrieval mode: 'hybrid' (Dense + BM25), 'dense' (MiniLM), or 'sparse' (BM25)",
    )
    alpha: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense retrieval in hybrid mode (1.0 = dense only, 0.0 = sparse only)",
    )


class MatchItem(BaseModel):
    index: int
    title: Optional[str] = None
    names: Optional[str] = None
    year: Optional[float] = None
    rating: Optional[float] = None
    similarity: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class MatchResponse(BaseModel):
    matches: list[MatchItem]
    search_mode: str = "hybrid"
    alpha: float = 0.6
    ood_score: float
    match_strength: str
    corpus_size: int


class HealthResponse(BaseModel):
    status: str
    corpus_size: int

