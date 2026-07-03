"""Pydantic schemas for the API surface."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Draft transcript text")
    k: int = Field(5, ge=1, le=20, description="Number of neighbors to return")


class MatchItem(BaseModel):
    index: int
    title: Optional[str] = None
    names: Optional[str] = None
    year: Optional[float] = None
    rating: Optional[float] = None
    similarity: float
    topic_mix: dict[str, float]


class MatchResponse(BaseModel):
    matches: list[MatchItem]
    ood_score: float
    match_strength: str
    corpus_size: int


class TopicInfo(BaseModel):
    name: str
    top_words: list[str]
    special_count: int
    avg_rating: Optional[float] = None
    corpus_share_pct: float


class TopicsResponse(BaseModel):
    topics: list[TopicInfo]


class SpecialInfo(BaseModel):
    title: Optional[str] = None
    names: Optional[str] = None
    year: Optional[float] = None
    rating: Optional[float] = None
    topic_weight: float


class SpecialsResponse(BaseModel):
    topic: str
    specials: list[SpecialInfo]


class HealthResponse(BaseModel):
    status: str
    corpus_size: int
