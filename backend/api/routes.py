"""FastAPI routes for the TransNLP API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from backend import service
from backend.schemas import (
    HealthResponse,
    MatchRequest,
    MatchResponse,
    SpecialsResponse,
    TopicsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.get_health())


@router.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    try:
        result = service.get_match(req.text, k=req.k)
    except FileNotFoundError as e:
        logger.error("Corpus assets missing: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    return MatchResponse(**result)


@router.get("/topics", response_model=TopicsResponse)
def topics() -> TopicsResponse:
    return TopicsResponse(**service.get_topics())


@router.get("/specials", response_model=SpecialsResponse)
def specials(
    topic: str = Query(..., description="Topic label, e.g. 'Politics'"),
    limit: int = Query(20, ge=1, le=100),
) -> SpecialsResponse:
    return SpecialsResponse(**service.get_specials(topic, limit=limit))
