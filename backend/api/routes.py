"""FastAPI routes for the TransNLP API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from backend import service
from backend.schemas import (
    HealthResponse,
    MatchRequest,
    MatchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.get_health())


@router.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    """Retrieve top-k transcripts via hybrid, dense, or sparse search."""
    try:
        result = service.get_match(
            req.text,
            k=req.k,
            search_mode=req.search_mode,
            alpha=req.alpha,
        )
    except FileNotFoundError as e:
        logger.error("Corpus assets missing: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    return MatchResponse(**result)

