"""Shared HTTP client for the Streamlit frontend.

All backend communication goes through here so the API base URL is
configurable in one place and the rest of the UI stays simple.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def api_base_url() -> str:
    """Return the backend base URL, configurable via env var.

    Default: http://localhost:8000 (matches the README's run instructions).
    """
    return os.environ.get("TRANSNLP_API_URL", "http://localhost:8000").rstrip("/")


def _show_error(msg: str) -> None:
    """Log an error and surface it in Streamlit if the runtime is active.

    This avoids importing streamlit at module-load time, so api_client
    can be imported from tests or scripts without side effects.
    """
    logger.error(msg)
    st = sys.modules.get("streamlit")
    if st is not None:
        st.error(msg)


def check_backend_health(base: str) -> bool:
    """Return True iff /health returns 200 with status='ok'."""
    try:
        r = httpx.get(f"{base}/health", timeout=2.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False



def post_match(
    base: str,
    text: str,
    k: int = 5,
    search_mode: str = "hybrid",
    alpha: float = 0.6,
) -> Optional[dict]:
    try:
        r = httpx.post(
            f"{base}/match",
            json={
                "text": text,
                "k": k,
                "search_mode": search_mode,
                "alpha": alpha,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        _show_error(f"/match returned {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        _show_error(f"/match failed: {e}")
        return None
