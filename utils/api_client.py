"""Shared HTTP client for the Streamlit frontend.

All backend communication goes through here so the API base URL is
configurable in one place and the rest of the UI stays simple.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx


def api_base_url() -> str:
    """Return the backend base URL, configurable via env var.

    Default: http://localhost:8000 (matches the README's run instructions).
    """
    return os.environ.get("TRANSNLP_API_URL", "http://localhost:8000").rstrip("/")


def check_backend_health(base: str) -> bool:
    """Return True iff /health returns 200 with status='ok'."""
    try:
        r = httpx.get(f"{base}/health", timeout=2.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def get_topics(base: str) -> Optional[dict]:
    try:
        r = httpx.get(f"{base}/topics", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st_error_console(f"/topics failed: {e}")
        return None


def get_specials(base: str, topic: str, limit: int = 20) -> Optional[dict]:
    try:
        r = httpx.get(
            f"{base}/specials",
            params={"topic": topic, "limit": limit},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st_error_console(f"/specials failed: {e}")
        return None


def post_match(base: str, text: str, k: int = 5) -> Optional[dict]:
    try:
        r = httpx.post(
            f"{base}/match",
            json={"text": text, "k": k},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st_error_console(f"/match returned {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st_error_console(f"/match failed: {e}")
        return None


def st_error_console(msg: str) -> None:
    """Tiny shim so this module doesn't import streamlit at module-load time."""
    import streamlit as st

    st.error(msg)
