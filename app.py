"""Streamlit multi-page entry point.

The frontend is a thin HTTP client. All model and corpus logic lives in
the FastAPI service (run separately with uvicorn).
"""

from __future__ import annotations

import os

import streamlit as st

from utils.api_client import api_base_url, check_backend_health

st.set_page_config(
    page_title="TransNLP",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎤 TransNLP — Stand-up Similarity")

st.write(
    "Find existing stand-up specials your draft most resembles, and explore "
    "the topic landscape of our corpus. Similarity is computed from the "
    "vocabulary and topic mix of your transcript against 500 scraped specials."
)

# Surface backend status in the sidebar so users know if the API is up.
with st.sidebar:
    st.header("Service status")
    base = api_base_url()
    st.write(f"Backend: `{base}`")
    if check_backend_health(base):
        st.success("Backend reachable")
    else:
        st.error(
            "Backend is not reachable. Start it with:\n\n"
            "```\nuvicorn backend.main:app --port 8000\n```"
        )
    st.divider()
    st.caption("Run the backend on `:8000` and the Streamlit app on `:8501`.")

st.info(
    "Use the **sidebar navigation** to go to *Transcript Matcher* (paste a "
    "draft) or *Topic Explorer* (browse the corpus by topic)."
)
