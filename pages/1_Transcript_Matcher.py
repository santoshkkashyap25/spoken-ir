"""Transcript Matcher — main page.

Paste a draft (or upload a .txt) and the app returns the existing specials
that most resemble it, with similarity scores and topic mix.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.api_client import api_base_url, post_match


# --- Render helpers ---
# Defined BEFORE the button handler so Streamlit's rerun-after-click
# doesn't hit a NameError. Streamlit re-executes the entire script on
# every interaction, so the button-handler block must run only after
# these names are bound.


def _render_results(result: dict) -> None:
    """Render the match results: OOD banner + per-match cards."""
    ood = result["ood_score"]
    strength = result["match_strength"]
    corpus_size = result.get("corpus_size", 0)

    if strength == "strong":
        st.success(
            f"**Strong match.** Your draft sits close to our corpus of {corpus_size} "
            f"specials (mean top-{len(result['matches'])} similarity: {ood:.2f}). "
            "The matches below are likely meaningful."
        )
    elif strength == "moderate":
        st.info(
            f"**Moderate match.** Your draft has some thematic overlap with our "
            f"{corpus_size}-special corpus (mean top-{len(result['matches'])} "
            f"similarity: {ood:.2f}). Read the matches with some care."
        )
    elif strength == "weak":
        st.warning(
            f"**Weak match.** Your draft doesn't closely resemble anything in our "
            f"{corpus_size}-special corpus (mean similarity: {ood:.2f}). "
            "The matches below are rough neighbors at best."
        )
    else:
        st.error(
            f"**Very weak match.** Your draft is largely outside the topic/vocabulary "
            f"of our {corpus_size}-special corpus (mean similarity: {ood:.2f}). "
            "Either the corpus doesn't cover your style yet, or your draft uses "
            "vocabulary we don't have a strong signal for."
        )

    st.subheader("Top matches")

    for i, m in enumerate(result["matches"], start=1):
        _render_match_card(rank=i, m=m, total=len(result["matches"]))


def _render_match_card(rank: int, m: dict, total: int) -> None:
    """Render a single match as an expander card."""
    title = m.get("title") or "(untitled)"
    names = m.get("names") or "Unknown"
    year = m.get("year")
    rating = m.get("rating")
    sim = m["similarity"]

    year_str = f" ({int(year)})" if year is not None else ""
    rating_str = f" · IMDb {rating:.1f}" if rating is not None else ""

    header = f"#{rank} — **{title}** — {names}{year_str}{rating_str}  ·  similarity {sim:.3f}"

    with st.expander(header, expanded=(rank <= min(3, total))):
        st.progress(min(max(sim, 0.0), 1.0))

        st.write(f"**Cosine similarity:** {sim:.3f}")
        if rating is not None:
            st.write(f"**IMDb rating:** {rating:.1f}")
        if year is not None:
            st.write(f"**Year:** {int(year)}")

        topic_mix = m.get("topic_mix", {})
        if topic_mix:
            tm_df = pd.DataFrame(
                {
                    "Topic": list(topic_mix.keys()),
                    "Proportion": [float(v) for v in topic_mix.values()],
                }
            ).sort_values("Proportion", ascending=False)
            st.bar_chart(tm_df.set_index("Topic"))


# --- Page body ---

st.title("🎤 Transcript Matcher")
st.write(
    "Paste a draft stand-up transcript — a paragraph is enough — and we'll "
    "show you the existing specials in our corpus that most resemble it. "
    "Similarity is computed from vocabulary and topic mix; this is a "
    "neighborhood lookup, not a forecast of success."
)

base = api_base_url()

# --- Input ---
col_input, col_upload = st.columns([3, 1])
with col_input:
    input_text = st.text_area(
        "Paste your draft here:",
        height=240,
        placeholder=(
            "e.g., 'I went home for Thanksgiving and my mother had started "
            "dating again. She's 73. She met a man at the community center. "
            "He knits. They watched four episodes of a Swedish crime show "
            "in one weekend. I'm concerned.'"
        ),
    )
with col_upload:
    uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])

content = ""
if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8")
        st.success(f"Loaded {len(content):,} characters from {uploaded.name}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read file: {e}")
elif input_text:
    content = input_text

k = st.slider("How many similar specials to show?", min_value=1, max_value=10, value=5)

# --- Run ---
if st.button("Find similar specials", type="primary"):
    if not content or len(content.strip()) < 50:
        st.warning(
            "Please paste a draft of at least 50 characters. The similarity "
            "search needs enough text to be meaningful."
        )
    else:
        with st.spinner("Finding the closest specials in the corpus..."):
            result = post_match(base, content, k=k)

        if result is None:
            st.error("The backend did not return a result. Check that the API is running.")
        else:
            _render_results(result)
