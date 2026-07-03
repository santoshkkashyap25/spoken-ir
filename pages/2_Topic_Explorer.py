"""Topic Explorer — second page.

Browse the corpus by topic. Pick a topic and see its top words, how many
specials carry it, the average IMDb rating for specials where it dominates,
and the list of specials themselves.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.api_client import api_base_url, get_specials, get_topics

st.title("📚 Topic Explorer")
st.write(
    "Browse the 7 topics our LDA model learned from the corpus. Click a "
    "topic to see its top words and the specials that carry it most strongly."
)

base = api_base_url()

topics_data = get_topics(base)
if not topics_data:
    st.error("Could not load topics from the backend. Is it running?")
    st.stop()

topics = topics_data["topics"]
labels = [t["name"] for t in topics]

# Topic overview
st.subheader("Corpus at a glance")
overview_rows = [
    {
        "Topic": t["name"],
        "Specials (with weight > 0)": t["special_count"],
        "Corpus share": f"{t['corpus_share_pct']:.1f}%",
        "Avg rating (when dominant)": f"{t['avg_rating']:.2f}" if t["avg_rating"] is not None else "—",
    }
    for t in topics
]
st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

# Topic detail
st.subheader("Topic detail")
selected = st.selectbox("Pick a topic to dig into:", labels)

t = next(t for t in topics if t["name"] == selected)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("**Top words**")
    st.write(", ".join(t["top_words"]))
    st.metric("Specials carrying this topic", t["special_count"])
with col2:
    st.metric("Corpus share", f"{t['corpus_share_pct']:.1f}%")
    st.metric(
        "Avg IMDb rating (when dominant)",
        f"{t['avg_rating']:.2f}" if t["avg_rating"] is not None else "—",
    )

st.divider()
st.subheader(f"Top specials for **{selected}**")
limit = st.slider("How many?", min_value=5, max_value=50, value=15, key="specials_limit")

specials_data = get_specials(base, selected, limit=limit)
if not specials_data:
    st.error("Could not load specials for this topic.")
    st.stop()

specials = specials_data.get("specials", [])
if not specials:
    st.info("No specials carry this topic in our corpus.")
else:
    df = pd.DataFrame(
        [
            {
                "Title": s.get("title") or "(untitled)",
                "Comedian": s.get("names") or "Unknown",
                "Year": int(s["year"]) if s.get("year") is not None else "—",
                "IMDb": f"{s['rating']:.1f}" if s.get("rating") is not None else "—",
                "Topic weight": round(s.get("topic_weight", 0.0), 4),
            }
            for s in specials
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
