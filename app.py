"""Spoken-IR — Stand-Up Comedy Semantic Search & Hybrid Retrieval.

A unified platform benchmarking:
1. Dense Bi-Encoder Embeddings (sentence-transformers/all-MiniLM-L6-v2)
2. Probabilistic Lexical Relevance (BM25Okapi)
3. Convex Hybrid Score Fusion & Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.api_client import (
    api_base_url,
    check_backend_health,
    post_match,
)

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spoken-IR — Stand-Up Comedy Semantic Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling: Tab Spacing, Typography & Cards ────────────────────────────
st.markdown(
    """
    <style>
    /* ── Page Padding & Max Width ── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1220px;
    }

    /* ── Tab Bar Container & Spacing ── */
    div[data-testid="stTabs"] {
        margin-top: 1.5rem;
    }

    div[data-testid="stTabs"] > div[role="tablist"] {
        gap: 1.75rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.22);
        padding-bottom: 2px;
    }

    /* ── Individual Tabs ── */
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 1.05rem;
        font-weight: 500;
        padding: 0.75rem 1.6rem;
        border-radius: 6px 6px 0 0;
        transition: all 0.2s ease-in-out;
        border: none;
        background: transparent;
    }

    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] button[role="tab"] span {
        font-size: 1.05rem;
        font-weight: 500;
        transition: color 0.2s ease-in-out;
    }

    /* Hover state: highlight with primary accent, never dark-on-dark */
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background-color: rgba(128, 128, 128, 0.12);
    }

    div[data-testid="stTabs"] button[role="tab"]:hover p,
    div[data-testid="stTabs"] button[role="tab"]:hover span {
        color: var(--primary-color, #38bdf8) !important;
    }

    /* Active selected tab */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        border-bottom: 3px solid var(--primary-color, #0284c7) !important;
        background-color: transparent;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span {
        color: var(--primary-color, #0284c7) !important;
        font-weight: 600;
    }


    /* ── Tab Content Spacing ── */
    div[data-testid="stTabs"] > div[role="tabpanel"] {
        padding-top: 2rem;
    }

    /* ── Metric Box Styling ── */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
    }

    /* ── Result Cards ── */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Spoken-IR: Stand-Up Comedy Semantic Search & Hybrid Retrieval")
st.caption(
    "Hybrid Information Retrieval comparing Dense Bi-Encoders (MiniLM-L6-v2) and BM25 Lexical Ranking "
    "across 582 stand-up comedy special transcripts."
)

base = api_base_url()
backend_alive = check_backend_health(base)

# ── Header Health Status ───────────────────────────────────────────────────────
if not backend_alive:
    st.error(
        f"Backend Service Offline (`{base}`). "
        "Launch the FastAPI backend service via:\n\n"
        "```bash\n.venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000\n```"
    )

# ── Core Benchmark Tabs ────────────────────────────────────────────────────────
tab_search, tab_theory = st.tabs(
    [
        "Retrieval Benchmark & Query Engine",
        "Empirical Methodology & Formulations",
    ]
)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RETRIEVAL BENCHMARK & QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("Search Stand-Up Comedy Transcripts")
    st.markdown(
        "Find comedy routines, jokes, and specials by comparing dense semantic vectors (`all-MiniLM-L6-v2`) "
        "against exact BM25 keyword matching and tunable hybrid score fusion."
    )

    # Retrieval Configuration
    cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 2, 1])

    with cfg_col1:
        mode_label = st.radio(
            "Retrieval Strategy",
            options=[
                "Hybrid (Dense + BM25 Fusion)",
                "Dense Semantic (all-MiniLM-L6-v2)",
                "Sparse Lexical (BM25Okapi)",
            ],
            index=0,
            horizontal=False,
        )
        mode_key = "hybrid" if "Hybrid" in mode_label else ("dense" if "Dense" in mode_label else "sparse")

    with cfg_col2:
        if mode_key == "hybrid":
            alpha = st.slider(
                "Dense Weight (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
                help="1.0 = Purely Dense Semantic | 0.0 = Purely BM25 Lexical",
            )
            st.caption(
                f"Weight Allocation: **{int(alpha*100)}% Dense Semantic** / "
                f"**{int((1-alpha)*100)}% BM25 Lexical**"
            )
        else:
            alpha = 1.0 if mode_key == "dense" else 0.0
            st.info(f"Isolated Single-Mode Strategy: **{mode_key.upper()}**")

    with cfg_col3:
        k = st.number_input("Top-k Depth", min_value=1, max_value=20, value=5)

    st.divider()

    # Query Input
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        input_text = st.text_area(
            "Input query passage or transcript excerpt:",
            height=180,
            placeholder=(
                "e.g., 'The psychological fear of aging, knee pain, and realizing your parents were right.'\n"
                "or: 'A comedian talking about buying candy at Walgreens and old guy phrases.'"
            ),
        )
    with q_col2:
        uploaded_file = st.file_uploader(
            "Or upload document (PDF, Markdown, Text):",
            type=["pdf", "md", "txt"],
            help="Extract text from PDF (.pdf), Markdown (.md), or plain text (.txt) files",
        )


    query_content = ""
    if uploaded_file is not None:
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith(".pdf"):
                import io
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(uploaded_file.read()))
                pages_text = [page.extract_text() or "" for page in reader.pages]
                query_content = "\n".join(pages_text).strip()
                st.success(f"Extracted {len(query_content):,} characters from {len(reader.pages)} PDF page(s)")
            else:
                query_content = uploaded_file.read().decode("utf-8", errors="replace")
                st.success(f"Loaded {len(query_content):,} characters from `{uploaded_file.name}`")

            if query_content:
                with st.expander("Preview Extracted Document Content", expanded=False):
                    st.text(query_content[:1500] + ("..." if len(query_content) > 1500 else ""))
        except Exception as e:
            st.error(f"Error extracting text from file: {e}")
    elif input_text:
        query_content = input_text


    if st.button("Search Stand-Up Specials", type="primary"):
        if not query_content or len(query_content.strip()) < 20:
            st.warning("Please provide a query passage of at least 20 characters.")
        elif not backend_alive:
            st.error("Backend API is unreachable. Ensure the FastAPI service is active on port 8000.")
        else:
            with st.spinner(f"Computing {mode_key.upper()} scores across 582 comedy specials..."):
                result = post_match(base, query_content, k=k, search_mode=mode_key, alpha=alpha)

            if result is None:
                st.error("Retrieval failed. Inspect server logs for details.")
            else:
                ood = result["ood_score"]
                strength = result["match_strength"]
                matches = result["matches"]
                corpus_n = result.get("corpus_size", 582)

                # Confidence Metric Banner
                b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
                with b_col1:
                    st.metric("Mean Top-k Alignment Score", f"{ood:.4f}")
                with b_col2:
                    st.metric("Distribution Confidence", strength.upper())
                with b_col3:
                    st.metric("Evaluated Corpus Documents", corpus_n)

                st.markdown(f"### Ranked Results (Top-{len(matches)})")

                for rank, m in enumerate(matches, start=1):
                    title = m.get("title") or "(Untitled Document)"
                    speaker = m.get("names") or "Unspecified Speaker"
                    year = m.get("year")
                    rating = m.get("rating")
                    score = m["similarity"]
                    d_score = m.get("dense_score")
                    s_score = m.get("sparse_score")

                    year_badge = f" ({int(year)})" if year is not None else ""
                    rating_badge = f" · Rating {rating:.1f}" if rating is not None else ""

                    card_title = f"Rank #{rank}: **{title}** — {speaker}{year_badge}{rating_badge} | Score: {score:.4f}"

                    with st.expander(card_title, expanded=(rank <= 3)):
                        st.progress(min(max(score, 0.0), 1.0))

                        # Score Decomposition
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.metric("Final Fused Score", f"{score:.4f}")
                        with sc2:
                            st.metric(
                                "Dense Cosine S_dense",
                                f"{d_score:.4f}" if d_score is not None else "N/A",
                            )
                        with sc3:
                            st.metric(
                                "BM25 Lexical S_sparse",
                                f"{s_score:.4f}" if s_score is not None else "N/A",
                            )

                        # Metadata
                        m1, m2 = st.columns(2)
                        with m1:
                            st.write(f"**Primary Speaker / Creator:** {speaker}")
                            if year is not None:
                                st.write(f"**Year of Recording:** {int(year)}")
                        with m2:
                            if rating is not None:
                                st.write(f"**Audience / Reference Rating:** {rating:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EMPIRICAL METHODOLOGY & THEORETICAL FORMULATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_theory:

    st.subheader("Theoretical Grounding & Mathematical Formulations")
    st.markdown(
        "This project investigates the empirical trade-offs between **dense continuous semantic representations** "
        "and **sparse probabilistic lexical scoring (BM25)** when applied to stand-up comedy transcripts."
    )

    st.markdown("---")


    # 1. Dense Bi-Encoder
    st.markdown("### 1. Dense Semantic Retrieval (Bi-Encoder)")
    st.markdown(
        r"Given a query passage $q$ and a document $d$, a pre-trained transformer encoder "
        r"$\mathcal{E}_{\phi}(\cdot)$ maps each text sequence to a fixed-dimensional continuous embedding vector "
        r"$\mathbf{e} \in \mathbb{R}^{D}$ (with $D=384$ using `all-MiniLM-L6-v2`):"
    )
    st.latex(r"\mathbf{e}_q = \mathcal{E}_{\phi}(q), \quad \mathbf{e}_d = \mathcal{E}_{\phi}(d)")
    st.markdown("Cosine similarity is computed via dot-product over unit-normalized representations:")
    st.latex(
        r"S_{\text{dense}}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2}"
    )

    st.markdown("---")

    # 2. Sparse BM25
    st.markdown("### 2. Sparse Lexical Retrieval (BM25Okapi)")
    st.markdown(
        r"BM25 establishes relevance based on non-linear term frequency saturation and document length normalization. "
        r"For query tokens $t \in q$ and document $d$:"
    )
    st.latex(
        r"\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}"
    )
    st.markdown("Inverse Document Frequency (IDF) penalizes ubiquitously distributed terms:")
    st.latex(
        r"\text{IDF}(t) = \ln \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)"
    )

    st.markdown("---")

    # 3. Hybrid Fusion
    st.markdown("### 3. Convex Score Fusion & Reciprocal Rank Fusion")
    st.markdown(
        r"Individual raw scores are normalized into the interval $[0, 1]$ via min-max scaling $\bar{S}$. "
        r"Convex score combination interpolates between contextual abstraction and lexical exactness:"
    )
    st.latex(
        r"S_{\text{hybrid}}(q, d) = \alpha \cdot \bar{S}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \bar{S}_{\text{BM25}}(q, d), \quad \alpha \in [0, 1]"
    )
    st.markdown(
        "Alternatively, **Reciprocal Rank Fusion (RRF)** combines ranked permutations without score scale sensitivity:"
    )
    st.latex(
        r"\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k_{\text{rrf}} + r_m(d)}, \quad k_{\text{rrf}} = 60"
    )


    st.markdown("---")

    # 4. Comparative Empirical Insights
    st.markdown("### 4. Empirical Observations on Stand-Up Comedy Transcripts")

    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.markdown(
            """
            **When Dense Retrieval Dominates:**
            - Conversational metaphors, slang, and comedy premises.
            - Storytelling routines where you remember the topic but not the exact wording.
            - Robustness to transcription artifacts and informal sentence structure.
            """
        )
    with ins_col2:
        st.markdown(
            """
            **When BM25 Lexical Retrieval Is Essential:**
            - Distinctive punchlines, uncommon words (e.g., product names, places).
            - Comedian names, special titles, and specific entity mentions.
            - High-precision matching when exact phrasing is known.
            """
        )
