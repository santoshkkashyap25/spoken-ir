"""TransNLP — Domain-Specific Semantic Search & Information Retrieval Benchmark.

A unified, educational, and empirical platform benchmarking:
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
    page_title="TransNLP — Information Retrieval Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 TransNLP: Spoken-Word Information Retrieval Benchmark")
st.caption(
    "An empirical platform comparing Dense Bi-Encoders, BM25 Lexical Ranking, "
    "and Hybrid Convex Fusion across 580+ spoken monologue transcripts."
)

base = api_base_url()
backend_alive = check_backend_health(base)

# ── Header Health Status ───────────────────────────────────────────────────────
if not backend_alive:
    st.error(
        f"⚠️ **Backend Service Offline** (`{base}`). "
        "Launch the FastAPI backend service via:\n\n"
        "```bash\n.venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000\n```"
    )

# ── Core Benchmark Tabs ────────────────────────────────────────────────────────
tab_search, tab_theory = st.tabs(
    [
        "🔍 Retrieval Benchmark & Query Engine",
        "📐 Empirical Methodology & Formulations",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RETRIEVAL BENCHMARK & QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("Query Execution & Multi-Strategy Retrieval")
    st.markdown(
        "Evaluate retrieval efficacy on spoken text by comparing dense semantic vectors against "
        "exact BM25 inverted-index matching and tunable hybrid convex combinations."
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
                "e.g., 'Technological progress in artificial intelligence is reshaping institutional trust. "
                "Conversations around algorithmic governance and corporate accountability highlight "
                "how public discourse struggles to keep pace with innovation.'"
            ),
        )
    with q_col2:
        uploaded_file = st.file_uploader("Or upload text file (.txt)", type=["txt"])

    query_content = ""
    if uploaded_file is not None:
        try:
            query_content = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded {len(query_content):,} characters from `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    elif input_text:
        query_content = input_text

    if st.button("Execute Retrieval Benchmark", type="primary"):
        if not query_content or len(query_content.strip()) < 20:
            st.warning("Please provide a query passage of at least 20 characters.")
        elif not backend_alive:
            st.error("Backend API is unreachable. Ensure the FastAPI service is active on port 8000.")
        else:
            with st.spinner(f"Computing {mode_key.upper()} scores across 582 corpus documents..."):
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
        "This project investigates the trade-offs between **dense semantic representations**, "
        "**sparse probabilistic keyword scoring**, and **probabilistic topic distributions** "
        "when applied to unstructured spoken-word corpora."
    )

    st.markdown("---")

    # 1. Dense Bi-Encoder
    st.markdown("### 1. Dense Semantic Retrieval (Bi-Encoder)")
    st.markdown(
        "Given a query passage $q$ and a document $d$, a pre-trained transformer encoder "
        "$\mathcal{E}_{\phi}(\cdot)$ maps each text sequence to a fixed-dimensional continuous embedding vector "
        "$\mathbf{e} \in \mathbb{R}^{D}$ (with $D=384$ using `all-MiniLM-L6-v2`):"
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
        "BM25 establishes relevance based on non-linear term frequency saturation and document length normalization. "
        "For query tokens $t \in q$ and document $d$:"
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
        "Individual raw scores are normalized into the interval $[0, 1]$ via min-max scaling $\bar{S}$. "
        "Convex score combination interpolates between contextual abstraction and lexical exactness:"
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
    st.markdown("### 4. Empirical Observations on Spoken-Word Text")

    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.markdown(
            """
            **When Dense Retrieval Dominates:**
            - Conversational metaphors, idioms, and colloquial phrasing.
            - Thematic cross-domain matching where shared concept vocabulary diverges.
            - Robustness to transcription noise and minor grammatical variance.
            """
        )
    with ins_col2:
        st.markdown(
            """
            **When BM25 Lexical Retrieval Is Essential:**
            - Proper nouns, personal names, city references, and specific event tags.
            - Niche jargon that pre-trained bi-encoders may dilute into generic semantic neighborhoods.
            - High-precision filtering against vocabulary hallucination.
            """
        )
