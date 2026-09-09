# TransNLP: Empirical Benchmarking of Dense, Sparse, and Hybrid Retrieval on Spoken Monologue Corpora

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

**TransNLP** is an open-source Information Retrieval (IR) research testbed and educational platform. It provides an empirical environment to benchmark **Dense Bi-Encoder Semantic Embeddings**, **Probabilistic Lexical Ranking (BM25Okapi)**, **Convex Score Fusion**, and **Unsupervised Topic Modeling (Latent Dirichlet Allocation)** over a corpus of 582 long-form spoken-word monologue transcripts.

---

## Abstract

Spoken monologue transcripts present fundamental challenges to classical information retrieval models. Unlike formal written text, spoken language is characterized by colloquialisms, pervasive figurative language, rhetorical repetition, and transcription noise. 

Pure dense bi-encoder models capture abstract conceptual intent but exhibit semantic drift when queries demand exact named-entity or proper-noun precision. Conversely, classical sparse lexical models (such as BM25) maintain exact term precision but fail when conversational paraphrasing diverges from the corpus vocabulary.

TransNLP provides a reproducible testbed to analyze these trade-offs by evaluating:
1. **Dense Bi-Encoder Contextual Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`, $D=384$)
2. **Sparse Probabilistic Inverted-Index Scoring** (`BM25Okapi`)
3. **Parametric Hybrid Score Fusion** ($S_{\text{hybrid}} = \alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{sparse}}$) and **Reciprocal Rank Fusion (RRF)**
4. **Unsupervised Latent Dirichlet Allocation (LDA)** ($K=15$ latent topic distributions)

---

## Theoretical Formulations

### 1. Dense Bi-Encoder Retrieval

Let $q$ denote an arbitrary input query passage and $d \in \mathcal{D}$ denote a document within corpus $\mathcal{D}$. A transformer encoder $\mathcal{E}_{\phi}(\cdot)$ maps text sequences into fixed-dimensional continuous vector representations $\mathbf{e} \in \mathbb{R}^{D}$ ($D=384$):

$$\mathbf{e}_q = \mathcal{E}_{\phi}(q), \quad \mathbf{e}_d = \mathcal{E}_{\phi}(d)$$

Cosine similarity measures the angular displacement between query and document vectors:

$$S_{\text{dense}}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2}$$

When vectors are unit-normalized ($\|\mathbf{e}\|_2 = 1$), similarity simplifies to the Euclidean inner product $S_{\text{dense}}(q, d) = \mathbf{e}_q \cdot \mathbf{e}_d$, enabling vectorized matrix multiplication over the precomputed corpus matrix $\mathbf{M}_{\text{dense}} \in \mathbb{R}^{N \times D}$.

### 2. Sparse Lexical Retrieval (BM25Okapi)

Relevance scoring under the probabilistic BM25 framework incorporates term frequency saturation and document length penalization:

$$\text{BM25}(q, d) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where:
- $f(t, d)$ is the frequency of token $t$ in document $d$.
- $|d|$ is the token length of document $d$, and $\text{avgdl}$ is the average document length across $\mathcal{D}$.
- $k_1 = 1.5$ regulates term frequency saturation non-linearity.
- $b = 0.75$ controls document length normalization scaling.

Inverse Document Frequency (IDF) is calculated with Robertson-Spärck Jones smoothing:

$$\text{IDF}(t) = \ln \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)$$

where $N = |\mathcal{D}|$ and $n(t)$ is the count of documents containing token $t$.

### 3. Convex Score Combination & Reciprocal Rank Fusion

To resolve differing scale distributions between dense cosine similarity ($[-1, 1]$) and BM25 scores ($[0, \infty)$), raw scores are normalized into the unit interval $[0, 1]$ via min-max normalization:

$$\bar{S}(d) = \frac{S(d) - \min_{d'} S(d')}{\max_{d'} S(d') - \min_{d'} S(d') + \epsilon}$$

**Convex Score Fusion:**
$$S_{\text{hybrid}}(q, d) = \alpha \cdot \bar{S}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \bar{S}_{\text{BM25}}(q, d), \quad \alpha \in [0, 1]$$

- When $\alpha = 1.0$, retrieval evaluates purely dense semantic similarity.
- When $\alpha = 0.0$, retrieval executes standard sparse lexical matching.
- Intermediate values ($0 < \alpha < 1$) establish a Pareto-optimal trade-off between conceptual recall and keyword precision.

**Reciprocal Rank Fusion (RRF):**
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k_{\text{rrf}} + r_m(d)}$$

where $r_m(d)$ is the ordinal rank of document $d$ under model $m$, and $k_{\text{rrf}} = 60$ is a smoothing constant mitigating outlier sensitivity.

### 4. Latent Dirichlet Allocation (LDA)

Topic distributions are modeled via a 15-topic LDA generative process. Each document $d$ is characterized by a categorical distribution over $K=15$ topics sampled from a Dirichlet prior $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$:

$$p(\mathbf{w}_d \mid \boldsymbol{\alpha}, \boldsymbol{\beta}) = \int p(\boldsymbol{\theta}_d \mid \boldsymbol{\alpha}) \left( \prod_{n=1}^{N_d} \sum_{z_{dn}} p(z_{dn} \mid \boldsymbol{\theta}_d) p(w_{dn} \mid z_{dn}, \boldsymbol{\beta}) \right) d\boldsymbol{\theta}_d$$

The resulting vector $\boldsymbol{\theta}_d \in \Delta^{14}$ represents document allocation across the latent thematic simplex.

---

## Dataset Profile & Corpus Geometry

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Document Count ($N$)** | `582` | Full-length monologue and conversational transcripts |
| **Dense Matrix Geometry** | `(582, 384)` | Unit-normalized `all-MiniLM-L6-v2` dense embeddings |
| **Sparse Matrix Geometry** | `(582, 1000)` | Bigram-aware TF-IDF baseline representation |
| **Topic Simplex Geometry** | `(582, 15)` | Unsupervised Latent Dirichlet Allocation posterior mixtures |
| **BM25 Inverted Index** | `582 docs` | Inverted token postings list over lemmatized content |
| **Linguistic Preprocessing** | spaCy + NLTK | POS filtering (NOUN, ADJ, VERB, ADV) + stopword removal |

---

## System Architecture

The codebase enforces strict modular separation into three decoupled tiers:

```
transnlp/
├── ai/                              # Pure Algorithmic/ML Layer (No Web/HTTP dependencies)
│   ├── corpus.py                    # Corpus loader & tabular accessors
│   ├── embed.py                     # Bi-encoder encoders, BM25 indices & vectorizers
│   ├── similarity.py                # Dense cosine, BM25, hybrid fusion, RRF & OOD math
│   ├── topics.py                    # 15-topic LDA inference & distribution accessors
│   ├── nlp.py                       # spaCy lemmatization + POS tagging + text cleaning
│   └── nltk_setup.py                # Isolated NLTK resource bootstrapper
│
├── backend/                         # Asynchronous High-Throughput REST Service (port 8000)
│   ├── main.py                      # FastAPI lifespan preloading & CORS middleware
│   ├── api/routes.py                # Endpoints: /health, /match, /topics, /specials
│   ├── schemas.py                   # Pydantic v2 data transfer objects & validation
│   └── service.py                   # Business logic orchestrating the ai/ layer
│
├── app.py                           # Consolidated Single-Entrypoint Streamlit Platform (port 8501)
│                                    # Tab 1: Retrieval Benchmark & Query Engine
│                                    # Tab 2: Latent Topic Topology (LDA)
│                                    # Tab 3: Empirical Methodology & Formulations
│
├── scripts/                         # Reproducible Data Engineering Pipeline
│   ├── scrape_data.py               # Resumable scraper with JSON caching
│   ├── preprocess_data.py           # NLP normalization pipeline -> corpus CSV
│   └── build_corpus_embeddings.py   # Precomputes dense, BM25, TF-IDF & LDA indices
│
├── tests/                           # Comprehensive Automated Test Suite (pytest)
│   ├── test_similarity.py           # Vector math, normalization, and hybrid fusion tests
│   ├── test_api.py                  # Integration tests for FastAPI endpoints
│   ├── test_topics.py               # LDA topic assertions
│   ├── test_nlp.py                  # Preprocessing, lemmatization & token cleaning tests
│   ├── test_pipeline_idempotency.py # Pipeline determinism test
│   └── test_scraper.py              # Parsing & metadata extraction tests
│
├── data/                            # Persistent Storage (gitignored)
│   ├── raw/                         # Raw transcript files
│   ├── processed/                   # processed_content_data.csv
│   ├── ai/                          # Serialized numpy matrices and BM25 pickles
│   └── models/                      # Pickled LDA and TF-IDF artifacts
│
├── config.py                        # Centralized paths and environment configuration
├── Dockerfile                       # Multi-stage production container build
├── docker-compose.yml               # Service orchestration definition
└── requirements.txt                 # Frozen dependency manifest
```

---

## Experimental Reproduction

### 1. Environment Initialization

```bash
# Clone the repository
git clone https://github.com/your-username/transnlp.git
cd transnlp

# Create and activate Python virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy linguistic model
python -m spacy download en_core_web_sm
```

### 2. Precomputing Retrieval Matrices & Indices

To guarantee sub-second execution at inference time, all corpus indices are precomputed and serialized to disk:

```bash
python scripts/build_corpus_embeddings.py
```

*Build Log:*
```
INFO - Step 1/4: Building dense bi-encoder embeddings (all-MiniLM-L6-v2)...
INFO - Saved dense corpus embeddings: data/ai/corpus_dense_embeddings.npy shape=(582, 384)
INFO - Step 2/4: Building BM25 lexical index...
INFO - Saved BM25 index to data/ai/corpus_bm25.pkl (documents=582)
INFO - Step 3/4: Building TF-IDF matrix...
INFO - Saved corpus TF-IDF embeddings: data/ai/corpus_embeddings.npy shape=(582, 1000)
INFO - Step 4/4: Building LDA topic distribution vectors...
INFO - Saved corpus topic vectors: data/ai/corpus_topic_vectors.npy shape=(582, 15)
INFO - Corpus retrieval build complete: dense=(582, 384), bm25_docs=582, tfidf=(582, 1000), topics=(582, 15)
```

### 3. Executing the Services

Start the backend REST API and Streamlit interface in separate processes:

**Backend Service (FastAPI — Port 8000):**
```bash
uvicorn backend.main:app --port 8000
```
- Health Check: `http://localhost:8000/health`
- OpenAPI Specification: `http://localhost:8000/docs`

**Frontend Benchmark Interface (Streamlit — Port 8501):**
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access the unified 3-tab benchmark interface.

---

## REST API Specification

### `POST /match`
Executes multi-strategy retrieval over the indexed corpus.

**Request Schema:**
```json
{
  "text": "Institutional governance and the role of technological regulation in modern media.",
  "k": 5,
  "search_mode": "hybrid",
  "alpha": 0.65
}
```

| Field | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | `string` | *required* | $\text{len} \ge 10$ | Query passage to match against corpus. |
| `k` | `integer` | `5` | $1 \le k \le 20$ | Retrieval depth (top-$k$ documents). |
| `search_mode` | `string` | `"hybrid"` | `hybrid` \| `dense` \| `sparse` | Active retrieval strategy. |
| `alpha` | `float` | `0.6` | $0.0 \le \alpha \le 1.0$ | Dense semantic weighting parameter. |

**Response Schema:**
```json
{
  "matches": [
    {
      "index": 42,
      "title": "Document Title",
      "names": "Speaker / Creator Name",
      "year": 2021,
      "rating": 8.2,
      "similarity": 0.7421,
      "dense_score": 0.7812,
      "sparse_score": 0.6695,
      "topic_mix": {
        "Law, Crime & Society": 0.42,
        "Politics, Police & Religion": 0.31
      }
    }
  ],
  "search_mode": "hybrid",
  "alpha": 0.65,
  "ood_score": 0.7105,
  "match_strength": "strong",
  "corpus_size": 582
}
```

---

## Automated Test Suite

The test suite validates theoretical correctness, boundary edge cases, and API contracts:

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health PASSED
tests/test_api.py::test_match_returns_top_k_with_schema PASSED
tests/test_api.py::test_match_dense_and_sparse_modes PASSED
tests/test_api.py::test_match_rejects_too_short_text PASSED
tests/test_api.py::test_match_handles_empty_query_gracefully PASSED
tests/test_api.py::test_topics_endpoint PASSED
tests/test_api.py::test_specials_endpoint PASSED
tests/test_api.py::test_specials_unknown_topic_returns_empty PASSED
tests/test_nlp.py::test_clean_text PASSED
tests/test_nlp.py::test_remove_stopwords PASSED
tests/test_nlp.py::test_preprocess PASSED
tests/test_nlp.py::test_preprocess_batch PASSED
tests/test_pipeline_idempotency.py::test_preprocess_is_deterministic PASSED
tests/test_pipeline_idempotency.py::test_preprocess_handles_empty_and_non_string PASSED
tests/test_pipeline_idempotency.py::test_preprocess_strips_punctuation_and_lowercases PASSED
tests/test_scraper.py::test_combine_text PASSED
tests/test_scraper.py::test_extract_name PASSED
tests/test_scraper.py::test_extract_title PASSED
tests/test_similarity.py::test_cosine_top_k_returns_k_items PASSED
tests/test_similarity.py::test_cosine_top_k_picks_closest_first PASSED
tests/test_similarity.py::test_cosine_top_k_handles_1d_query PASSED
tests/test_similarity.py::test_cosine_top_k_zero_query_returns_zeros PASSED
tests/test_similarity.py::test_cosine_top_k_ties_preserve_count PASSED
tests/test_similarity.py::test_cosine_top_k_k_larger_than_corpus PASSED
tests/test_similarity.py::test_cosine_top_k_empty_corpus PASSED
tests/test_similarity.py::test_ood_score_in_distribution PASSED
tests/test_similarity.py::test_ood_score_out_of_distribution PASSED
tests/test_similarity.py::test_ood_score_zero_query PASSED
tests/test_similarity.py::test_match_strength_label_thresholds PASSED
tests/test_similarity.py::test_dense_top_k PASSED
tests/test_similarity.py::test_hybrid_top_k_convex_combination PASSED
tests/test_similarity.py::test_reciprocal_rank_fusion PASSED
tests/test_topics.py::test_avg_rating_for_topic PASSED
tests/test_topics.py::test_avg_rating_empty_corpus PASSED
tests/test_topics.py::test_avg_rating_missing_rating_col PASSED

======================= 34 passed, 1 warning in 17.5s ========================
```

---

## Containerized Deployment

Execute both services via Docker Compose:

```bash
docker compose up --build
```
The benchmark interface will be exposed on port `8501`, connecting internally to the FastAPI container on port `8000`.
