# Spoken-IR: Stand-Up Comedy Semantic Search & Hybrid Retrieval

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests](https://img.shields.io/badge/Tests-28%20Passed-brightgreen.svg?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ingestion](https://img.shields.io/badge/Ingestion-PDF%20%7C%20Markdown%20%7C%20TXT-orange.svg?style=flat-square)](https://pypi.org/project/pypdf/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

Spoken-IR is a semantic search and hybrid retrieval engine built over **582 full-length stand-up comedy specials** (scraped from [Scraps from the loft](https://scrapsfromtheloft.com/stand-up-comedy-scripts/)). It combines dense neural embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with sparse lexical matching (`BM25Okapi`) to help you find comedy routines, bits, and specials—even when you don't remember the exact punchline or comedian.

---

## Table of Contents

- [Overview](#overview)
- [The Problem: Why Searching Stand-Up Is Hard](#the-problem-why-searching-stand-up-is-hard)
- [How Search Works](#how-search-works)
  - [1. Dense Semantic Search (MiniLM-L6-v2)](#1-dense-semantic-search-minilm-l6-v2)
  - [2. Lexical Keyword Search (BM25Okapi)](#2-lexical-keyword-search-bm25okapi)
  - [3. Hybrid Convex Fusion & Reciprocal Rank Fusion](#3-hybrid-convex-fusion--reciprocal-rank-fusion)
  - [4. Out-of-Distribution (OOD) Guardrail](#4-out-of-distribution-ood-guardrail)
- [Corpus Details](#corpus-details)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Precomputing Search Indices](#precomputing-search-indices)
  - [Running the App](#running-the-app)
- [Automated Tests](#automated-tests)
- [Docker Deployment](#docker-deployment)
- [License](#license)

---

## Overview

Spoken-IR indexes 582 stand-up comedy specials from comedians including Dave Chappelle, George Carlin, Bill Burr, Ali Wong, John Mulaney, Ricky Gervais, Hannah Gadsby, and Jeff Arcuri. 

Instead of relying solely on exact keyword searches, Spoken-IR lets you query by:
- **Themes & Concepts:** e.g., *"the fear of aging and running out of time"* or *"getting caught doing something embarrassing by your parents"*.
- **Rough Quotes & Paraphrases:** e.g., *"comedian talking about buying candy at a pharmacy"*.
- **Uploaded Documents:** Drag and drop `.pdf`, `.md`, or `.txt` files to find specials with matching subject matter.

---

## The Problem: Why Searching Stand-Up Is Hard

Stand-up comedy transcripts are conversational, slang-heavy, and full of storytelling. This creates two distinct search problems:

1. **The Vocabulary Mismatch Problem:** When you recall a joke, you rarely remember the comedian's exact words. If you search for *"fear of getting old"*, a classic lexical search (like BM25 or SQL `LIKE`) will miss a routine where the comedian says *"my knees crack when I stand up and all my friends are having babies"*, because none of the search words appear in the text. Dense embeddings solve this by mapping concepts to nearby points in vector space.
2. **The Exact-Match Problem:** Conversely, if you remember a very specific, quirky punchline or name (e.g., *"Big Mouth Billy Bass"* or *"Hanalei Bay"*), a dense neural network might dilute that exact term in favor of general vibe. BM25 excels here because it strictly rewards exact token occurrences.

**Hybrid Search** gives you the best of both worlds: dense embeddings find the thematic neighborhood, while BM25 boosts specials containing the exact keywords you remembered.

---

## How Search Works

### 1. Dense Semantic Search (MiniLM-L6-v2)

Each transcript $d$ and query $q$ is encoded into a 384-dimensional continuous vector using `sentence-transformers/all-MiniLM-L6-v2`:

$$\mathbf{e}_q = \mathcal{E}(q), \quad \mathbf{e}_d = \mathcal{E}(d)$$

Because all vectors are unit-normalized ($\|\mathbf{e}\|_2 = 1$), cosine similarity reduces to a fast dot product:

$$S_{\text{dense}}(q, d) = \mathbf{e}_q \cdot \mathbf{e}_d$$

This runs against the precomputed corpus matrix $\mathbf{M}_{\text{dense}} \in \mathbb{R}^{582 \times 384}$ using optimized NumPy BLAS matrix-vector operations.

### 2. Lexical Keyword Search (BM25Okapi)

For keyword matching, transcripts are lemmatized with spaCy and indexed using BM25Okapi:

$$\text{BM25}(q, d) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

- $f(t, d)$: token frequency in document $d$.
- $|d|$ and $\text{avgdl}$: document length and average corpus length.
- $k_1 = 1.5$: term frequency saturation parameter.
- $b = 0.75$: document length penalty.
- $\text{IDF}(t) = \ln \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)$ with Robertson-Spärck Jones smoothing.

### 3. Hybrid Convex Fusion & Reciprocal Rank Fusion

Dense cosine scores reside in $[-1, 1]$ while BM25 scores are unbounded $[0, \infty)$. Spoken-IR normalizes both to $[0, 1]$ via min-max scaling before blending:

$$\bar{S}(d) = \frac{S(d) - \min S}{\max S - \min S + \epsilon}$$

**Convex Score Combination:**
$$S_{\text{hybrid}}(q, d) = \alpha \cdot \bar{S}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \bar{S}_{\text{BM25}}(q, d)$$

- $\alpha = 1.0$: 100% Dense semantic search.
- $\alpha = 0.0$: 100% Lexical BM25 search.
- $\alpha = 0.6$ (default): Balanced hybrid (60% semantic, 40% keyword precision).

**Reciprocal Rank Fusion (RRF):**
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + r_m(d)}$$
Combines the ranked lists directly by ordinal position without needing score normalization.

### 4. Out-of-Distribution (OOD) Guardrail

Because the index contains exclusively stand-up comedy, queries about unrelated topics (e.g., corporate financial filings or medical textbooks) will produce misleading "top" results. Spoken-IR flags this with an Out-of-Distribution metric based on maximum semantic alignment:

$$\text{OOD}(q) = 1.0 - \max_{d \in \mathcal{D}} S_{\text{dense}}(q, d)$$

- **Strong Match:** $\text{OOD} \le 0.40$ (query fits comedy material well)
- **Moderate Match:** $0.40 < \text{OOD} \le 0.65$
- **Weak / Out of Distribution:** $\text{OOD} > 0.65$ (query likely has nothing to do with stand-up comedy)

---

## Corpus Details

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total Specials ($N$)** | `582` | Full stand-up comedy transcripts |
| **Source** | Scraps From The Loft | Stand-up comedy scripts archive |
| **Dense Matrix** | `(582, 384)` | Precomputed float32 embeddings (`all-MiniLM-L6-v2`) |
| **BM25 Inverted Index** | `582 docs` | Tokenized & lemmatized vocabulary index |
| **TF-IDF Matrix** | `(582, 1000)` | Bigram-aware baseline representation |
| **Preprocessing** | spaCy + NLTK | POS filtering (NOUN, ADJ, VERB, ADV) and stopword removal |
| **File Upload Support** | `.pdf`, `.md`, `.txt` | Multi-format parsing using `pypdf` |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/santoshkkashyap25/spoken-ir.git
cd spoken-ir


# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Precomputing Search Indices

Before running searches, precompute the dense vectors and BM25 index from the raw corpus:

```bash
python scripts/build_corpus_embeddings.py
```

### Running the App

Start the backend API and frontend interface in separate terminals:

**1. FastAPI Backend (Port 8000):**
```bash
uvicorn backend.main:app --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**2. Streamlit Web UI (Port 8501):**
```bash
streamlit run app.py
```
Open `http://localhost:8501` to search routines, adjust hybrid weights, or upload files.

---

## Automated Tests

Run the test suite with pytest:

```bash
pytest tests/ -v
```

---

## Docker Deployment

Build and run with Docker Compose:

```bash
docker compose up --build
```

Access the Streamlit UI on `http://localhost:8501`. It connects to the backend API running on `http://localhost:8000`.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
