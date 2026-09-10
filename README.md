# Spoken-IR: Stand-Up Comedy Semantic Search & Hybrid Retrieval

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg?style=flat-square)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

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

Stand-up comedy is conversational, slang-heavy, and narrative-driven, creating two core search challenges:

- **Vocabulary Mismatch:** Users recall concepts, not verbatim jokes (e.g., searching *"fear of aging"* misses bits about *"creaky knees and buying fiber supplements"*). Dense embeddings solve this via conceptual vector proximity.
- **Exact Punchline Dilution:** Rare names and punchlines (e.g., *"Big Mouth Billy Bass"*) can be diluted by pure semantic embeddings. BM25 guarantees exact keyword precision.
- **Hybrid Solution:** Blends dense semantic intent with BM25 keyword scoring to deliver both conceptual discovery and exact punchline accuracy.

---

## How Search Works

### 1. Dense Semantic Search (MiniLM-L6-v2)
- **What it does:** Converts your query and all transcripts into numerical embeddings that capture **meaning** rather than just words.
- **Why it matters:** Finds routines based on general ideas, story premises, and themes. Searching for *"nervous about getting married"* finds wedding anxiety jokes even if the word "nervous" was never said.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).

### 2. Lexical Keyword Search (BM25Okapi)
- **What it does:** Searches for exact token and phrase matches across lemmatized transcripts.
- **Why it matters:** BM25 rewards rare, distinctive words (like a comedian's name, a unique punchline, or a specific prop) while ignoring common conversational filler.
- **Implementation:** `rank-bm25` index over pre-tokenized transcripts.

### 3. Hybrid Convex Fusion & Reciprocal Rank Fusion
- **What it does:** Combines semantic scores and keyword scores into one balanced ranking.
- **Tunable weight ($\alpha$):**
  - **`1.0`**: 100% Dense semantic search (pure concept matching).
  - **`0.0`**: 100% Lexical BM25 search (strict keyword matching).
  - **`0.6` (Default)**: Balanced hybrid (60% semantic discovery + 40% keyword precision).
- **Reciprocal Rank Fusion (RRF):** Also supports rank-position consensus without needing score normalization.

### 4. Out-of-Distribution (OOD) Guardrail
- **What it does:** Checks whether your query actually belongs in a stand-up comedy database.
- **Why it matters:** If you paste something unrelated (like legal contracts or quantum physics), the engine alerts you instead of giving false confidence:
  - **Strong Match:** Well-aligned with comedy topics.
  - **Moderate Match:** Partial overlap with comedy themes.
  - **Weak / Out of Distribution:** The query has little to nothing to do with stand-up comedy.

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
