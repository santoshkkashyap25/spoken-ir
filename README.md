# TransNLP — Stand-up Similarity

An honest NLP tool for stand-up comedy writers. Paste a draft (or upload a `.txt`),
and TransNLP shows you the existing specials in our 500-special corpus that
most resemble it — by vocabulary and topic mix. **It does not predict success.**
It surfaces neighbors so you can read them and decide for yourself.

[Live app](https://transnlp.streamlit.app/) (still running the old version —
the refactor is in development).

## What it does

- **Transcript Matcher** — paste a draft, get the top-k specials it most
  resembles, with cosine similarity scores, IMDb rating, year, and a
  side-by-side topic-mix bar chart. A "match strength" banner tells you
  how well the draft fits the corpus.
- **Topic Explorer** — browse the 7 topics our LDA model learned from the
  corpus. See each topic's top words, corpus share, average rating when
  dominant, and the specials that carry it.

## What it does NOT do

- Predict whether your script will be a hit.
- Forecast ratings.
- Replace reading the work yourself.

The earlier version of this project framed the output as a success predictor.
That framing over-promised. This version is a similarity search — useful,
defensible, and honest about its limits.

## Architecture

```
transnlp/
├── ai/                  # Pure ML/NLP. No web, no Streamlit.
│   ├── corpus.py        # Load processed_content_data.csv
│   ├── embed.py         # TF-IDF corpus matrix + query embedding
│   ├── similarity.py    # Cosine top-k + OOD score
│   ├── topics.py        # Topic labels, top words, distribution
│   ├── nlp.py           # Clean → spaCy → stopwords
│   └── nltk_setup.py    # NLTK resource bootstrap
│
├── backend/             # FastAPI service (port 8000)
│   ├── main.py          # App + lifespan + CORS
│   ├── api/routes.py    # /health, /match, /topics, /specials
│   ├── schemas.py       # Pydantic models
│   └── service.py       # Orchestration over ai/
│
├── pages/               # Streamlit multi-page UI (port 8501)
│   ├── 1_Transcript_Matcher.py
│   └── 2_Topic_Explorer.py
│
├── scripts/             # Offline pipeline
│   ├── scrape_data.py            # Scrape transcripts (resumable)
│   ├── preprocess_data.py        # NLP pipeline → corpus CSV
│   └── build_corpus_embeddings.py # Build TF-IDF matrix for the corpus
│
├── tests/               # pytest
│   ├── test_similarity.py
│   ├── test_api.py
│   └── test_pipeline_idempotency.py
│
├── data/
│   ├── raw/             # Scraped transcripts (.pkl per special)
│   ├── processed/       # processed_content_data.csv (corpus)
│   ├── ai/              # Generated: corpus_embeddings.npy
│   └── models/          # Trained LDA + TF-IDF artifacts
│
├── app.py               # Streamlit entry point
├── config.py            # Project-level paths and constants
└── requirements.txt
```

The three layers are explicit: `ai/` is pure Python and has no web imports,
`backend/` orchestrates the AI layer over HTTP, and the Streamlit pages are
thin HTTP clients. Models load once in the FastAPI lifespan; the Streamlit
app never imports them directly.

## Getting started

### Prerequisites

```bash
python -m venv .venv
# On Linux/Mac:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Build the corpus (one time)

The `data/processed/processed_content_data.csv` and `data/ai/corpus_embeddings.npy`
files are gitignored. Build them locally:

```bash
# 1. Scrape (resumable; safe to re-run; ~30 min for 500 specials)
python scripts/scrape_data.py

# 2. Preprocess — runs the NLP pipeline and writes the corpus CSV
python scripts/preprocess_data.py

# 3. Build the TF-IDF matrix AND per-document topic vectors for similarity search
python scripts/build_corpus_embeddings.py
```

This step writes two artifacts:

- `data/ai/corpus_embeddings.npy` — (500, 1000) TF-IDF matrix for similarity
- `data/ai/corpus_topic_vectors.npy` — (500, 7) LDA topic probabilities per document

### Run the app

Two processes. Open two terminals.

```bash
# Terminal 1 — backend (port 8000)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend (port 8501)
streamlit run app.py
```

Open http://localhost:8501. The home page will show whether the backend is
reachable. To point the frontend at a different backend, set `TRANSNLP_API_URL`:

```bash
TRANSNLP_API_URL=http://my-host:8000 streamlit run app.py
```

On Windows with a local venv, use the venv's executables directly:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
.venv\Scripts\python.exe -m streamlit run app.py
```

### Run the tests

```bash
pytest tests/ -v
```

Tests that require the corpus on disk are auto-skipped with a clear message
when the artifacts aren't present.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status": "ok", "corpus_size": N}` |
| `/match` | POST | Body: `{"text": str, "k": int}`. Returns top-k similar specials + OOD score. |
| `/topics` | GET | All 7 topics with top words, special count, avg rating, corpus share. |
| `/specials?topic=X&limit=20` | GET | Specials where `topic` dominates, ordered by topic weight. |

OpenAPI docs at http://localhost:8000/docs when the backend is running.

## How similarity is computed

For each special in the corpus we precompute a TF-IDF vector using the
trained `tfidf_vectorizer.pkl` (max 1000 features, bigram-aware, identity
tokenizer/analyzer). When a query comes in, we run the same NLP pipeline
(clean → spaCy lemmatize → keep NOUN/ADJ/VERB/ADV → NLTK stopwords) and
vectorize with the same vocabulary. Cosine similarity between the query
vector and every corpus row gives the top-k neighbors.

The OOD score is the mean of the top-5 cosine similarities. It's a single
number that says "how well does this draft fit the corpus at all?" — used
to surface the "doesn't closely resemble our corpus" warning when appropriate.

## Honest limits

- The corpus is 500 specials from scrapsfromtheloft.com. It's English-language
  stand-up, biased toward American/British specials that have already circulated.
- LDA + TF-IDF on 500 documents is a real representation, not a deep one.
  Similarity scores should be read as "thematic/lexical neighborhood," not
  "this is the same kind of work."
- No novelty detection beyond cosine distance from the corpus — being unlike
  existing work is read as "no match," not "opportunity."

## License

Personal project. Scraped transcripts belong to their original publishers.
