# TransNLP — Stand-up Similarity

TransNLP is a similarity search tool for stand-up comedy writers. Paste a draft or
upload a `.txt` file to find existing specials from a 500-transcript corpus that
most closely match your work by vocabulary and topic mix. Use the results as
reference — the app surfaces neighbors to explore, not predictions of success.

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

You need three things on disk before the app runs: a Python environment with
dependencies, a built corpus (CSV + TF-IDF matrix + LDA topic vectors), and
two running processes (backend + frontend). The first two happen once. The
third is what you do every time you want to use the app.

### 1. Set up the Python environment

```bash
# Create a venv in the project root
python -m venv .venv

# Activate it
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model (one-time, ~40 MB)
python -m spacy download en_core_web_sm
```

> **On Windows:** if PowerShell blocks `Activate.ps1` with an execution policy
> error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> once, then retry. Or skip activation entirely and call the venv's `python.exe`
> directly (see the run steps below).

### 2. Build the corpus (one time)

The corpus CSV, TF-IDF matrix, and LDA topic vectors are all gitignored —
they're large and regenerable. Build them in three stages:

```bash
# Stage 1 — Scrape transcripts from scrapsfromtheloft.com (~30 min for 500 specials)
# Resumable: safe to re-run, skips already-cached transcripts.
python scripts/scrape_data.py

# Stage 2 — Run the NLP pipeline on each transcript, write the corpus CSV
# Reads data/raw/transcripts/*.pkl, writes data/processed/processed_content_data.csv
python scripts/preprocess_data.py

# Stage 3 — Build the matrices the API loads at startup
# Writes data/ai/corpus_embeddings.npy (TF-IDF, shape 500x1000)
# and data/ai/corpus_topic_vectors.npy (LDA, shape 500x7)
python scripts/build_corpus_embeddings.py
```

After stage 3 you should have:

- `data/raw/transcripts/{0..499}.pkl` — 500 transcript files
- `data/processed/processed_content_data.csv` — the corpus
- `data/ai/corpus_embeddings.npy`
- `data/ai/corpus_topic_vectors.npy`

If you re-scrape (e.g. you add more sources), re-run stages 2 and 3 in order.

### 3. Run the app

The app is two processes: a FastAPI backend (port 8000) and a Streamlit
frontend (port 8501). Open two terminals, both with the venv activated.

**Terminal 1 — backend:**

```bash
uvicorn backend.main:app --reload --port 8000
```

On Windows with a local venv (if you didn't activate it):

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

You should see startup logs that include:

```
INFO - Loading corpus and embeddings at startup...
INFO - Startup complete: corpus=500 rows, tfidf=(500, 1000), topics=(500, 7)
```

Sanity check from any terminal:

```bash
curl http://localhost:8000/health
# → {"status":"ok","corpus_size":500}
```

OpenAPI docs: http://localhost:8000/docs

**Terminal 2 — frontend:**

```bash
streamlit run app.py
```

On Windows:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open http://localhost:8501. The home page shows a backend health indicator
— if it's green, paste a draft on the "Transcript Matcher" page and click
"Find similar specials."

### Pointing the frontend at a different backend

By default the frontend expects the backend on `http://localhost:8000`. To
point it elsewhere (a remote server, a different port, a deployed instance),
set `TRANSNLP_API_URL`:

```bash
# Linux / macOS
TRANSNLP_API_URL=https://my-api.example.com streamlit run app.py

# Windows (PowerShell)
$env:TRANSNLP_API_URL = "https://my-api.example.com"; streamlit run app.py

# Windows (cmd)
set TRANSNLP_API_URL=https://my-api.example.com && streamlit run app.py
```

### Troubleshooting

- **`RuntimeError: Startup failed: ... Run scripts/preprocess_data.py then scripts/build_corpus_embeddings.py first.`**
  You skipped one of the corpus build stages. Re-run stage 2 and stage 3 from
  step 2 above.

- **`ModuleNotFoundError: No module named 'spacy'`** (or any other dep) when
  starting the backend. The venv isn't activated in that terminal, or you're
  using the system Python instead of the venv's. Use the venv's `python.exe`
  directly as shown in the Windows examples.

- **`OSError: [E050] Can't find model 'en_core_web_sm'`** when the backend
  starts. Run `python -m spacy download en_core_web_sm` once.

- **Frontend shows "Backend unreachable" or red status.** The backend isn't
  running, it's on a different port, or `TRANSNLP_API_URL` is pointing
  somewhere wrong. Check `curl http://localhost:8000/health` from the same
  machine the browser is on.

- **NLTK download errors on Windows.** NLTK data is written to a temp dir
  (`%TEMP%\nltk_data` on Windows, `/tmp/nltk_data` on Linux). If downloads
  fail, check that the temp dir is writable.

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
