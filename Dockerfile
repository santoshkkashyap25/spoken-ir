# ─────────────────────────────────────────────────────────────────────────────
# TransNLP — single shared image for the FastAPI backend + Streamlit frontend.
#
# Build once, run twice via docker-compose.yml (different CMD per service).
# Runtime corpus data (processed CSV + embeddings) are volume-mounted; only
# trained model artifacts ship inside the image.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# lxml needs libxml2/libxslt.  curl added for optional health-check debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this expensive layer
# independently of application-code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── NLP model downloads ───────────────────────────────────────────────────────
# spaCy English model
RUN python -m spacy download en_core_web_sm

# NLTK corpora (punkt, stopwords, wordnet, averaged_perceptron_tagger)
ENV NLTK_DATA=/app/nltk_data
RUN mkdir -p /app/nltk_data \
    && python - <<'EOF'
import nltk
for pkg in (
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
):
    nltk.download(pkg, download_dir="/app/nltk_data", quiet=True)
EOF

# ── Application source ────────────────────────────────────────────────────────
# Trained model artifacts (small, version-controlled) ship inside the image.
# Large runtime data (processed CSV + embeddings) are volume-mounted at runtime.
COPY data/models/        data/models/
COPY ai/                 ai/
COPY backend/            backend/
COPY pages/              pages/
COPY utils/              utils/
COPY .streamlit/         .streamlit/
COPY config.py           config.py
COPY app.py              app.py

# ── Runtime defaults ──────────────────────────────────────────────────────────
# Each service in docker-compose overrides CMD; this default starts the backend.
EXPOSE 8000 8501
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
