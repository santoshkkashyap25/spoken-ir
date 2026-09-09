"""Build the corpus retrieval matrices and indices.

Generates:
  data/ai/corpus_dense_embeddings.npy  — (N, 384) sentence-transformers matrix
  data/ai/corpus_bm25.pkl              — BM25Okapi lexical index
  data/ai/corpus_embeddings.npy        — (N, V) dense TF-IDF matrix

These artifacts are loaded at API startup to provide real-time hybrid retrieval.
Regenerate whenever the corpus CSV changes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path as _Path

# Ensure project root is importable when running directly (no pip install -e)
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.corpus import load_corpus
from ai.embed import (
    build_corpus_bm25_index,
    build_corpus_dense_embeddings,
    build_corpus_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_corpus_embeddings")


def main() -> int:
    corpus = load_corpus()
    logger.info("Loaded corpus with %d rows", len(corpus))

    logger.info("Step 1/3: Building dense bi-encoder embeddings (all-MiniLM-L6-v2)...")
    dense = build_corpus_dense_embeddings(corpus)

    logger.info("Step 2/3: Building BM25 lexical index...")
    bm25 = build_corpus_bm25_index(corpus)

    logger.info("Step 3/3: Building TF-IDF matrix...")
    tfidf = build_corpus_embeddings(corpus)

    logger.info(
        "Corpus retrieval build complete: dense=%s, bm25_docs=%d, tfidf=%s",
        dense.shape,
        len(corpus),
        tfidf.shape,
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
