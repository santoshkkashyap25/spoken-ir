"""Build the corpus embeddings matrix.

Run once after preprocess_data.py. Output:
  data/ai/corpus_embeddings.npy        — (N, V) dense TF-IDF matrix
  data/ai/corpus_topic_vectors.npy    — (N, 7) LDA topic-probability matrix

Both artifacts are loaded at API startup. Regenerate whenever the corpus
CSV changes.
"""

from __future__ import annotations

import logging
import sys

# Allow running this script directly: `python scripts/build_corpus_embeddings.py`.
sys.path.insert(0, ".")

from ai.corpus import load_corpus
from ai.embed import build_corpus_embeddings, build_corpus_topic_vectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_corpus_embeddings")


def main() -> int:
    corpus = load_corpus()
    logger.info("Loaded corpus with %d rows", len(corpus))
    tfidf = build_corpus_embeddings(corpus)
    topics = build_corpus_topic_vectors(corpus)
    logger.info("Done. tfidf=%s, topics=%s", tfidf.shape, topics.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
