"""Text preprocessing primitives for the AI layer.

The pipeline:
  1. clean_text — lowercase, strip punctuation, keep alphabetic only
  2. spaCy lemmatization + POS-keep NOUN/ADJ/VERB/ADV (with NLTK fallback)
  3. NLTK stopword removal

These primitives are referenced by the saved TF-IDF vectorizer via
identity_tokenizer / identity_analyzer, so any change to their signatures
would break the pickled artifact. Keep the no-arg shape.
"""

from __future__ import annotations

import logging
import re
import string

from ai import nltk_setup  # noqa: F401 — bootstraps NLTK data on import

logger = logging.getLogger(__name__)

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:  # noqa: BLE001
    logger.error(
        "Error loading spaCy model: %s. "
        "Install with: python -m spacy download en_core_web_sm",
        e,
    )
    _NLP = None


# --- NLTK imports (data bootstrapped by nltk_setup) ---
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # noqa: E402

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and non-alphabetic characters."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def lemmatize_text(text: str) -> str:
    """NLTK WordNet lemmatizer over word_tokenize."""
    tokens = word_tokenize(text)
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)


def filter_pos(text: str, allowed_pos=("NOUN", "ADJ", "VERB", "ADV")) -> str:
    """Keep tokens whose NLTK POS tag starts with one of the allowed prefixes."""
    allowed = {p[0] for p in allowed_pos}
    tokens = word_tokenize(text)
    tagged = __import__("nltk").pos_tag(tokens)
    return " ".join(w for w, t in tagged if t[:1] in allowed)


def remove_stopwords(text: str) -> str:
    """Drop English stopwords."""
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in _STOPWORDS)


def preprocess(text: str) -> str:
    """Full NLP pipeline used for the corpus and queries.

    Returns a single space-joined string suitable for vectorization.
    """
    if not isinstance(text, str):
        return ""

    cleaned = clean_text(text)

    if _NLP is not None:
        doc = _NLP(cleaned)
        kept = [t.lemma_ for t in doc if t.pos_ in {"NOUN", "ADJ", "VERB", "ADV"}]
        spacy_out = " ".join(kept)
    else:
        # NLTK fallback when spaCy isn't available.
        logger.warning("Using NLTK fallback for lemmatization/POS filtering.")
        spacy_out = filter_pos(lemmatize_text(cleaned))

    return remove_stopwords(spacy_out)


# --- Identity tokenizer/analyzer ---
# These are referenced by the pickled TF-IDF vectorizer; their signatures
# and locations must remain stable. Do not change.


def identity_tokenizer(text):
    return text


def identity_analyzer(text):
    return text
