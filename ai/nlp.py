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

import nltk
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

_STOPWORDS = set(stopwords.words("english"))

# Conversational disfluencies, spoken artifacts, profanity, and generic noise tokens
_PROFANITY = {
    # Profanity & crude language
    "fuck", "fucking", "fuckin", "fucker", "fucks", "fck", "fcking", "fcke", "fckin",
    "motherfucker", "motherfucking", "motherfcker", "motherfckin",
    "shit", "bullshit",
    "bitch", "bitches", "btch",
    "ass", "asshole", "assholes",
    "dick", "dicks", "cock", "pussy", "cunt", "tit", "tits", "penis", "vagina",
    "goddamn", "damn",
    "nigga", "niggas", "nigger", "ngga",
    
    # Anatomical / Crude
    "ball", "balls", "fart", "farts", "blow", "suck",
    
    # Filler / Slang / Disfluencies
    "wan", "na", "gonna", "gotta", "lemme", "gimme", "yeah", "yes", "no", "oh", "uh", "um", "like",
    "dude", "bro", "mate", "mum", "bloke", "guy", "man", "sort", "quite", "g", "n", "l", "e", "c",
    
    # Transcript Artifacts
    "laughter", "applause", "cheer", "chuckle", "applaud", "voice", "music", "playing",
    
    # Foreign leaks
    "perch", "sono", "essere", "sapete", "bambini", "di", "era", "alla", "fare", "quando",

    # Ultra-Common Family Terms (blurring topics)
    "mom", "wife", "mother", "son", "parent", "child", "brother", "husband", "boyfriend", "daughter", "daddy", "mama",

    # Spoken performance & presentation noise
    "comedy", "comedian", "standup", "movie", "film", "audience", "applauding", "write", "song", "picture",

    # Abstract Verbs & General Noise
    "course", "moment", "fact", "true", "realize", "due", "anymore", "high", "mad", "lovely", "speak", "learn", "stick", "sex", "gay",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/non-alphabetic characters, and normalize whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Drop English stopwords, disfluencies, and noise tokens."""
    tokens = text.split()
    return " ".join(t for t in tokens if t not in _STOPWORDS and t not in _PROFANITY)


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
        # Fallback when spaCy is unavailable
        logger.warning("spaCy model unavailable; using basic whitespace tokenization fallback.")
        spacy_out = cleaned

    return remove_stopwords(spacy_out)



def preprocess_batch(texts: list[str]) -> list[str]:
    """Batch version of preprocess that uses spaCy's nlp.pipe() for speed."""
    if _NLP is None:
        # Fallback to serial processing
        return [preprocess(t) for t in texts]

    cleaned_texts = [clean_text(t) if isinstance(t, str) else "" for t in texts]
    
    # Run nlp.pipe on the batch
    docs = _NLP.pipe(cleaned_texts, batch_size=50)
    
    out = []
    for doc in docs:
        kept = [t.lemma_ for t in doc if t.pos_ in {"NOUN", "ADJ", "VERB", "ADV"}]
        spacy_out = " ".join(kept)
        out.append(remove_stopwords(spacy_out))
        
    return out


# --- Identity tokenizer/analyzer ---
# These are referenced by the pickled TF-IDF vectorizer; their signatures
# and locations must remain stable. Do not change.


def identity_tokenizer(text):
    return text


def identity_analyzer(text):
    return text
