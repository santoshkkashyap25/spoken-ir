"""NLTK resource bootstrap.

Downloads required NLTK data to a temp directory on first import. Works on
Linux (Streamlit Cloud) and Windows (local dev) without a hardcoded /tmp path.
"""

import logging
import os
import tempfile

import nltk

logger = logging.getLogger(__name__)


def _nltk_data_dir() -> str:
    """Return a writable NLTK data directory for the current platform."""
    path = os.path.join(tempfile.gettempdir(), "nltk_data")
    os.makedirs(path, exist_ok=True)
    return path


_REQUIRED = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def setup_nltk_data() -> None:
    """Ensure required NLTK packages are available; download any that are missing."""
    data_dir = _nltk_data_dir()
    os.environ["NLTK_DATA"] = data_dir
    if data_dir not in nltk.data.path:
        nltk.data.path.append(data_dir)
    logger.info("NLTK_DATA set to: %s", data_dir)

    for package, test_path in _REQUIRED.items():
        try:
            nltk.data.find(test_path)
            logger.info("NLTK resource found: %s", package)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", package)
            try:
                nltk.download(package, download_dir=data_dir, quiet=True)
                logger.info("Downloaded: %s", package)
            except Exception as e:  # noqa: BLE001 — NLTK raises broad exceptions
                logger.error("Failed to download %s: %s", package, e)


# Run setup immediately on import.
setup_nltk_data()
