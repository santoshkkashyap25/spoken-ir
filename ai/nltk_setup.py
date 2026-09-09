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
    env_path = os.environ.get("NLTK_DATA")
    if env_path:
        os.makedirs(env_path, exist_ok=True)
        return env_path
    path = os.path.join(tempfile.gettempdir(), "nltk_data")
    os.makedirs(path, exist_ok=True)
    return path


_REQUIRED = {
    "stopwords": "corpora/stopwords",
}


def setup_nltk_data() -> None:
    """Ensure required NLTK packages are available; download any that are missing."""
    data_dir = _nltk_data_dir()
    os.environ["NLTK_DATA"] = data_dir
    if data_dir not in nltk.data.path:
        nltk.data.path.append(data_dir)

    for package, test_path in _REQUIRED.items():
        try:
            nltk.data.find(test_path)
            logger.info("NLTK resource found: %s", package)
        except Exception:
            logger.info("Downloading missing NLTK resource: %s", package)
            try:
                nltk.download(package, download_dir=data_dir, quiet=True)
                logger.info("Downloaded: %s", package)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to download %s: %s", package, e)


# Run setup immediately on import.
setup_nltk_data()
