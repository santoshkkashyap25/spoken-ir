"""Train a new LDA model on the preprocessed corpus."""

import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

# Ensure project root is importable when running directly
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import PROCESSED_DATA_DIR, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_lda")


def train_lda_model(num_topics: int = 15):
    csv_path = os.path.join(PROCESSED_DATA_DIR, "processed_content_data.csv")
    if not os.path.exists(csv_path):
        logger.error("Preprocessed data not found. Run preprocess_data.py first.")
        return

    logger.info("Loading corpus from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Tokenize the preprocessed content
    logger.info("Tokenizing text...")
    docs = []
    for text in df["preprocessed_content"]:
        if isinstance(text, str) and text.strip():
            docs.append(text.split())
        else:
            docs.append([])

    # Create dictionary
    logger.info("Building Dictionary...")
    dictionary = Dictionary(docs)
    
    # Filter extremes to remove too rare or too common words
    # e.g. keep words that appear in at least 5 transcripts, and no more than 60% of them
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    
    logger.info("Dictionary contains %d unique tokens", len(dictionary))

    # Create Bag of Words corpus
    logger.info("Building BoW Corpus...")
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Train LDA model
    logger.info("Training LdaMulticore with %d topics...", num_topics)
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        workers=3,
        passes=20,
        iterations=100,
        random_state=42,
    )

    # Save to data/models
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, "lda_model.pkl")
    dict_path = os.path.join(MODELS_DIR, "lda_model_dict.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(lda_model, f)
    logger.info("Saved model to %s", model_path)
    
    with open(dict_path, "wb") as f:
        pickle.dump(dictionary, f)
    logger.info("Saved dictionary to %s", dict_path)

    # Print topics for user labeling
    print(f"\n--- TOP {num_topics} TOPICS DISCOVERED ---")
    for i, topic in lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False):
        words = ", ".join([word for word, prob in topic])
        print(f"Topic {i}: {words}")
    print("----------------------------------\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train new LDA topic model.")
    parser.add_argument("--topics", type=int, default=15, help="Number of topics")
    args = parser.parse_args()
    
    train_lda_model(num_topics=args.topics)
