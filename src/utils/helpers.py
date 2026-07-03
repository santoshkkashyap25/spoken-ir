"""Stub for loading legacy pickled artifacts.

The TF-IDF vectorizer at data/models/tfidf_vectorizer.pkl was pickled with
its tokenizer/analyzer closures bound to functions in this module. Live
code uses ai/nlp.py instead. This stub exists only so the legacy pickle
can be unpickled.
"""


def identity_tokenizer(text):
    return text


def identity_analyzer(text):
    return text