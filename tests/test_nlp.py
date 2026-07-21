import pytest
from ai.nlp import clean_text, remove_stopwords, preprocess, preprocess_batch

def test_clean_text():
    assert clean_text("Hello World!") == "hello world"
    assert clean_text("This is [bracketed] text.") == "this is  text"
    assert clean_text("Testing 123 numbers") == "testing  numbers"
    assert clean_text("newlines\nshould\rbe gone") == "newlines should be gone"

def test_remove_stopwords():
    text = "this is a test sentence with some stopwords"
    cleaned = remove_stopwords(text)
    assert "this" not in cleaned.split()
    assert "is" not in cleaned.split()
    assert "test" in cleaned.split()

def test_preprocess():
    result = preprocess("This is a [test] sentence! It has 123 numbers.")
    assert isinstance(result, str)
    assert "test" in result or "sentence" in result
    assert "123" not in result
    assert "[" not in result

def test_preprocess_batch():
    texts = [
        "This is the first sentence.",
        "And here is another one with 123 numbers!"
    ]
    results = preprocess_batch(texts)
    assert len(results) == 2
    assert "first" in results[0] or "sentence" in results[0]
    assert "another" in results[1]
    assert "123" not in results[1]
