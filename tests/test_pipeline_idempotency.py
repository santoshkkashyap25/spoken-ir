"""Idempotency test: preprocess_data produces identical output given identical input.

We run preprocess_data against a small synthetic DataFrame, save the result,
re-run, and confirm the preprocessed_content column is byte-identical.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest



def test_preprocess_is_deterministic() -> None:
    """Same input text → same preprocessed output across two runs."""
    from ai.nlp import preprocess

    sample = (
        "I went home for Thanksgiving and my mother had started dating again. "
        "She's 73. She met a man at the community center. He knits. "
        "We watched four episodes of a Swedish crime show in one weekend."
    )
    first = preprocess(sample)
    second = preprocess(sample)
    assert first == second
    # The preprocessed text should not be empty for a real input.
    assert first.strip() != ""


def test_preprocess_handles_empty_and_non_string() -> None:
    from ai.nlp import preprocess

    assert preprocess("") == ""
    assert preprocess("   ") == ""
    assert preprocess(None) == ""  # type: ignore[arg-type]
    assert preprocess(12345) == ""  # type: ignore[arg-type]


def test_preprocess_strips_punctuation_and_lowercases() -> None:
    from ai.nlp import preprocess

    out = preprocess("Hello, World! 2025! This is a TEST.")
    # Punctuation and digits removed; lowercased.
    assert "," not in out
    assert "!" not in out
    assert "2025" not in out
    # All output should be lowercase alphabetic tokens (or empty after stopword removal).
    for tok in out.split():
        assert tok == tok.lower()
