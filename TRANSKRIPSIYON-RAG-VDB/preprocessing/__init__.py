"""Preprocessing module for data cleaning, normalization, and TID analysis."""

from .cleaning import (
    clean_sozluk_entry,
    normalize_gloss,
    remove_boilerplate,
    normalize_whitespace,
    extract_word_type,
)

from .tid_preprocessor import (
    TIDPreprocessor,
    PreprocessedInput,
    extract_words_for_rag,
    get_tense_hint,
)

__all__ = [
    # Cleaning functions
    "clean_sozluk_entry",
    "normalize_gloss",
    "remove_boilerplate",
    "normalize_whitespace",
    "extract_word_type",
    # TID Preprocessor
    "TIDPreprocessor",
    "PreprocessedInput",
    "extract_words_for_rag",
    "get_tense_hint",
]
