"""Evaluation module for BLEU/BERTScore benchmarking."""

from .benchmark import TranslationBenchmark
from .baseline import BaselineTranslator

__all__ = ["TranslationBenchmark", "BaselineTranslator"]
