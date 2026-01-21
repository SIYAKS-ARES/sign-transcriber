"""
RAG (Retrieval-Augmented Generation) Module for TID Translation
================================================================
This module provides advanced RAG capabilities for Turkish Sign Language
transcription to Turkish translation.

Components:
- preprocessing: TID-specific text preprocessing
- tid_collections: ChromaDB collection management (Sozluk + Hafiza)
- retriever: Dual-collection retrieval system
- prompt_builder: Dynamic prompt construction with few-shot examples
- llm: LLM client and response parsing
- pipeline: Complete translation pipeline orchestrator
- feedback: Human-in-the-loop feedback handling
"""

from rag.config import (
    VECTORSTORE_PATH,
    TID_SOZLUK_PATH,
    EMBEDDING_MODEL,
    DEFAULT_LLM,
)

__all__ = [
    'VECTORSTORE_PATH',
    'TID_SOZLUK_PATH',
    'EMBEDDING_MODEL',
    'DEFAULT_LLM',
]
