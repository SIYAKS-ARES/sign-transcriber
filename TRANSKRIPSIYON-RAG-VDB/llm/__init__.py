"""LLM module for TID translation."""

from .response_parser import (
    ResponseParser,
    TranslationAlternative,
    TranslationResult,
    parse_translation_response,
)
from .llm_client import (
    LLMClient,
    translate_with_llm,
)

__all__ = [
    # Response Parser
    "ResponseParser",
    "TranslationAlternative",
    "TranslationResult",
    "parse_translation_response",
    # LLM Client
    "LLMClient",
    "translate_with_llm",
]
