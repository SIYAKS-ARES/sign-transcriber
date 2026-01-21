"""Integration adapters for the sign-transcriber ecosystem."""

from .input_adapter import TranscriptionInput, adapt_input
from .anlamlandirma_adapter import AnlamlandirmaAdapter

__all__ = ["TranscriptionInput", "adapt_input", "AnlamlandirmaAdapter"]
