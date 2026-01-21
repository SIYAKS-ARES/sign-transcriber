"""
Translation Pipeline Orchestrator
==================================
Combines all components: preprocessing, RAG retrieval, prompt building, and LLM translation.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.tid_preprocessor import (
    TIDPreprocessor,
    PreprocessedInput,
    extract_words_for_rag,
)
from retriever.dual_retriever import DualRetriever, RetrievalResult
from prompt_builder.few_shot_builder import FewShotBuilder
from prompt_builder.templates import build_user_prompt
from prompt_builder.system_instructions import build_dynamic_system_instruction
from llm.llm_client import LLMClient, LLMConfig
from llm.response_parser import TranslationResult


ProviderType = Literal["gemini", "openai"]


@dataclass
class PipelineResult:
    """Complete result from the translation pipeline."""
    
    # Final translation result
    translation_result: TranslationResult = None
    
    # Intermediate results
    preprocessed: PreprocessedInput = None
    retrieval_result: RetrievalResult = None
    
    # Prompts (for debugging/logging)
    system_instruction: str = ""
    user_prompt: str = ""
    
    # Metadata
    provider: str = ""
    
    @property
    def best_translation(self) -> str:
        """Get the best translation."""
        if self.translation_result and self.translation_result.best:
            return self.translation_result.best.translation
        return ""
    
    @property
    def confidence(self) -> int:
        """Get confidence of best translation."""
        if self.translation_result and self.translation_result.best:
            return self.translation_result.best.confidence
        return 0
    
    @property
    def is_successful(self) -> bool:
        """Check if translation was successful."""
        return self.translation_result is not None and self.translation_result.is_successful
    
    def to_display_string(self) -> str:
        """Format for display."""
        lines = []
        
        # Original transcription
        if self.preprocessed:
            lines.append(f"Transkripsiyon: {self.preprocessed.original}")
            lines.append("")
        
        # Best translation
        if self.translation_result and self.translation_result.best:
            best = self.translation_result.best
            lines.append(f"Ceviri: {best.translation}")
            lines.append(f"Guven: {best.confidence}/10")
            lines.append(f"Aciklama: {best.explanation}")
            lines.append("")
        
        # All alternatives
        if self.translation_result and len(self.translation_result.alternatives) > 1:
            lines.append("Tum Alternatifler:")
            for alt in self.translation_result.get_sorted_alternatives():
                lines.append(f"  [{alt.confidence}/10] {alt.translation}")
        
        # Preprocessing info
        if self.preprocessed:
            lines.append("")
            lines.append("Dilbilgisi Analizi:")
            if self.preprocessed.detected_tense:
                lines.append(f"  - Zaman: {self.preprocessed.detected_tense}")
            if self.preprocessed.is_question:
                lines.append("  - Tip: Soru cumlesi")
            if self.preprocessed.is_negative:
                lines.append("  - Tip: Olumsuz cumle")
            if self.preprocessed.repetitions:
                lines.append(f"  - Tekrarlar: {self.preprocessed.repetitions}")
        
        return "\n".join(lines)


class TranslationPipeline:
    """
    Complete TID translation pipeline.
    
    Pipeline stages:
    1. Preprocessing: TID-specific markers, tense detection, etc.
    2. RAG Retrieval: Similar sentences (Hafiza) + word info (Sozluk)
    3. Few-shot Building: Dynamic example selection
    4. Prompt Building: Combine all context
    5. LLM Translation: Generate translation
    6. Response Parsing: Extract alternatives
    """
    
    def __init__(
        self,
        provider: ProviderType = "gemini",
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            provider: LLM provider ("gemini" or "openai")
            llm_config: Optional LLM configuration
        """
        self.provider = provider
        self.llm_config = llm_config or LLMConfig(provider=provider)
        
        # Initialize components
        self.preprocessor = TIDPreprocessor()
        self.retriever = DualRetriever()
        self.few_shot_builder = FewShotBuilder()
        self.llm_client = None  # Lazy initialization
    
    def _ensure_llm_client(self, system_instruction: str):
        """Ensure LLM client is initialized with current system instruction."""
        if self.llm_client is None:
            self.llm_client = LLMClient(
                provider=self.provider,
                config=self.llm_config,
                system_instruction=system_instruction,
            )
        else:
            # Update system instruction if needed
            if self.llm_client.system_instruction != system_instruction:
                self.llm_client = LLMClient(
                    provider=self.provider,
                    config=self.llm_config,
                    system_instruction=system_instruction,
                )
    
    def translate(self, transcription: str) -> PipelineResult:
        """
        Translate a TID transcription to Turkish.
        
        Args:
            transcription: Raw TID transcription string
            
        Returns:
            PipelineResult with translation and all intermediate results
        """
        result = PipelineResult(provider=self.provider)
        
        # =====================================================================
        # Stage 1: Preprocessing
        # =====================================================================
        preprocessed = self.preprocessor.preprocess(transcription)
        result.preprocessed = preprocessed
        
        # =====================================================================
        # Stage 2: RAG Retrieval
        # =====================================================================
        # Use preprocessed text for RAG to handle markers
        retrieval_result = self.retriever.retrieve(preprocessed.processed)
        result.retrieval_result = retrieval_result
        
        # =====================================================================
        # Stage 3: Few-shot Example Building
        # =====================================================================
        # Get similar translations for dynamic examples
        hafiza_results = retrieval_result.similar_translations
        
        few_shot_examples = self.few_shot_builder.build_examples(
            detected_tense=preprocessed.detected_tense,
            is_question=preprocessed.is_question,
            is_negative=preprocessed.is_negative,
            repetitions=preprocessed.repetitions,
            compound_words=preprocessed.compound_words,
            hafiza_results=hafiza_results,
        )
        
        # =====================================================================
        # Stage 4: Build System Instruction (dynamic)
        # =====================================================================
        system_instruction = build_dynamic_system_instruction(
            tense=preprocessed.detected_tense,
            tense_source=preprocessed.tense_source,
            is_question=preprocessed.is_question,
            is_negative=preprocessed.is_negative,
            repetitions=preprocessed.repetitions,
            use_compact=self.llm_config.use_compact_system,
        )
        result.system_instruction = system_instruction
        
        # =====================================================================
        # Stage 5: Build User Prompt
        # =====================================================================
        rag_context = retrieval_result.to_context_string()
        
        user_prompt = build_user_prompt(
            transcription=preprocessed.processed,
            few_shot_examples=few_shot_examples,
            rag_context=rag_context,
            detected_tense=preprocessed.detected_tense,
            tense_source=preprocessed.tense_source,
            is_question=preprocessed.is_question,
            is_negative=preprocessed.is_negative,
            repetitions=preprocessed.repetitions,
            linguistic_hints=preprocessed.linguistic_hints,
        )
        result.user_prompt = user_prompt
        
        # =====================================================================
        # Stage 6: LLM Translation
        # =====================================================================
        self._ensure_llm_client(system_instruction)
        translation_result = self.llm_client.translate(user_prompt)
        result.translation_result = translation_result
        
        return result
    
    def translate_simple(self, transcription: str) -> str:
        """
        Simple interface that returns just the best translation.
        
        Args:
            transcription: Raw TID transcription
            
        Returns:
            Best translation string
        """
        result = self.translate(transcription)
        return result.best_translation
    
    def get_stats(self) -> dict:
        """Get statistics about the system."""
        return self.retriever.get_stats()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def translate_transcription(
    transcription: str,
    provider: ProviderType = "gemini",
) -> PipelineResult:
    """
    Convenience function to translate a TID transcription.
    
    Args:
        transcription: Raw TID transcription
        provider: LLM provider
        
    Returns:
        PipelineResult with translation
    """
    pipeline = TranslationPipeline(provider=provider)
    return pipeline.translate(transcription)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import os
    
    print("Translation Pipeline Test")
    print("=" * 60)
    
    # Check if API keys are available
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"Gemini API Key: {'Set' if gemini_key else 'Not set'}")
    print(f"OpenAI API Key: {'Set' if openai_key else 'Not set'}")
    
    # Test preprocessing only (no API key needed)
    print("\n" + "-" * 60)
    print("Testing Preprocessing (no API needed):")
    print("-" * 60)
    
    preprocessor = TIDPreprocessor()
    
    test_cases = [
        "BUGÜN ARKADAŞ BULUŞMAK GEZMEK GEZMEK BİTMEK",
        "SEN NEREDE ÇALIŞMAK",
        "BEN YARIN OKUL GITMEK",
        "YEMEK BEGENMEK DEGIL",
    ]
    
    for test in test_cases:
        result = preprocessor.preprocess(test)
        print(f"\nInput: {test}")
        print(f"  Processed: {result.processed}")
        print(f"  Tense: {result.detected_tense} ({result.tense_source})")
        print(f"  Question: {result.is_question}")
        print(f"  Negative: {result.is_negative}")
        print(f"  Repetitions: {result.repetitions}")
    
    # Full pipeline test (requires API key)
    if gemini_key or openai_key:
        provider = "gemini" if gemini_key else "openai"
        print(f"\n" + "=" * 60)
        print(f"Testing Full Pipeline with {provider}:")
        print("=" * 60)
        
        try:
            pipeline = TranslationPipeline(provider=provider)
            
            # Get stats
            stats = pipeline.get_stats()
            print(f"Sozluk entries: {stats['sozluk_count']}")
            print(f"Hafiza entries: {stats['hafiza_count']}")
            
            # Test translation
            test_input = "DÜN ARKADAŞ BULUŞMAK KAHVE İÇMEK"
            print(f"\nTranslating: {test_input}")
            print("-" * 60)
            
            result = pipeline.translate(test_input)
            print(result.to_display_string())
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n" + "=" * 60)
        print("Skipping full pipeline test (no API key)")
        print("Set GOOGLE_API_KEY or OPENAI_API_KEY to test.")
