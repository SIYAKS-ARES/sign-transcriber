"""
Anlamlandirma System Integration Adapter
========================================
Integrates the RAG system with the existing anlamlandirma-sistemi.
"""

from typing import Dict, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever.dual_retriever import DualRetriever
from prompt_builder.augmented_prompt import AugmentedPromptBuilder
from feedback.feedback_handler import FeedbackHandler
from integration.input_adapter import adapt_input, TranscriptionInput


class AnlamlandirmaAdapter:
    """
    Adapter to integrate RAG system with anlamlandirma-sistemi.
    
    This can replace the existing preprocessor.py functions
    with RAG-augmented versions.
    """
    
    def __init__(self):
        """Initialize the adapter with all RAG components."""
        self.retriever = DualRetriever()
        self.prompt_builder = AugmentedPromptBuilder(retriever=self.retriever)
        self.feedback_handler = FeedbackHandler(hafiza=self.retriever.hafiza)
    
    def preprocess_text_for_llm(self, transcription: str) -> str:
        """
        RAG-enhanced version of preprocess_text_for_llm.
        
        This is a drop-in replacement for the function in 
        anlamlandirma-sistemi/preprocessor.py
        
        Args:
            transcription: Raw transcription from model
            
        Returns:
            Processed transcription (normalized)
        """
        # Normalize the input
        input_data = adapt_input(transcription)
        return input_data.raw_string
    
    def create_final_prompt(self, processed_transcription: str) -> str:
        """
        RAG-enhanced version of create_final_prompt.
        
        This is a drop-in replacement for the function in
        anlamlandirma-sistemi/preprocessor.py
        
        Args:
            processed_transcription: Processed transcription string
            
        Returns:
            Complete LLM prompt with RAG context
        """
        return self.prompt_builder.build_prompt(processed_transcription)
    
    def create_baseline_prompt(self, transcription: str) -> str:
        """
        Create a baseline prompt without RAG for comparison.
        
        Args:
            transcription: The transcription to translate
            
        Returns:
            Simple prompt without RAG augmentation
        """
        return self.prompt_builder.build_simple_prompt(transcription)
    
    def handle_translation_result(
        self,
        transcription: str,
        translation: str,
        provider: str,
        confidence: float,
    ) -> str:
        """
        Process a translation result for potential feedback.
        
        Args:
            transcription: Original transcription
            translation: Generated translation
            provider: LLM provider used
            confidence: Translation confidence
            
        Returns:
            Feedback ID for later approval/rejection
        """
        return self.feedback_handler.create_feedback(
            transkripsiyon=transcription,
            generated_translation=translation,
            provider=provider,
            confidence=confidence,
        )
    
    def approve_translation(
        self,
        feedback_id: str,
        corrected: Optional[str] = None,
    ) -> Optional[str]:
        """
        Approve a translation and save to memory.
        
        Args:
            feedback_id: Feedback ID from handle_translation_result
            corrected: Optional corrected translation
            
        Returns:
            Hafiza document ID if saved
        """
        return self.feedback_handler.approve_translation(
            feedback_id,
            corrected_translation=corrected,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        retriever_stats = self.retriever.get_stats()
        feedback_stats = self.feedback_handler.get_hafiza_stats()
        return {
            **retriever_stats,
            **feedback_stats,
        }


# Singleton instance for easy import
_adapter_instance: Optional[AnlamlandirmaAdapter] = None


def get_adapter() -> AnlamlandirmaAdapter:
    """Get or create the singleton adapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = AnlamlandirmaAdapter()
    return _adapter_instance


# Drop-in replacement functions for anlamlandirma-sistemi
def preprocess_text_for_llm(transcription: str) -> str:
    """Drop-in replacement for preprocessor.preprocess_text_for_llm"""
    return get_adapter().preprocess_text_for_llm(transcription)


def create_final_prompt(processed_transcription: str) -> str:
    """Drop-in replacement for preprocessor.create_final_prompt"""
    return get_adapter().create_final_prompt(processed_transcription)


if __name__ == "__main__":
    # Test the adapter
    print("Testing Anlamlandirma Adapter...")
    print("=" * 60)
    
    adapter = AnlamlandirmaAdapter()
    stats = adapter.get_stats()
    print(f"System stats: {stats}")
    
    # Test the drop-in functions
    test_trans = "OKUL GITMEK ISTEMEK"
    processed = adapter.preprocess_text_for_llm(test_trans)
    print(f"\nProcessed: {processed}")
    
    prompt = adapter.create_final_prompt(processed)
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"First 500 chars:\n{prompt[:500]}...")
