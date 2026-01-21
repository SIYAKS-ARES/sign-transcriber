"""
Baseline Translator for Comparison
==================================
Zero-shot LLM translation without RAG augmentation.
"""

import os
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_builder.augmented_prompt import AugmentedPromptBuilder


class BaselineTranslator:
    """
    Baseline translator using zero-shot LLM prompting.
    Used for comparison with RAG-augmented translation.
    """
    
    def __init__(self, provider: str = "gemini"):
        """
        Initialize baseline translator.
        
        Args:
            provider: LLM provider to use ("gemini", "openai")
        """
        self.provider = provider
        self.prompt_builder = AugmentedPromptBuilder()
    
    def get_prompt(self, gloss: str) -> str:
        """Get the simple (non-RAG) prompt."""
        return self.prompt_builder.build_simple_prompt(gloss)
    
    def translate(self, gloss: str) -> str:
        """
        Translate using zero-shot LLM.
        
        Args:
            gloss: TID transcription
            
        Returns:
            Translated text
        """
        prompt = self.get_prompt(gloss)
        
        try:
            # Try to use llm_services if available
            parent_dir = Path(__file__).parent.parent.parent / "anlamlandirma-sistemi"
            if parent_dir.exists():
                sys.path.insert(0, str(parent_dir))
                from llm_services import translate_with_llm
                
                result = translate_with_llm(self.provider, prompt)
                return result.get("translation", "")
        except ImportError:
            pass
        except Exception as e:
            print(f"LLM translation error: {e}")
        
        # Fallback: return placeholder
        return f"[Baseline translation needed for: {gloss}]"


if __name__ == "__main__":
    print("Testing Baseline Translator...")
    translator = BaselineTranslator()
    
    test_gloss = "OKUL GITMEK"
    prompt = translator.get_prompt(test_gloss)
    print(f"Prompt:\n{prompt}")
