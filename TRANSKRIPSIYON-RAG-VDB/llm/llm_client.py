"""
LLM Client with System Instruction Support
==========================================
Unified client for Gemini and OpenAI with system instruction support.
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_builder.system_instructions import (
    TID_SYSTEM_INSTRUCTION,
    build_dynamic_system_instruction,
)
from .response_parser import ResponseParser, TranslationResult


# Provider type
ProviderType = Literal["gemini", "openai"]


# Available Gemini models
GEMINI_MODELS = {
    "lite": "gemini-2.5-flash-lite",      # Hizli, ekonomik
    "flash": "gemini-2.5-flash",          # Dengeli, thinking destegi
    "pro": "gemini-2.5-pro",              # En guclu (kota sinirli)
    "pro3": "gemini-3-pro-preview",       # Yeni nesil (kota sinirli)
}


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: ProviderType = "gemini"
    model_name: Optional[str] = None  # Auto-select based on provider
    model_tier: str = "flash"  # "lite", "flash", "pro", "pro3"
    temperature: float = 0.7
    max_tokens: int = 2048
    use_compact_system: bool = False
    use_alt_api_key: bool = True  # Alternatif API key kullan
    
    def get_model_name(self) -> str:
        """Get the model name based on provider."""
        if self.model_name:
            return self.model_name
        
        if self.provider == "gemini":
            return GEMINI_MODELS.get(self.model_tier, GEMINI_MODELS["flash"])
        elif self.provider == "openai":
            return "gpt-4o"
        else:
            return GEMINI_MODELS["flash"]


class LLMClient:
    """
    Unified LLM client with system instruction support.
    
    Supports:
    - Google Gemini (with system_instruction parameter)
    - OpenAI GPT (with system message)
    """
    
    def __init__(
        self,
        provider: ProviderType = "gemini",
        config: Optional[LLMConfig] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider ("gemini" or "openai")
            config: Optional configuration override
            system_instruction: Custom system instruction (defaults to TID_SYSTEM_INSTRUCTION)
        """
        self.config = config or LLMConfig(provider=provider)
        self.provider = self.config.provider
        self.system_instruction = system_instruction or TID_SYSTEM_INSTRUCTION
        self.parser = ResponseParser()
        
        # Initialize provider-specific client
        self._client = None
        self._model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
    
    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            
            # Alternatif API key secimi
            if self.config.use_alt_api_key:
                api_key = os.getenv("GEMINI_API_KEY_2") or os.getenv("GEMINI_API_KEY_ALT")
            
            # Fallback to primary key
            if not api_key or not self.config.use_alt_api_key:
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEY_2")
            
            genai.configure(api_key=api_key)
            
            # Create model with system instruction
            self._model = genai.GenerativeModel(
                model_name=self.config.get_model_name(),
                system_instruction=self.system_instruction,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
            )
            
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._client = OpenAI(api_key=api_key)
            
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def generate(self, user_prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            user_prompt: The user prompt (system instruction is already configured)
            
        Returns:
            Raw LLM response text
        """
        if self.provider == "gemini":
            return self._generate_gemini(user_prompt)
        elif self.provider == "openai":
            return self._generate_openai(user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_gemini(self, user_prompt: str) -> str:
        """Generate response using Gemini."""
        if not self._model:
            raise RuntimeError("Gemini model not initialized")
        
        response = self._model.generate_content(user_prompt)
        return response.text
    
    def _generate_openai(self, user_prompt: str) -> str:
        """Generate response using OpenAI."""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")
        
        response = self._client.chat.completions.create(
            model=self.config.get_model_name(),
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def translate(self, user_prompt: str) -> TranslationResult:
        """
        Generate translation and parse the response.
        
        Args:
            user_prompt: The user prompt with transcription and context
            
        Returns:
            Parsed TranslationResult
        """
        raw_response = self.generate(user_prompt)
        return self.parser.parse(raw_response)
    
    def translate_simple(self, user_prompt: str) -> str:
        """
        Generate translation and return just the best translation.
        
        Args:
            user_prompt: The user prompt
            
        Returns:
            Best translation string
        """
        result = self.translate(user_prompt)
        return result.best_translation if result.is_successful else result.raw_response
    
    def update_system_instruction(
        self,
        detected_tense: Optional[str] = None,
        tense_source: Optional[str] = None,
        is_question: bool = False,
        is_negative: bool = False,
        repetitions: Optional[dict] = None,
    ):
        """
        Update system instruction with dynamic context.
        Only works for Gemini (OpenAI uses per-request system message).
        
        Args:
            detected_tense: Detected tense from preprocessor
            tense_source: Source of tense detection
            is_question: Whether question markers detected
            is_negative: Whether negation markers detected
            repetitions: Word repetitions
        """
        self.system_instruction = build_dynamic_system_instruction(
            tense=detected_tense,
            tense_source=tense_source,
            is_question=is_question,
            is_negative=is_negative,
            repetitions=repetitions,
            use_compact=self.config.use_compact_system,
        )
        
        # Reinitialize for Gemini to update system instruction
        if self.provider == "gemini":
            self._init_gemini()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def translate_with_llm(
    user_prompt: str,
    provider: ProviderType = "gemini",
    system_instruction: Optional[str] = None,
) -> TranslationResult:
    """
    Convenience function to translate with LLM.
    
    Args:
        user_prompt: The user prompt
        provider: LLM provider
        system_instruction: Custom system instruction
        
    Returns:
        Parsed TranslationResult
    """
    client = LLMClient(provider=provider, system_instruction=system_instruction)
    return client.translate(user_prompt)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("LLM Client Test")
    print("=" * 50)
    
    # Check if API keys are available
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"Gemini API Key: {'Set' if gemini_key else 'Not set'}")
    print(f"OpenAI API Key: {'Set' if openai_key else 'Not set'}")
    
    if not gemini_key and not openai_key:
        print("\nNo API keys found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
    else:
        # Test with available provider
        provider = "gemini" if gemini_key else "openai"
        print(f"\nTesting with {provider}...")
        
        try:
            client = LLMClient(provider=provider)
            
            test_prompt = """
## DILBILGISI IPUCLARI
- Tespit edilen zaman: Gecmis zaman (-di/-mis) (acik zarf)

## ORNEKLER
Transkripsiyon: DUN ARKADAS BULUSMAK KAHVE ICMEK
Ceviri: Dun arkadasimla bulustum ve kahve ictik.

## RAG BAGLAMI
- BULUSMAK: Eylem - bir araya gelmek
- KAHVE: Ad - sicak icecek

## CEVIRILECEK TRANSKRIPSIYON
```
DUN OKUL GITMEK SINAV VERMEK
```

3 alternatif ceviri sun.
"""
            
            result = client.translate(test_prompt)
            print("\nResult:")
            print(result.to_display_string())
            
        except Exception as e:
            print(f"Error: {e}")
