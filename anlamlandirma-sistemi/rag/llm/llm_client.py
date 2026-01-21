"""
LLM Client with System Instruction Support
==========================================
Unified client for Gemini and OpenAI with system instruction support.
Includes automatic API key rotation on rate limit errors.
"""

import os
import time
from typing import Optional, Literal, List
from dataclasses import dataclass

from rag.prompt_builder.system_instructions import (
    TID_SYSTEM_INSTRUCTION,
    build_dynamic_system_instruction,
)
from rag.llm.response_parser import ResponseParser, TranslationResult


# Provider type
ProviderType = Literal["gemini", "openai"]


# Available Gemini models
GEMINI_MODELS = {
    "lite": "gemini-2.5-flash-lite",      # Hizli, ekonomik
    "flash": "gemini-2.5-flash",          # Dengeli, thinking destegi
    "pro": "gemini-2.5-pro",              # En guclu (kota sinirli)
    "pro3": "gemini-3-pro-preview",       # Yeni nesil (kota sinirli)
}

# Rate limit retry configuration
RATE_LIMIT_RETRY_DELAY = 2.0  # seconds between retries
MAX_RETRIES_PER_KEY = 1  # retries before switching keys


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
    - Automatic API key rotation on rate limit (429) errors
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
        self._genai = None  # Store genai module for key switching
        
        # API key management for Gemini
        self._gemini_api_keys: List[str] = []
        self._current_key_index = 0
        self._load_gemini_api_keys()
        
        self._initialize_client()
    
    def _load_gemini_api_keys(self):
        """Load all available Gemini API keys."""
        keys = []
        
        # Primary keys
        for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
            key = os.getenv(key_name)
            if key and key not in keys:
                keys.append(key)
        
        # Secondary/alternative keys
        for key_name in ["GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_ALT", "GOOGLE_API_KEY_2"]:
            key = os.getenv(key_name)
            if key and key not in keys:
                keys.append(key)
        
        self._gemini_api_keys = keys
        if keys:
            # Start with alt key if configured
            if self.config.use_alt_api_key and len(keys) > 1:
                self._current_key_index = 1
            else:
                self._current_key_index = 0
    
    def _get_current_gemini_key(self) -> Optional[str]:
        """Get the current Gemini API key."""
        if not self._gemini_api_keys:
            return None
        return self._gemini_api_keys[self._current_key_index]
    
    def _rotate_gemini_key(self) -> bool:
        """
        Rotate to the next Gemini API key.
        
        Returns:
            True if successfully rotated, False if no more keys
        """
        if len(self._gemini_api_keys) <= 1:
            return False
        
        self._current_key_index = (self._current_key_index + 1) % len(self._gemini_api_keys)
        print(f"  [API Key rotated to key #{self._current_key_index + 1}]")
        
        # Reinitialize with new key
        self._init_gemini()
        return True
    
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
    
    def _init_gemini(self):
        """Initialize Google Gemini client with current API key."""
        try:
            import google.generativeai as genai
            self._genai = genai
            
            api_key = self._get_current_gemini_key()
            
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
        """
        Generate response using Gemini with automatic key rotation on rate limit.
        """
        if not self._model:
            raise RuntimeError("Gemini model not initialized")
        
        tried_keys = set()
        last_error = None
        
        while len(tried_keys) < len(self._gemini_api_keys):
            current_key = self._get_current_gemini_key()
            tried_keys.add(current_key)
            
            try:
                response = self._model.generate_content(user_prompt)
                return response.text
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    # Try to rotate to another key
                    if self._rotate_gemini_key():
                        # Small delay before retry
                        time.sleep(RATE_LIMIT_RETRY_DELAY)
                        continue
                
                # Not a rate limit error, or no more keys to try
                raise e
        
        # All keys exhausted
        raise last_error or RuntimeError("All API keys exhausted")
    
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
