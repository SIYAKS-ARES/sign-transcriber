"""
LLM API Client
OpenAI ve Anthropic destegi ile soru uretimi
"""

import json
from abc import ABC, abstractmethod
from typing import Optional
import asyncio

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


class BaseLLMClient(ABC):
    """LLM Client temel sinifi"""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        LLM'den yanit al
        
        Args:
            system_prompt: Sistem rolu promptu
            user_prompt: Kullanici promptu
            temperature: Randomness (0-2)
            max_tokens: Maksimum token sayisi
            
        Returns:
            LLM yaniti (string)
        """
        pass


class OpenAIClient(BaseLLMClient):
    """
    OpenAI GPT modelleri icin client
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            api_key: OpenAI API key (None ise settings'den alinir)
            model: Model adi (None ise settings'den alinir)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """OpenAI API ile yanit al"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},  # JSON mode
        )

        return response.choices[0].message.content


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude modelleri icin client (opsiyonel)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model

        if self.api_key:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                self.client = None
        else:
            self.client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Anthropic API ile yanit al"""
        if self.client is None:
            raise ValueError("Anthropic client baslatilmadi. API key kontrol edin.")

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.content[0].text


class GeminiClient(BaseLLMClient):
    """Google Gemini modelleri icin client - yeni google.genai SDK kullanir"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.llm_model

        if self.api_key:
            try:
                from google import genai
                from google.genai import types

                self.client = genai.Client(api_key=self.api_key)
                self._types = types
            except ImportError:
                self.client = None
                self._types = None
        else:
            self.client = None
            self._types = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Gemini API ile yanit al"""
        if self.client is None:
            raise ValueError("Gemini client baslatilmadi. GEMINI_API_KEY veya kutuphane eksik.")

        def _call_model() -> str:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        self._types.Content(
                            role="user",
                            parts=[self._types.Part(text=f"{system_prompt}\n\n{user_prompt}")]
                        )
                    ],
                    config=self._types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        # Cevabin saf JSON olmasini zorla
                        response_mime_type="application/json",
                    ),
                )
                result = response.text
                print(f"[GeminiClient] Raw response (first 500 chars): {result[:500]}")
                # Finish reason ve usage bilgisini log'la
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        print(f"[GeminiClient] Finish reason: {candidate.finish_reason}")
                if hasattr(response, 'usage_metadata'):
                    print(f"[GeminiClient] Usage: {response.usage_metadata}")
                return result
            except Exception as e:
                print(f"[GeminiClient] generate_content hatasi: {e!r}")
                raise ValueError(f"Gemini generate_content hatasi: {e!r}")

        return await asyncio.to_thread(_call_model)


class LLMClientFactory:
    """
    LLM Client factory pattern
    """

    _instances: dict[str, BaseLLMClient] = {}

    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMClient:
        """
        LLM client olustur veya mevcut instance'i don
        
        Args:
            provider: LLM provider (openai/anthropic)
            **kwargs: Provider-spesifik argumanlar
            
        Returns:
            BaseLLMClient instance
        """
        provider = provider or settings.llm_provider

        cache_key = f"{provider}:{kwargs.get('model', 'default')}"

        if cache_key not in cls._instances:
            if provider == "openai":
                cls._instances[cache_key] = OpenAIClient(**kwargs)
            elif provider == "anthropic":
                cls._instances[cache_key] = AnthropicClient(**kwargs)
            elif provider == "gemini":
                cls._instances[cache_key] = GeminiClient(**kwargs)
            else:
                raise ValueError(f"Desteklenmeyen provider: {provider}")

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls):
        """Instance cache'i temizle"""
        cls._instances.clear()


def parse_llm_response(response: str) -> dict:
    """
    LLM yanitini JSON olarak parse et
    
    Args:
        response: LLM yaniti (string)
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: Parse hatasi
    """
    response = response.strip()

    # ```json ... ``` bloklarini temizle
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end == -1:
            # Closing backticks missing, use rest of string
            response = response[start:].strip()
        else:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end == -1:
            # Closing backticks missing, use rest of string
            response = response[start:].strip()
        else:
            response = response[start:end].strip()

    # Bazi LLM'ler JSON'dan once/sonra aciklama yazabiliyor, sadece { } arasini al
    if '{' in response and '}' in response:
        start = response.find('{')
        # Son } karakterini bul
        end = response.rfind('}') + 1
        response = response[start:end]

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"[parse_llm_response] JSON parse hatasi: {e}")
        print(f"[parse_llm_response] Problematic response (first 1000 chars): {response[:1000]}")
        raise


# Default client instance
def get_llm_client() -> BaseLLMClient:
    """
    Default LLM client instance al
    """
    return LLMClientFactory.create()

