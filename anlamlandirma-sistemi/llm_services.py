import os
import re
import time
from typing import List, Dict, Optional

# Rate limit configuration
RATE_LIMIT_RETRY_DELAY = 2.0  # seconds between retries


def parse_multi_alternative_output(llm_response: str) -> dict:
    """
    LLM'den gelen 3 alternatifli ciktiyi ayristirir.
    
    Format:
    ## ALTERNATIF 1
    Ceviri: ...
    Guven: X/10
    Aciklama: ...
    
    Returns dict with 'translation', 'confidence', 'explanation', 'alternatives', 'error'
    """
    if not llm_response:
        return {
            "translation": "",
            "confidence": 0,
            "explanation": "",
            "alternatives": [],
            "error": "Bos yanit",
        }
    
    alternatives = []
    
    # Try to parse multi-alternative format
    # Match patterns like "## ALTERNATIF 1" or "## ALTERNATİF 1" or "## Alternatif 1"
    alt_pattern = re.compile(
        r"##\s*(?:ALTERNATIF|ALTERNATİF|Alternatif)\s*(\d+).*?"
        r"(?:Ceviri|Çeviri|CEVIRI|ÇEVİRİ):\s*([^\n]+).*?"
        r"(?:Guven|Güven|GUVEN|GÜVEN):\s*(\d+)\s*/\s*10.*?"
        r"(?:Aciklama|Açıklama|ACIKLAMA|AÇIKLAMA):\s*([^\n]+(?:\n(?!##)[^\n]*)*)",
        re.IGNORECASE | re.DOTALL
    )
    
    matches = alt_pattern.findall(llm_response)
    
    for match in matches:
        alt_num, translation, confidence, explanation = match
        alternatives.append({
            "translation": translation.strip(),
            "confidence": int(confidence),
            "explanation": explanation.strip()[:200],  # Limit explanation length
            "alternative_number": int(alt_num),
        })
    
    # Sort by alternative number
    alternatives.sort(key=lambda x: x.get("alternative_number", 0))
    
    if alternatives:
        # Find best translation (highest confidence)
        best = max(alternatives, key=lambda x: x["confidence"])
        return {
            "translation": best["translation"],
            "confidence": best["confidence"],
            "explanation": best["explanation"],
            "alternatives": alternatives,
            "error": None,
        }
    
    # Fallback: try simple single-output format
    return parse_structured_output(llm_response)


def parse_structured_output(llm_response: str) -> dict:
    """LLM'den gelen yapilandirilmis ciktiyi ayristirir (tek ceviri formati)."""
    try:
        # Support both Turkish characters and ASCII variants
        translation_match = re.search(r"(?:Ceviri|Çeviri):\s*([^\n]+)", llm_response, re.IGNORECASE)
        confidence_match = re.search(r"(?:Guven|Güven):\s*(\d+)\s*/\s*10", llm_response, re.IGNORECASE)
        explanation_match = re.search(r"(?:Aciklama|Açıklama):\s*(.+?)(?=\n\n|\n##|$)", llm_response, re.DOTALL | re.IGNORECASE)

        translation = (translation_match.group(1) if translation_match else "").strip()
        confidence = int(confidence_match.group(1)) if confidence_match else 0
        explanation = (explanation_match.group(1) if explanation_match else "").strip()

        if not translation:
            raise ValueError("Ceviri alani bos")

        return {
            "translation": translation,
            "confidence": confidence,
            "explanation": explanation,
            "alternatives": [{
                "translation": translation,
                "confidence": confidence,
                "explanation": explanation,
            }],
            "error": None,
        }
    except Exception as e:
        return {
            "translation": (llm_response or "").strip()[:200],
            "confidence": 0,
            "explanation": "LLM ciktisi beklenen formatta ayristirilamadi.",
            "alternatives": [],
            "error": str(e),
        }


def _translate_with_openai(prompt: str) -> dict:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"OpenAI kütüphanesi yüklenemedi: {exc}")

    client = OpenAI()
    model = os.environ.get('OPENAI_MODEL', 'gpt-4o')
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = (response.choices[0].message.content or "").strip()
    return parse_multi_alternative_output(text)


def _translate_with_claude(prompt: str) -> dict:
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(f"Anthropic kütüphanesi yüklenemedi: {exc}")

    client = anthropic.Anthropic()
    model = os.environ.get('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20240620')
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    # Claude yanıtı parça listesi olabilir
    text_parts = []
    for block in msg.content:
        if getattr(block, 'type', None) == 'text':
            text_parts.append(block.text)
        elif isinstance(block, dict) and block.get('type') == 'text':
            text_parts.append(block.get('text', ''))
    text = "".join(text_parts).strip()
    return parse_multi_alternative_output(text)


def _get_gemini_api_keys() -> List[str]:
    """Get all available Gemini API keys."""
    keys = []
    for key_name in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GOOGLE_API_KEY"]:
        key = os.environ.get(key_name)
        if key and key not in keys:
            keys.append(key)
    return keys


def _translate_with_gemini(prompt: str) -> dict:
    """Translate with Gemini, with automatic API key rotation on rate limit."""
    try:
        import google.generativeai as genai
    except Exception as exc:
        message = str(exc)
        fix_hint = ""
        if "TypeAliasType" in message:
            fix_hint = " - Cozum icin 'pip install --upgrade typing_extensions google-generativeai' komutlarini calistirmayi deneyin."
        raise RuntimeError(
            "Google Generative AI kutuphanesi yuklenemedi: "
            f"{message}{fix_hint}"
        )

    api_keys = _get_gemini_api_keys()
    if not api_keys:
        raise RuntimeError("GEMINI_API_KEY bulunamadi")
    
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
    last_error = None
    
    for key_index, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # SDK farkli alanlarda metin dondurebilir
            try:
                text = response.text
            except Exception:
                text = str(response)
            
            return parse_multi_alternative_output((text or "").strip())
            
        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                if key_index < len(api_keys) - 1:
                    print(f"  [Rate limit - switching to API key #{key_index + 2}]")
                    time.sleep(RATE_LIMIT_RETRY_DELAY)
                    continue
            
            # Not a rate limit error, or no more keys
            raise e
    
    # All keys exhausted
    raise last_error or RuntimeError("Tum API key'ler tukendi")


def translate_with_llm(provider: str, prompt: str) -> dict:
    provider_key = (provider or '').strip().lower()
    if provider_key in ('openai', 'gpt', 'gpt-4', 'gpt-4o'):
        return _translate_with_openai(prompt)
    if provider_key in ('claude', 'anthropic'):
        return _translate_with_claude(prompt)
    if provider_key in ('gemini', 'google'):
        return _translate_with_gemini(prompt)
    raise ValueError(f"Bilinmeyen LLM sağlayıcısı: {provider}")


