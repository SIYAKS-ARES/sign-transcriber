import os
import re


def parse_structured_output(llm_response: str) -> dict:
    """LLM'den gelen yapılandırılmış çıktıyı ayrıştırır."""
    try:
        translation_match = re.search(r"Çeviri:\s*(.*)", llm_response)
        confidence_match = re.search(r"Güven:\s*(\d+)\s*/\s*10", llm_response)
        explanation_match = re.search(r"Açıklama:\s*(.*)", llm_response, re.DOTALL)

        translation = (translation_match.group(1) if translation_match else "").strip()
        confidence = int(confidence_match.group(1)) if confidence_match else 0
        explanation = (explanation_match.group(1) if explanation_match else "").strip()

        if not translation:
            raise ValueError("Çeviri alanı boş")

        return {
            "translation": translation,
            "confidence": confidence,
            "explanation": explanation,
            "error": None,
        }
    except Exception as e:
        return {
            "translation": (llm_response or "").strip(),
            "confidence": 0,
            "explanation": "LLM çıktısı beklenen formatta ayrıştırılamadı.",
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
    return parse_structured_output(text)


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
    return parse_structured_output(text)


def _translate_with_gemini(prompt: str) -> dict:
    try:
        import google.generativeai as genai
    except Exception as exc:
        message = str(exc)
        fix_hint = ""
        if "TypeAliasType" in message:
            fix_hint = " - Çözüm için 'pip install --upgrade typing_extensions google-generativeai' komutlarını çalıştırmayı deneyin."
        raise RuntimeError(
            "Google Generative AI kütüphanesi yüklenemedi: "
            f"{message}{fix_hint}"
        )

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı")
    genai.configure(api_key=api_key)
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    # SDK farklı alanlarda metin döndürebilir
    try:
        text = response.text
    except Exception:
        text = str(response)
    return parse_structured_output((text or "").strip())


def translate_with_llm(provider: str, prompt: str) -> dict:
    provider_key = (provider or '').strip().lower()
    if provider_key in ('openai', 'gpt', 'gpt-4', 'gpt-4o'):
        return _translate_with_openai(prompt)
    if provider_key in ('claude', 'anthropic'):
        return _translate_with_claude(prompt)
    if provider_key in ('gemini', 'google'):
        return _translate_with_gemini(prompt)
    raise ValueError(f"Bilinmeyen LLM sağlayıcısı: {provider}")


