# LLM Katmanı Rehberi (`anlamlandirma-sistemi/llm/`)

Bu klasör, `anlamlandirma-sistemi` içindeki **LLM istemcisi ve yanıt parse** bileşenlerini içerir. Desteklenen sağlayıcılar: **Gemini**, **OpenAI**, **Claude** (llm_services.py entegrasyonu üzerinden).

---

## Dosyalar

| Dosya | Amaç |
|-------|------|
| `llm_client.py` | Gemini/OpenAI için istemci (sistem talimatı + user prompt + parsing) |
| `response_parser.py` | 3 alternatifli (Ceviri/Guven/Aciklama) ve tek-çeviri çıktılar için regex parser |
| `__init__.py` | Modül içi eksportlar |

Not: `anlamlandirma-sistemi/llm_services.py` daha üst katmanda, provider seçimi ve çoklu API key rotasyonu için kullanılır; bu klasör ise `rag/` modülünün kendi LLM katmanıdır.

---

## API Anahtarları ve Env Değişkenleri

Gemini:
```bash
GEMINI_API_KEY=...         # Birincil
GEMINI_API_KEY_2=...       # Alternatif (rate limit için)
GEMINI_MODEL=gemini-2.5-flash  # Önerilen (flash-lite da desteklenir)
```

OpenAI:
```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
```

Claude:
```bash
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

---

## Çıktı Formatı (Parser Beklentisi)

Tercihen 3 alternatifli format:
```
## ALTERNATIF 1
Ceviri: ...
Guven: X/10
Aciklama: ...
```

Parser ayrıca tek-çeviri, basit “Ceviri: ...” formatını da destekler (fallback).

---

## Hızlı Kullanım

```python
from llm.llm_client import LLMClient

client = LLMClient(provider="gemini")
result = client.translate("OKUL GITMEK ISTEMEK")

print(result.best_translation)         # En iyi çeviri
print(result.translation_result.alternatives)  # Tüm alternatifler
```

Bu katman genellikle `rag/pipeline/translation_pipeline.py` tarafından çağrılır; orada sistem talimatı, RAG bağlamı ve user prompt otomatik üretilir.

---

## Notlar

- Eğer API kotası dolarsa, `llm_services.py` çoklu anahtar rotasyonu (Gemini) veya manuel prompt/yanıt akışıyla kullanılabilir.
- Model adı override: `LLMConfig(model_name=...)` veya env `GEMINI_MODEL` / `OPENAI_MODEL`.

---

**Son Güncelleme:** 2026-01-21

