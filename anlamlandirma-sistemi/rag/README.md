# RAG Modülü (anlamlandirma-sistemi/rag)

Bu klasör, `anlamlandirma-sistemi` içine gömülü **TİD → Türkçe RAG çeviri modülüdür**. Amaç, LLM’in çeviri üretimini:

- **TID_Sozluk** (statik sözlük) ile kelime tabanlı anlam/örnek üzerinden “grounding”
- **TID_Hafiza** (dinamik çeviri hafızası) ile benzer cümle örnekleri üzerinden “few-shot”

yaparak güçlendirmektir.

---

## Mimarî (Kısaca)

```
TID Gloss
  |
  v
preprocessing/ (TIDPreprocessor)
  |
  v
retriever/ (DualRetriever)
  |                 \
  |                  \
  v                   v
tid_hafiza (top-k=3)  tid_sozluk (kelime başına top-k=1)
  \                   /
   \                 /
    v               v
prompt_builder/ (system + few-shot + RAG context)
  |
  v
llm/ (Gemini/OpenAI)
  |
  v
response_parser/ (3 alternatif parse)
  |
  v
çıktı: best_translation + alternatives[]
```

---

## Alt Modüller

| Klasör | Sorumluluk |
|--------|------------|
| `preprocessing/` | TİD’e özgü belirteçler, zaman/olumsuzluk/soru/tekrar tespiti |
| `tid_collections/` | ChromaDB koleksiyonları: `tid_sozluk` ve `tid_hafiza` |
| `retriever/` | İki aşamalı retrieval (hafıza + sözlük) |
| `prompt_builder/` | Prompt şablonları + few-shot seçimi + sistem talimatları |
| `llm/` | Sağlayıcı bağımsız LLM istemcisi + yanıt parse |
| `pipeline/` | Uçtan uca orkestrasyon (preprocess → retrieve → prompt → LLM → parse) |
| `feedback/` | Human-in-the-loop kayıt/geri besleme (hafızayı büyütme) |

---

## Konfigürasyon

Ana ayarlar: `rag/config.py`

Öne çıkanlar:
- `TID_SOZLUK_PATH`: Varsayılan `../../TID_Sozluk_Verileri` (env ile override edilebilir)
- `VECTORSTORE_PATH`: Varsayılan `anlamlandirma-sistemi/vectorstore`
- `SOZLUK_COLLECTION`: `tid_sozluk`
- `HAFIZA_COLLECTION`: `tid_hafiza`
- `HAFIZA_TOP_K=3`, `SOZLUK_TOP_K=1`, `SIMILARITY_THRESHOLD=0.5`
- `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Ortam değişkenleri:

```bash
export VECTORSTORE_PATH=/custom/vectorstore
export TID_SOZLUK_PATH=/custom/TID_Sozluk_Verileri
export DEFAULT_LLM=gemini
```

---

## Hızlı Kullanım

### 1) Pipeline ile çeviri

```python
from rag.pipeline.translation_pipeline import TranslationPipeline

pipeline = TranslationPipeline(provider="gemini")
result = pipeline.translate("IKI ABLA VAR EVLENMEK GITMEK")

print(result.best_translation)
for alt in result.translation_result.alternatives:
    print(alt.confidence, alt.translation)
```

### 2) Retrieval bağlamını görmek

```python
from rag.retriever.dual_retriever import DualRetriever

retriever = DualRetriever()
res = retriever.retrieve("OKUL GITMEK ISTEMEK")
print(res.to_context_string())
```

---

## Vectorstore Notları

Bu modül ChromaDB persistent storage kullanır:
- `anlamlandirma-sistemi/vectorstore/`

Eğer bu klasör boşsa veya taşındıysa:
- `VECTORSTORE_PATH` ile doğru yolu gösterin
- veya `TRANSKRIPSIYON-RAG-VDB/vectorstore`’dan buraya kopyalayın (bkz. `anlamlandirma-sistemi/RAG_ENTEGRASYON.md`)

---

## Makale İçin Not

Bu modül, “Iterative Dictionary-Augmented Generation with Feedback Loop” yaklaşımının uygulamasıdır:
- Statik sözlük (grounding)
- Dinamik hafıza (translation memory / few-shot)
- İnsan onayı ile hafızayı büyütme (feedback loop)

Benchmark/raporlar için:
- `TRANSKRIPSIYON-RAG-VDB/evaluation/` (BLEU/BERTScore raporları)
- `anlamlandirma-sistemi/experiments/` (3/4/5 kelime deney altyapısı)

---

**Son Güncelleme:** 2026-01-21

