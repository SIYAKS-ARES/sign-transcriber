# RAG Entegrasyonu Teknik Dokumantasyonu

Bu belge, anlamlandirma-sistemi'ne eklenen RAG (Retrieval-Augmented Generation) sisteminin teknik detaylarini aciklar.

## Genel Bakis

RAG sistemi, TRANSKRIPSIYON-RAG-VDB projesinden entegre edilmis olup, TID transkripsiyon cumlelerinin Turkce'ye cevirisini iki katmanli retrieval ile zenginlestirmektedir.

### Mimari

```
anlamlandirma-sistemi/
├── rag/                          # RAG modulu
│   ├── config.py                 # Yapilandirma
│   ├── preprocessing/            # TID on-isleme
│   │   ├── tid_preprocessor.py   # Dilbilgisi analizi
│   │   └── cleaning.py           # Veri temizleme
│   ├── tid_collections/          # ChromaDB koleksiyonlari
│   │   ├── sozluk_collection.py  # Statik sozluk
│   │   └── hafiza_collection.py  # Dinamik hafiza
│   ├── retriever/                # Retrieval sistemi
│   │   └── dual_retriever.py     # Ikili retrieval
│   ├── prompt_builder/           # Prompt olusturma
│   │   ├── system_instructions.py
│   │   ├── few_shot_builder.py
│   │   └── templates.py
│   ├── llm/                      # LLM istemcisi
│   │   ├── llm_client.py
│   │   └── response_parser.py
│   ├── pipeline/                 # Orchestrator
│   │   └── translation_pipeline.py
│   └── feedback/                 # HITL geribildirim
│       └── feedback_handler.py
└── vectorstore/                  # ChromaDB verileri
    ├── chroma.sqlite3
    └── [koleksiyon klasorleri]
```

## Bilesenler

### 1. TIDPreprocessor

TID'e ozgu dilbilgisi analizi yapan on-isleyici.

**Tespit edilen ozellikler:**
- Zaman belirtecleri (DUN, YARIN, SIMDI)
- Soru yapilari (NE, NEREDE, KIM)
- Olumsuzluk (DEGIL, YOK)
- Kelime tekrarlari (GEZMEK GEZMEK)
- Birlesik kelimeler (ARABA^SURMEK)

```python
from rag.preprocessing.tid_preprocessor import TIDPreprocessor

preprocessor = TIDPreprocessor()
result = preprocessor.preprocess("DUN BEN OKUL GITMEK")
print(result.detected_tense)  # "past"
print(result.processed)       # "DUN BEN OKUL GITMEK"
```

### 2. Dual Retriever

Iki katmanli retrieval sistemi:

1. **TID_Hafiza (Memory)**: Benzer ceviri ornekleri (top-k=3)
2. **TID_Sozluk (Dictionary)**: Kelime bazli anlam bilgisi (top-k=1/kelime)

```python
from rag.retriever.dual_retriever import DualRetriever

retriever = DualRetriever()
result = retriever.retrieve("BEN OKUL GITMEK")
print(result.similar_translations)  # Hafizadan ornekler
print(result.word_info)             # Sozlukten kelime bilgileri
```

### 3. TranslationPipeline

Tum bilesenleri koordine eden ana pipeline.

```python
from rag.pipeline.translation_pipeline import TranslationPipeline

pipeline = TranslationPipeline(provider="gemini")
result = pipeline.translate("BEN OKUL GITMEK")
print(result.best_translation)
print(result.alternatives)  # 3 alternatif ceviri
```

## Yapilandirma

### Environment Variables

```bash
# Zorunlu
GEMINI_API_KEY=your-key-here

# Opsiyonel path override
VECTORSTORE_PATH=/custom/path/to/vectorstore
TID_SOZLUK_PATH=/custom/path/to/TID_Sozluk_Verileri
```

### config.py Parametreleri

| Parametre | Deger | Aciklama |
|-----------|-------|----------|
| HAFIZA_TOP_K | 3 | Hafizadan alinacak ornek sayisi |
| SOZLUK_TOP_K | 1 | Her kelime icin sozluk sonucu |
| SIMILARITY_THRESHOLD | 0.5 | Minimum benzerlik esigi |
| EMBEDDING_MODEL | paraphrase-multilingual-MiniLM-L12-v2 | Embedding modeli |

## Fallback Mekanizmasi

RAG sistemi baslatılamadığında, sistem otomatik olarak basit TF-IDF tabanli fallback moduna gecer:

```python
from preprocessor import is_rag_available, translate_with_rag

if is_rag_available():
    result = translate_with_rag("BEN OKUL GITMEK")
else:
    # Fallback: Basit prompt ile ceviri
    ...
```

## Vectorstore

ChromaDB tabanlı vektorstore iki koleksiyon icerir:

1. **tid_sozluk**: ~2800 TID kelimesi ve anlamlari
2. **tid_hafiza**: ~2800+ ceviri ornegi

### Kontrol ve Baslama

```bash
# Vectorstore kontrolu
python scripts/init_vectorstore.py --check

# Istatistikleri goster
python scripts/init_vectorstore.py --stats
```

## API Endpoints

### RAG Status
```
GET /api/rag_status
```

### RAG ile Ceviri
```
POST /api/translate_rag
Content-Type: application/json

{"gloss": "BEN OKUL GITMEK"}
```

## Hata Ayiklama

### Import Hatalari

RAG modulu `rag.` prefix'i ile import edilmelidir:

```python
# Dogru
from rag.pipeline.translation_pipeline import TranslationPipeline
from rag.config import VECTORSTORE_PATH

# Yanlis
from pipeline.translation_pipeline import TranslationPipeline
```

### Vectorstore Bulunamadi

```bash
# Cozum: Hazir vectorstore'u kopyala
cp -r ../TRANSKRIPSIYON-RAG-VDB/vectorstore .
```

## Performans Notlari

- Ilk calistirmada sentence-transformers modeli indirilir (~400MB)
- ChromaDB sorgu suresi: ~50-100ms
- LLM ceviri suresi: ~1-3s (provider'a bagli)
