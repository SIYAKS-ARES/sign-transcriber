---
name: TID RAG Sistemi
overview: TID_Sozluk_Verileri'ndeki 1933 kelimeyi ve dinamik ceviri hafizasini kullanan, ChromaDB tabanli, Turkce destekli Dual-Collection RAG sistemi kurulacak ve mevcut anlamlandirma-sistemi ile entegre edilecek.
todos:
  - id: setup-structure
    content: TRANSKRIPSIYON-RAG-VDB klasor yapisini ve config.py dosyasini olustur
    status: pending
  - id: install-deps
    content: requirements.txt olustur ve conda ortamina gerekli paketleri yukle (chromadb, sentence-transformers)
    status: pending
  - id: sozluk-collection
    content: TID_Sozluk koleksiyonunu olustur - 1933 data.json dosyasini isle ve ChromaDB'ye yukle
    status: pending
  - id: hafiza-collection
    content: TID_Hafiza koleksiyonunu olustur - bos baslat, ornek verilerle seed et
    status: pending
  - id: dual-retriever
    content: Iki asamali retrieval sistemini implement et (hafiza + sozluk)
    status: pending
  - id: prompt-builder
    content: RAG destekli prompt builder olustur
    status: pending
  - id: feedback-handler
    content: Manuel onay ve hafiza guncelleme mekanizmasini implement et
    status: pending
  - id: integration
    content: Mevcut anlamlandirma-sistemi ile entegrasyon adaptorunu yaz
    status: pending
  - id: test-system
    content: End-to-end test scripti yaz ve sistemi dogrula
    status: pending
---

# TID Transkripsiyon RAG Sistemi

## Mimari Ozet

```mermaid
flowchart TB
    subgraph Input [Girdi]
        TR[Transkripsiyon<br/>OKUL GITMEK ISTEMEK]
    end
    
    subgraph RAG [Dual-Collection RAG]
        subgraph Retrieval [Two-Level Retrieval]
            L1[Level 1: Hafiza Yoklamasi]
            L2[Level 2: Kelime Madenciligi]
        end
        
        subgraph Collections [ChromaDB Koleksiyonlari]
            TH[(TID_Hafiza<br/>Dinamik Ceviri Hafizasi)]
            TS[(TID_Sozluk<br/>1933 Kelime)]
        end
        
        L1 --> TH
        L2 --> TS
    end
    
    subgraph Generation [Uretim]
        PB[Prompt Builder]
        LLM[LLM<br/>OpenAI/Gemini]
        OUT[Ceviri Ciktisi]
    end
    
    subgraph Feedback [Geri Besleme]
        MO[Manuel Onay]
        DB[(Hafizaya Kayit)]
    end
    
    TR --> L1
    TR --> L2
    L1 --> PB
    L2 --> PB
    PB --> LLM
    LLM --> OUT
    OUT --> MO
    MO -->|Onaylandi| DB
    DB --> TH
```

## Klasor Yapisi

```
TRANSKRIPSIYON-RAG-VDB/
├── config.py                 # Konfigürasyon (LLM, embedding model, paths)
├── requirements.txt          # Bagimliliklar
├── collections/
│   ├── __init__.py
│   ├── sozluk_collection.py  # TID_Sozluk yonetimi
│   └── hafiza_collection.py  # TID_Hafiza yonetimi
├── retriever/
│   ├── __init__.py
│   └── dual_retriever.py     # Iki asamali retrieval
├── prompt_builder/
│   ├── __init__.py
│   └── augmented_prompt.py   # RAG destekli prompt olusturma
├── feedback/
│   ├── __init__.py
│   └── feedback_handler.py   # Manuel onay ve hafiza guncelleme
├── scripts/
│   ├── init_sozluk.py        # TID_Sozluk koleksiyonunu olustur
│   ├── init_hafiza.py        # Bos TID_Hafiza olustur
│   └── test_rag.py           # Test scripti
├── vectorstore/              # ChromaDB persistent storage
│   └── .gitkeep
└── integration/
    ├── __init__.py
    └── anlamlandirma_adapter.py  # Mevcut sistem entegrasyonu
```

## Temel Bilesen Detaylari

### 1. Konfigürasyon ([TRANSKRIPSIYON-RAG-VDB/config.py](TRANSKRIPSIYON-RAG-VDB/config.py))

```python
# LLM Providers (configurable)
LLM_PROVIDERS = ["openai", "gemini"]
DEFAULT_LLM = "gemini"

# Embedding Model (Turkce destekli)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ChromaDB paths
VECTORSTORE_PATH = "./vectorstore"
SOZLUK_COLLECTION = "tid_sozluk"
HAFIZA_COLLECTION = "tid_hafiza"

# TID_Sozluk_Verileri path
TID_SOZLUK_PATH = "../TID_Sozluk_Verileri"
```

### 2. TID_Sozluk Koleksiyonu (Statik)

- **Kaynak:** [TID_Sozluk_Verileri/](TID_Sozluk_Verileri/) klasorundeki 1933 `data.json` dosyasi
- **Embedding:** Kelime adi (ornegin: "Agac")
- **Metadata:** 
  - `kelime`: Ana kelime
  - `tur`: Kelime turu (Ad, Eylem, Sifat vb.)
  - `aciklama`: Anlam aciklamasi
  - `ornek_transkripsiyon`: Ornek TID cumlesi
  - `ornek_ceviri`: Ornek Turkce ceviri

### 3. TID_Hafiza Koleksiyonu (Dinamik)

- **Embedding:** Tam transkripsiyon cumlesi
- **Metadata:**
  - `transkripsiyon`: Ham TID transkripsiyonu
  - `ceviri`: Onaylanmis Turkce ceviri
  - `provider`: Kullanilan LLM
  - `onay_tarihi`: Onay timestamp'i

### 4. Two-Level Retrieval Akisi

```python
def rag_sorgusu_hazirla(transkripsiyon: str) -> dict:
    # Level 1: Hafiza yoklamasi (benzer cumleler)
    benzer_cumleler = hafiza_collection.query(
        query_texts=[transkripsiyon],
        n_results=3
    )
    
    # Level 2: Kelime madenciligi
    kelimeler = transkripsiyon.split()
    kelime_bilgileri = []
    for kelime in kelimeler:
        sonuc = sozluk_collection.query(
            query_texts=[kelime],
            n_results=1
        )
        kelime_bilgileri.append(sonuc)
    
    return {
        "referans_ceviriler": benzer_cumleler,
        "kelime_detaylari": kelime_bilgileri
    }
```

### 5. Entegrasyon Noktasi

Mevcut [anlamlandirma-sistemi/preprocessor.py](anlamlandirma-sistemi/preprocessor.py) dosyasindaki `create_final_prompt` fonksiyonu yeni RAG sistemini kullanacak sekilde guncellenecek.

### 6. Geri Besleme API'si

Yeni endpoint: `POST /api/approve_translation`

```python
{
    "transkripsiyon": "OKUL GITMEK ISTEMEK",
    "ceviri": "Okula gitmek istiyorum.",
    "provider": "gemini"
}
```

## Bagimliliklar

```
chromadb>=0.4.0
sentence-transformers>=2.2.0
openai>=1.0.0
google-generativeai>=0.3.0
```

## Basari Kriterleri

1. TID_Sozluk koleksiyonunda 1933 kelime vektorize edilmis
2. Transkripsiyon sorgusunda hem hafiza hem sozluk bilgisi cekilebiliyor
3. LLM promptu dinamik RAG baglami ile zenginlestiriliyor
4. Manuel onay ile ceviri hafizaya kaydedilebiliyor
5. Mevcut anlamlandirma-sistemi sorunsuz calisabiliyor