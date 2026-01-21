# LGS RAG System with LangChain and Manim

Bu dizin, LGS (Liseye Gecis Sinavi) matematik sorulari ureten RAG (Retrieval-Augmented Generation) sistemini icerir.

## Ozellikler

- **R-01**: Embedding Pipeline - Turkce soru metinleri icin FAISS index
- **R-02**: Filtreleme Sistemi - Alt konu, zorluk, gorsel tipi filtreleme
- **R-03**: Benzer Soru Getirme - Katmanli arama stratejisi
- **R-04**: RAG Cikti Formatlama - Markdown/JSON/Text formatlari
- **R-05**: Prompt Tasarimi - LGS formatina uygun promptlar
- **R-06**: Soru Uretim Pipeline - Tam RAG akisi
- **R-07**: Cesitlilik Mekanizmasi - Tekrar onleme ve stil varyasyonlari
- **R-08**: LangChain Entegrasyonu - Basit RAG pipeline
- **R-09**: Manim Gorsel Uretimi - EBOB/EKOK animasyonlari
- **API**: FastAPI REST Endpoints

## Proje Yapisi

```
06-RAG-WITH-LANGCHAIN/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings
│   ├── langchain_rag.py          # LangChain tabanli RAG
│   │
│   ├── models/                   # Pydantic modeller
│   │   ├── __init__.py
│   │   ├── combination.py        # Kombinasyon modeli
│   │   ├── question.py           # Soru modeli
│   │   ├── config_schema.py      # configs.json semasi
│   │   └── api_models.py         # Request/Response
│   │
│   ├── services/                 # Is mantigi
│   │   ├── __init__.py
│   │   ├── embedding_service.py  # R-01: Embedding Pipeline
│   │   ├── filter_service.py     # R-02: Filtreleme
│   │   ├── retriever.py          # R-03: Benzer Soru Getirme
│   │   ├── output_formatter.py   # R-04: Formatlama
│   │   ├── prompt_builder.py     # R-05: Prompt Tasarimi
│   │   ├── question_generator.py # R-06: Soru Uretim
│   │   ├── diversity_service.py  # R-07: Cesitlilik
│   │   ├── combination_selector.py # Kombinasyon secimi
│   │   └── llm_client.py         # LLM API client
│   │
│   ├── manim_visuals/            # R-09: Gorsel uretim
│   │   ├── __init__.py
│   │   ├── ebob_visualization.py
│   │   └── visual_generator.py
│   │
│   └── api/                      # FastAPI
│       ├── __init__.py
│       └── main.py
│
├── scripts/
│   ├── init_vectorstore.py       # Index olusturma
│   ├── test_generation.py        # Uretim testi
│   └── run_api.py                # API baslatma
│
├── vectorstore/                  # FAISS index
├── data/
│   └── generated_questions/      # Uretilen sorular
├── generated_visuals/            # Manim ciktilari
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Testler
│
├── env.example                   # Ortam degiskenleri sablonu
├── environment.yml               # Miniconda ortami
├── requirements.txt              # Python bagimliliklari
└── README.md
```

## Kurulum

### 1. Python ortami (venv ile)

```bash
cd 06-RAG-WITH-LANGCHAIN

# Sanal ortam olustur (Python 3.11+ onerilir)
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

# Bagimliliklari kur
pip install --upgrade pip
pip install -r requirements.txt
```

> Not: Istersen `environment.yml` ile conda ortami da olusturabilirsin, ancak bu README venv akisini referans alir.

### 2. Ortam Degiskenlerini Ayarla (.env)

```bash
# .env dosyasi olustur
cp env.example .env

# .env dosyasini duzenle
# En azindan su alanlari doldur:
#   GEMINI_API_KEY=...
#   LLM_PROVIDER=gemini
#   LLM_MODEL=gemini-2.5-flash
```

Önemli noktalar:
- `GEMINI_API_KEY` hem `.env`'den hem de sistem ortam degiskeninden okunabilir.
- Eger PowerShell oturumunda `$Env:GEMINI_API_KEY` tanimliysa, **.env'deki degeri ezer**.
- Sadece .env kullanimak istiyorsan, oturumda `GEMINI_API_KEY` ortam degiskeni olmamasina dikkat et.

### 3. Gemini API key'ini test et (opsiyonel ama tavsiye edilir)

```bash
python scripts/test_gemini_key.py
```

Bu script sizden API key, model adi (varsayilan: `gemini-2.5-flash`) ve kisa bir prompt ister ve Gemini'den donen cevabi ve token istatistiklerini yazdirir.

### 4. Vector Store Olustur

```bash
python scripts/init_vectorstore.py
```

## Kullanim

### API Sunucusu

```bash
# API'yi baslat
python scripts/run_api.py

# veya
uvicorn src.api.main:app --reload --port 8000
```

API Dokumantasyonu: http://localhost:8000/docs

### Soru Uretimi Testi

```bash
python scripts/test_generation.py
```

Ek olarak, Gemini modeli ve mevcut modellar hakkinda bilgi almak icin:

```bash
python scripts/list_gemini_models.py
```

### LangChain ile Kullanim

```python
from src.langchain_rag import LangChainRAG

# RAG baslat
rag = LangChainRAG()

# Sorulari yukle
rag.load_questions()

# Chain olustur
rag.create_qa_chain()

# Soru uret
question = rag.generate_question({
    "alt_konu": "ebob_ekok",
    "zorluk": 4,
    "gorsel_tipi": "sematik"
})
print(question)
```

### Programatik Kullanim

```python
import asyncio
from src.services import (
    EmbeddingPipeline,
    FilterService,
    QuestionRetriever,
    QuestionGenerator,
    RAGOutputFormatter,
    PromptBuilder,
    LLMClientFactory,
)

async def generate():
    # Pipeline baslat
    pipeline = EmbeddingPipeline()
    pipeline.load_model()
    pipeline.load_index()
    
    # Servisler
    filter_service = FilterService(list(pipeline.id_to_metadata.values()))
    retriever = QuestionRetriever(pipeline, filter_service)
    llm_client = LLMClientFactory.create()
    
    generator = QuestionGenerator(
        retriever=retriever,
        formatter=RAGOutputFormatter(),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
    )
    
    # Soru uret
    question = await generator.generate_question({
        "alt_konu": "ebob_ekok",
        "zorluk": 3,
        "gorsel_tipi": "yok",
    })
    
    print(question.model_dump())

asyncio.run(generate())
```

## API Endpoints

| Method | Endpoint | Aciklama |
|--------|----------|----------|
| GET | `/` | Ana sayfa |
| GET | `/api/v1/health` | Saglik kontrolu |
| POST | `/api/v1/generate` | Tek soru uret |
| POST | `/api/v1/generate/batch` | Toplu uretim |
| GET | `/api/v1/combinations` | Kombinasyonlari listele |
| GET | `/api/v1/stats` | Istatistikler |

### Ornek Request

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "alt_konu": "ebob_ekok",
      "zorluk": 4
    }
  }'
```

## Manim Gorselleri

```python
from src.manim_visuals import VisualGenerator, EBOBVisualization

# Gorsel uret
generator = VisualGenerator()
video_path = generator.generate_ebob_visual(24, 36)
```

## Gelistirme

```bash
# Test calistir
pytest tests/

# Lint
ruff check src/

# Format
ruff format src/
```

## Bagimlilklar

Ana bagimliliklar:
- `faiss-cpu` - Vector index
- `sentence-transformers` - Turkce embedding modeli
- `google-genai` - Gemini LLM API client'i (su anki varsayilan LLM)
- `openai` (opsiyonel) - Alternatif LLM destegi icin
- `fastapi`, `uvicorn` - API sunucu
- `manim` - Gorsel uretim
- `pydantic`, `pydantic-settings` - Veri modelleri ve konfigurasyon
- `numpy`, `pandas` - Veri isleme

## Lisans

MIT
