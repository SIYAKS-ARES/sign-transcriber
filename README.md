# CODA — Turkish Sign Language Translator (SignTranscriber)

**CODA** is a graduation capstone that turns Turkish Sign Language (TİD / TID) video into gloss labels, then translates those glosses into natural Turkish with an LLM plus dual-collection RAG.

This repository is the public codebase for that pipeline: recognition model, gloss-to-Turkish RAG system, demo app, documentation, and paper drafts.

---

## Problem

Turkish Sign Language is not a word-for-word encoding of spoken Turkish. It has its own syntax (often topic–comment / SOV-leaning gloss order), limited morphological marking in gloss transcriptions, and meaning carried by non-manual markers that text glosses omit.

End-to-end neural translation is hard under limited labeled TİD data. CODA therefore splits the problem:

1. **Recognize** signs from video → gloss sequence  
2. **Translate** gloss → fluent Turkish using retrieval (dictionary + translation memory) and an LLM

---

## Architecture

```
Video / camera frames
        │
        ▼
MediaPipe Holistic keypoints (258-dim / frame)
        │
        ▼
Temporal Transformer (226 AUTSL classes)  →  gloss sequence
        │
        ▼
Linguistic preprocessing (tense, negation, question, repetition)
        │
        ├──────────────► TID_Sozluk (dictionary retrieval)
        └──────────────► TID_Hafiza (translation-memory / few-shot)
        │
        ▼
Augmented prompt → LLM (e.g. Gemini) → Turkish sentence(s)
```

Architecture diagrams in-repo: [`TRANSKRIPSIYON-RAG-VDB/mimari.png`](./TRANSKRIPSIYON-RAG-VDB/mimari.png), [`TRANSKRIPSIYON-RAG-VDB/prompt-pipeline.png`](./TRANSKRIPSIYON-RAG-VDB/prompt-pipeline.png).

| Stage | Role | Primary location |
|-------|------|------------------|
| Recognition | Video → gloss | [`transformer-signlang/`](./transformer-signlang/) |
| RAG + LLM translation | Gloss → Turkish | [`TRANSKRIPSIYON-RAG-VDB/`](./TRANSKRIPSIYON-RAG-VDB/) |
| Integrated demo (Flask) | Camera/video UI + model + optional RAG | [`anlamlandirma-sistemi/`](./anlamlandirma-sistemi/) |

---

## Repository structure

Paths below exist in this tree:

```
.
├── transformer-signlang/                 # Temporal Transformer (AUTSL, 226 classes)
│   ├── train.py, evaluate.py
│   ├── checkpoints/                      # best_model.pth, last_model.pth
│   └── results/evaluation_report.json    # test metrics
├── TRANSKRIPSIYON-RAG-VDB/               # Dual-RAG gloss→Turkish + Streamlit app
│   ├── pipeline/, retriever/, prompt_builder/, llm/
│   ├── scripts/init_sozluk.py, init_hafiza.py
│   ├── evaluation/                       # BLEU / BERTScore artifacts
│   └── app.py                            # streamlit run app.py
├── anlamlandirma-sistemi/                # Flask demo (recognition + LLM/RAG)
│   ├── app.py                            # default http://localhost:5005
│   ├── rag/, experiments/, models/
│   └── DEMO_CALISTIRMA.md, DENEY_KILAVUZU.md
├── transcription-interpretation-system/  # Earlier Flask demo variant
├── tid_scraper/                          # Scraper for TİD Güncel Sözlük
├── character-pipeline/                   # Early static hand-shape experiments
├── RAG-WITH-LANGCHAIN-EXAMPLE/           # LangChain / RAG reference example
├── Docs/                                 # Design notes (transcription & translation)
├── makale/                               # Paper / write-up drafts
├── requirements.txt                      # Root ML/CV dependencies (transformer-oriented)
├── synthetic_generator.py, validate_grammar.py
└── LICENSE                               # MIT
```

Dictionary dump directories such as `TID_Sozluk_Verileri/` are **produced by** `tid_scraper` and are used when initializing ChromaDB; they are not shipped in this repository.

---

## Setup and run

Requires **Python 3.10+** (3.9+ for the RAG package). Create a virtualenv, then install the module(s) you need.

```bash
git clone https://github.com/SIYAKS-ARES/sign-transcriber.git
cd sign-transcriber
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### A) Integrated demo (recognition + translation UI)

```bash
cd anlamlandirma-sistemi
pip install -r requirements.txt
# Set at least one LLM key (loaded via python-dotenv / environment):
export GEMINI_API_KEY="your_key"
python app.py
```

Open `http://localhost:5005/` (demo: `/demo`). Host/port: `HOST` / `PORT` (default `0.0.0.0:5005`).

### B) Gloss→Turkish RAG system (standalone)

```bash
cd TRANSKRIPSIYON-RAG-VDB
pip install -r requirements.txt
# Build Chroma collections (needs local dictionary data; see tid_scraper)
python scripts/init_sozluk.py
python scripts/init_hafiza.py
export GOOGLE_API_KEY="your_key"   # or OPENAI_API_KEY
streamlit run app.py
```

### C) Train / evaluate the recognition model

```bash
cd transformer-signlang
pip install -r requirements.txt
# AUTSL-style video + label layout required for training; see that folder's README
python train.py
python evaluate.py
```

Optional: root [`requirements.txt`](./requirements.txt) covers core PyTorch / MediaPipe deps used by the transformer stack.

---

## Results

Figures below are taken from artifacts in this repository. They are not inflated.

### Sign recognition (Transformer)

Source: [`transformer-signlang/results/evaluation_report.json`](./transformer-signlang/results/evaluation_report.json)

| Metric | Value |
|--------|-------|
| Test accuracy | **76.96%** (`0.7696`) |
| Macro F1 | 0.762 |
| Weighted F1 | 0.764 |
| Classes | 226 (AUTSL) |
| Test samples (support sum) | 3,742 |

Supporting plots/CSVs live under [`transformer-signlang/results/`](./transformer-signlang/results/).

### Gloss→Turkish (Dual-RAG + LLM)

Sources: [`TRANSKRIPSIYON-RAG-VDB/evaluation/benchmark_results_corrected.json`](./TRANSKRIPSIYON-RAG-VDB/evaluation/benchmark_results_corrected.json), [`full_benchmark_results.json`](./TRANSKRIPSIYON-RAG-VDB/evaluation/full_benchmark_results.json), [`BENCHMARK_REPORT_EN.md`](./TRANSKRIPSIYON-RAG-VDB/evaluation/BENCHMARK_REPORT_EN.md)

**Caveat:** The test set has **50** gloss–reference pairs. API rate limiting meant only **12 of 50** cases produced successful translations. Headline BLEU / BERTScore numbers below are computed on those **12 successful** outputs only. Metrics over all 50 (failures counted as empty/failed) are much lower.

| Scope | Samples | BLEU | BERTScore F1 | Exact match | Avg. confidence |
|-------|---------|------|--------------|-------------|-----------------|
| Successful only (`benchmark_results_corrected.json`) | **12 / 50** | **25.51** | **0.847** | 66.7% | 9.92 / 10 |
| Full set including failures (`full_benchmark_results.json`) | 50 | 2.09 | 0.203 | 8.0% | 2.38 / 10 |

Model recorded in the artifacts: `gemini-2.5-flash-lite`. Collections sized at evaluation time: Sozluk 2,867 · Hafiza 2,845.

---

## Documentation and paper drafts

| Resource | Path |
|----------|------|
| Transcription system design notes | [`Docs/Gemini-1-Isaret-Dili-Transkripsiyon-Sistemi-Tasarimi.md`](./Docs/Gemini-1-Isaret-Dili-Transkripsiyon-Sistemi-Tasarimi.md) |
| Translation experimental design notes | [`Docs/Gemini-2-Isaret-Dili-Ceviri-Sistemi-Deneysel-Kurgusu.md`](./Docs/Gemini-2-Isaret-Dili-Ceviri-Sistemi-Deneysel-Kurgusu.md) |
| Prompt notes | [`Docs/Prompts/`](./Docs/Prompts/) |
| Paper / write-up drafts | [`makale/`](./makale/) (see [`makale/master.md`](./makale/master.md)) |
| RAG benchmark reports | [`TRANSKRIPSIYON-RAG-VDB/evaluation/BENCHMARK_REPORT_EN.md`](./TRANSKRIPSIYON-RAG-VDB/evaluation/BENCHMARK_REPORT_EN.md), [`BENCHMARK_RAPORU.md`](./TRANSKRIPSIYON-RAG-VDB/evaluation/BENCHMARK_RAPORU.md) |
| Demo / experiment guides | [`anlamlandirma-sistemi/DEMO_CALISTIRMA.md`](./anlamlandirma-sistemi/DEMO_CALISTIRMA.md), [`DENEY_KILAVUZU.md`](./anlamlandirma-sistemi/DENEY_KILAVUZU.md) |

Module-level READMEs under each top-level folder go deeper on APIs and file layouts.

---

## Acknowledgements

- **AUTSL** — isolated sign recognition dataset used for the 226-class Transformer evaluation  
- **TİD Güncel Sözlük** (Republic of Türkiye, Ministry of Family and Social Services) — dictionary content harvested via [`tid_scraper/`](./tid_scraper/) for RAG collections  
- **MediaPipe Holistic** — pose, face, and hand keypoints  
- **ChromaDB** and **sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) — vector retrieval  

Licensed under the [MIT License](./LICENSE). Copyright (c) 2025 Mehmet Said Hüseyinoğlu.
