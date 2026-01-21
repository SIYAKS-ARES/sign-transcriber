# TID Dictionary-Augmented RAG Architecture

This document outlines the architecture for the "Iterative Dictionary-Augmented Generation with Feedback Loop" system for translating Turkish Sign Language (TID) glosses to Turkish.

## 1. System Overview

The system utilizes a Dual-Collection Vector Database approach to improve translation accuracy by combining:
1.  **Static Dictionary (TID_Sozluk)**: Retrieval of word-level definitions and types.
2.  **Dynamic Memory (TID_Hafiza)**: Retrieval of similar whole-sentence context (Translation Memory).

## 2. Directory Structure

```text
TRANSKRIPSIYON-RAG-VDB/
├── data/
│   └── tid_vector_db/       # ChromaDB Persistent Storage
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── database.py          # ChromaDB manager (Singleton/Wrapper)
│   ├── ingestion.py         # ETL pipeline for TID_Sozluk_Verileri
│   ├── rag_engine.py        # Retrieval and Prompt Construction Logic
│   └── main.py              # CLI / Interactive Session Entry Point
├── README.md
└── requirements.txt
```

## 3. Module Details

### 3.1. `database.py`
- Handles `chromadb` client initialization.
- Manages `tid_sozluk` and `tid_hafiza` collections.
- Provides abstraction methods: `add_word`, `add_sentence`, `query_word`, `query_sentence`.

### 3.2. `ingestion.py`
- Scans `../TID_Sozluk_Verileri` directory.
- Parses `data.json` files.
- Cleans "Explanation" text (removes web scrap artifacts if possible).
- Batches and inserts data into `tid_sozluk`.

### 3.3. `rag_engine.py`
- Implements the 2-Level Retrieval Strategy.
  - **Level 1**: Query `tid_hafiza` with the full gloss sentence.
  - **Level 2**: Tokenize gloss sentence and query `tid_sozluk` for each token.
- Constructs the Context Window (Prompt Engineering).

### 3.4. `main.py`
- Runs the interactive loop:
  1. Input Gloss.
  2. Retrieve Context.
  3. Generate Prompt (Display to user/LLM).
  4. Get Output (Simulated or Real LLM call).
  5. Feedback: User confirms/edits translation -> Save to `tid_hafiza`.

## 4. Implementation Steps

1.  **Environment Setup**: Create `requirements.txt`.
2.  **Database Core**: Implement `src/database.py`.
3.  **Data Ingestion**: Implement `src/ingestion.py` and populate the dictionary.
4.  **RAG Logic**: Implement `src/rag_engine.py`.
5.  **Interactive Loop**: Implement `src/main.py`.
6.  **Verification**: Run a test case (e.g., "AĞAÇ UZUN YAŞAMAK").

## 5. Notes / Assumptions
- The source data is at `../TID_Sozluk_Verileri`.
- We use the default embedding function for now, but the code will allow swapping for `paraphrase-multilingual` later.
- LLM interaction will be mocked or require an API key input in `main.py` if testing live generation is desired. For this stage, we focus on *Prompt Construction* and *retrieval accuracy*.
