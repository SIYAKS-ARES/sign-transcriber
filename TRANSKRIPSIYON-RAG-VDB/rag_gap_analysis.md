# RAG System Gap Analysis for Academic Thesis

This document details the missing components in the current architectural plan (`opus.plan.md`) required to transform the project from a "working software prototype" into a "defensible academic thesis methodology".

## 1. Quantitative Evaluation Framework (Measurement)
**Status:** ❌ Missing
**Why it's critical:** In a thesis, you cannot just say "it works better." You must prove *how much* better it works compared to a baseline.
**Missing Components:**
*   **Benchmark Module (`src/benchmark.py`):** A script that runs a test set of glosses through the system and calculates scores.
*   **Baseline Comparison:** The system needs a "toggle" to run with and without RAG to generate comparative data:
    *   *Baseline:* Raw LLM translation (Zero-shot).
    *   *Experiment:* RAG-Augmented translation.
*   **Metrics:**
    *   **BLEU Score:** Measures n-gram overlap (Standard in Machine Translation).
    *   **BERTScore / Cosine Similarity:** Measures semantic meaning preservation (Crucial for sign language where structure changes but meaning is kept).

## 2. Interactive Feedback Loop Interface (HCI)
**Status:** ⚠️ Incomplete (API defined, Interaction missing)
**Why it's critical:** The "Iterative" part of your thesis title implies a cycle. A database update API exists, but the *human agent* in this loop needs a tool to view, verify, and correct data.
**Missing Components:**
*   **Review Dashboard (Streamlit/CLI):** A UI that presents:
    1.  Input Gloss
    2.  Retrieved Context (What did the vector DB find?)
    3.  Generated Translation
    4.  **Edit Field:** User corrects the translation.
    5.  **"Save to Learning Memory" Button:** Closes the loop.

## 3. Data Preprocessing & normalizationStrategy
**Status:** ⚠️ Vague
**Why it's critical:** Garbage In, Garbage Out. `TID_Sozluk_Verileri` comes from web scraping and likely contains noise (HTML tags, non-standard punctuation). Raw ingestion will lead to poor vector retrieval.
**Missing Components:**
*   **Cleaning Pipeline:**
    *   Removing non-text artifacts.
    *   Handling "O eş yap" vs "O iş yap" (Synonym normalization).
    *   Lemmatization (Root finding) for finding gloss matches effectively.

## 4. Hyperparameter Justification
**Status:** ❌ Missing
**Why it's critical:** An academic methodology requires justifying *why* certain choices were made.
**Missing Components:**
*   **Distance Metric:** Choice of Cosine Similarity (for semantic direction) vs Euclidean (L2).
*   **Chunking:** The definition of "Context Window Limit". If a word has 5 definitions, do we feed all 5? Or just the top 1? The plan needs to define a configuration for `MAX_CONTEXT_TOKENS`.

## 5. Integration with "Sign-Transcriber" Ecosystem
**Status:** ⚠️ Abstract
**Why it's critical:** The RAG system shouldn't just live in a vacuum; it needs to consume the specific output format of your existing transcription pipeline.
**Missing Components:**
*   **Input Adapter:** Your transcription system might output lists `['OKUL', 'GIT', 'ISTE']` or a string `"OKUL GIT ISTE"`. The RAG entry point needs strict typing to handle the exact output format of the upstream model.

---

## Recommended Action Plan Updates

To address these gaps, we should add the following tasks to the implementation plan:

1.  **[Task] Implement `cleaning.py`**: Robust text normalizer before Vector DB ingestion.
2.  **[Task] Implement `benchmark.py`**: Evaluation script with BLEU/BERTScore libraries.
3.  **[Task] Build `app.py` (Streamlit)**: A visual interface to demonstrate the "Human-in-the-loop" aspect during the thesis defense.
