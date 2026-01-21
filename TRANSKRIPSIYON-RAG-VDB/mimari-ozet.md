flowchart TB
    subgraph Input [Girdi Katmani]
        RAW[Raw Input`<br/>`str/list/dict]
        IA[Input Adapter`<br/>`Strict Typing]
        TR[TranscriptionInput`<br/>`OKUL GITMEK ISTEMEK]
        RAW --> IA --> TR
    end

    subgraph Preprocessing [On Isleme]
        CL[Cleaning Pipeline`<br/>`HTML/Noise Temizleme]
    end

    subgraph RAG [Dual-Collection RAG]
        subgraph Retrieval [Two-Level Retrieval]
            L1[Level 1: Hafiza Yoklamasi`<br/>`top_k=3, threshold=0.5]
            L2[Level 2: Kelime Madenciligi`<br/>`top_k=1 per word]
        end

    subgraph Collections [ChromaDB - Cosine Similarity]
            TH[(TID_Hafiza`<br/>`Dinamik Ceviri Hafizasi)]
            TS[(TID_Sozluk`<br/>`1933 Kelime - Temizlenmis)]
        end

    L1 --> TH
        L2 --> TS
    end

    subgraph Generation [Uretim]
        PB[Prompt Builder`<br/>`MAX_CONTEXT_TOKENS=2000]
        LLM[LLM`<br/>`OpenAI/Gemini]
        OUT[Ceviri Ciktisi]
    end

    subgraph Feedback [Human-in-the-Loop]
        SD[Streamlit Dashboard]
        MO[Manuel Onay/Duzeltme]
        DB[(Hafizaya Kayit)]
    end

    subgraph Evaluation [Degerlendirme]
        BM[Benchmark Module]
        BL[Baseline Zero-shot]
        MT[BLEU / BERTScore]
    end

    CL -.->|init_sozluk.py| TS
    TR --> L1
    TR --> L2
    L1 --> PB
    L2 --> PB
    PB --> LLM
    LLM --> OUT
    OUT --> SD
    SD --> MO
    MO -->|Onaylandi| DB
    DB --> TH

    OUT --> BM
    BL --> BM
    BM --> MT
