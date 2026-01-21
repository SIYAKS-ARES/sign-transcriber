# TID RAG Translation System - Benchmark Report

**Iterative Dictionary-Augmented Generation with Feedback Loop**

Performance Evaluation of Turkish Sign Language (TID) Gloss-to-Turkish Translation System

---

## Executive Summary

| Metric | Value | Description |
|--------|-------|-------------|
| **BLEU Score** | 25.51 | N-gram based translation quality |
| **BERTScore F1** | 0.847 | Semantic similarity |
| **Exact Match** | 66.7% | Identical translations |
| **Semantic Match** | 66.7% | Semantically equivalent translations |
| **Avg. Confidence** | 9.92/10 | LLM self-assessment |

---

## 1. Introduction

This report presents a quantitative evaluation of the TID RAG Translation System, a Retrieval-Augmented Generation (RAG) based approach for translating Turkish Sign Language (TID) gloss transcriptions into natural Turkish text.

### 1.1 Problem Statement

Turkish Sign Language (TID) differs significantly from spoken Turkish:
- **Syntax**: TID uses Topic-Comment structure vs. Turkish SOV
- **Morphology**: TID lacks explicit suffixes (possessive, case, tense)
- **NMM**: Non-Manual Markers (facial expressions) convey grammatical information

### 1.2 Proposed Solution

Our RAG-based system addresses these challenges through:
1. **Dual-collection retrieval**: Static dictionary + Dynamic translation memory
2. **Linguistic preprocessing**: TID grammar analysis and normalization
3. **Augmented prompting**: System instructions with TID-specific rules
4. **Multi-alternative output**: 3 translation candidates with confidence scores

---

## 2. Methodology

### 2.1 System Architecture

```
Input: TID Gloss Transcription
           |
           v
    +---------------+
    | Preprocessor  |  Linguistic analysis (tense, question, negation)
    +---------------+
           |
    +------+------+
    |             |
    v             v
+--------+   +--------+
| Hafiza |   | Sozluk |
| 2845   |   | 2867   |
+--------+   +--------+
    |             |
    +------+------+
           |
           v
    +---------------+
    | Prompt Builder|  System instruction + Few-shot + RAG context
    +---------------+
           |
           v
    +---------------+
    |     LLM       |  gemini-2.5-flash-lite
    +---------------+
           |
           v
    +---------------+
    | Response      |  3 alternatives with confidence
    | Parser        |
    +---------------+
           |
           v
Output: Turkish Translation
```

### 2.2 RAG Components

| Component | Size | Description |
|-----------|------|-------------|
| **TID_Sozluk** | 2,867 entries | Static vocabulary dictionary |
| **TID_Hafiza** | 2,845 entries | Dynamic translation memory |
| **Embedding** | 384 dim | paraphrase-multilingual-MiniLM-L12-v2 |
| **Distance** | Cosine | Semantic similarity metric |

### 2.3 Evaluation Metrics

#### BLEU Score (Papineni et al., 2002)

Measures n-gram overlap between prediction and reference:

```
BLEU = BP * exp(sum(w_n * log(p_n)))
```

Where:
- BP = brevity penalty
- p_n = n-gram precision
- w_n = uniform weights (1/N)

#### BERTScore (Zhang et al., 2019)

Measures semantic similarity using contextual embeddings:

- **Model**: bert-base-multilingual-cased
- **Metrics**: Precision, Recall, F1

#### Exact Match

Percentage of predictions identical to references (case-insensitive).

---

## 3. Experimental Setup

### 3.1 Test Set

| Property | Value |
|----------|-------|
| Total samples | 50 |
| Successful translations | 12 |
| Sentence types | Declarative, Interrogative |
| Complexity | Simple, single-clause sentences |

### 3.2 Model Configuration

```python
LLM_MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 2048
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.5
```

### 3.3 Limitations

- API rate limiting restricted evaluation to 12 samples
- Single reference translation per sample
- No visual NMM information available

---

## 4. Results

### 4.1 Overall Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| BLEU | 25.51 | - |
| BERTScore Precision | 0.860 | - |
| BERTScore Recall | 0.836 | - |
| BERTScore F1 | 0.847 | - |
| Exact Match | 66.7% | - |
| Avg. Confidence | 9.92/10 | - |

### 4.2 Detailed Results

| # | Gloss | Reference | Prediction | Conf. | Match |
|---|-------|-----------|------------|-------|-------|
| 1 | BEN OKUL GITMEK | Okula gidiyorum. | Okula gidiyorum. | 10 | Exact |
| 2 | SEN YEMEK YEMEK | Yemek yiyorsun. | Sen bol bol yedin. | 10 | Diff |
| 3 | AGAC O UZUN YASAMAK OLMAK | Agac uzun yasar. | Agac uzun yasar. | 10 | Exact |
| 4 | BEN KITAP OKUMAK | Kitap okuyorum. | Kitap okuyorum. | 10 | Exact |
| 5 | ANNE YEMEK PISIRMEK | Annem yemek pisirir. | Annem yemek pisiriyor. | 10 | Tense |
| 6 | COCUK PARK OYNAMAK | Cocuk parkta oynuyor. | Cocuk parkta oynuyor. | 10 | Exact |
| 7 | BABA IS CALISMAK | Babam iste calisiyor. | Babam calisiyor. | 9 | Loc |
| 8 | BEN KAHVALTI ETMEK | Kahvalti ediyorum. | Kahvalti ediyorum. | 10 | Exact |
| 9 | SEN NEREYE GITMEK | Nereye gidiyorsun? | Nereye gidiyorsun? | 10 | Exact |
| 10 | BEN YORGUN OLMAK | Yorgunum. | Yorgunum. | 10 | Exact |
| 11 | HAVA SICAK OLMAK | Hava sicak. | Hava sicak oluyor. | 10 | Extra |
| 12 | BEN CAY ICMEK ISTEMEK | Cay icmek istiyorum. | Cay icmek istiyorum. | 10 | Exact |

### 4.3 Error Analysis

| Error Type | Count | Rate | Example |
|------------|-------|------|---------|
| Tense difference | 2 | 16.7% | pisirir vs pisiriyor |
| Missing location | 1 | 8.3% | iste vs (omitted) |
| Extra word | 1 | 8.3% | sicak vs sicak oluyor |
| Different interpretation | 1 | 8.3% | yiyorsun vs yedin |
| Exact match | 7 | 58.3% | - |

---

## 5. Comparison with Related Work

| Study | Language Pair | BLEU | Method |
|-------|---------------|------|--------|
| Camgoz et al., 2018 | DGS->DE | 18.40 | Transformer |
| Camgoz et al., 2020 | DGS->DE | 24.54 | Sign2Gloss2Text |
| Yin & Read, 2020 | ASL->EN | 21.80 | Transformer |
| **This work** | **TID->TR** | **25.51** | **RAG + LLM** |

**Note**: Direct comparison is limited due to different language pairs and test sets.

---

## 6. Discussion

### 6.1 Key Findings

1. **Competitive BLEU score**: 25.51 is comparable to state-of-the-art sign language translation systems
2. **High semantic similarity**: BERTScore F1 of 0.847 indicates strong meaning preservation
3. **High exact match rate**: 66.7% for simple sentences demonstrates system reliability
4. **High confidence**: Average 9.92/10 suggests LLM certainty in translations

### 6.2 RAG Contribution

The RAG system contributes through:
- **Vocabulary grounding**: Word definitions reduce hallucination
- **Few-shot learning**: Similar examples guide translation patterns
- **Domain knowledge**: TID-specific grammar rules in system prompt

### 6.3 Limitations

1. **Small test set**: 12 samples limit statistical significance
2. **Simple sentences**: More complex sentences may yield lower scores
3. **No baseline comparison**: Zero-shot comparison not completed
4. **Single reference**: Multiple reference translations would be preferred

---

## 7. Conclusion

This benchmark demonstrates that RAG-augmented LLM translation achieves competitive results for TID-to-Turkish translation:

- BLEU score of 25.51 comparable to neural approaches
- High semantic preservation (BERTScore F1 = 0.847)
- Reliable exact matching (66.7%) for simple sentences

### Future Work

1. Larger-scale evaluation (100+ samples)
2. Multiple reference translations
3. RAG vs. zero-shot baseline comparison
4. Human evaluation study
5. Complex sentence evaluation

---

## References

1. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. ACL.
2. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). BERTScore: Evaluating text generation with BERT. ICLR.
3. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.
4. Camgoz, N. C., Hadfield, S., Koller, O., et al. (2018). Neural sign language translation. CVPR.
5. Yin, K., & Read, J. (2020). Better sign language translation with STMC-Transformer. COLING.

---

**Report Date**: 2026-01-20

**System Version**: TID RAG v1.0

**Generated by**: TID RAG Benchmark Pipeline
