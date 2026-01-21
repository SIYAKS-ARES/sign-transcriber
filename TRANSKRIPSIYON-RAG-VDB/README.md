# TID RAG Translation System

**Iterative Dictionary-Augmented Generation with Feedback Loop for Turkish Sign Language Translation**

Turk Isaret Dili (TID) transkripsiyon dizilerini dogal Turkce cumlelere ceviren, RAG (Retrieval-Augmented Generation) tabanli akademik ceviri sistemi.

---

## Icerik

- [Genel Bakis](#genel-bakis)
- [Akademik Baglam](#akademik-baglam)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Modul Detaylari](#modul-detaylari)
- [Kurulum](#kurulum)
- [Kullanim](#kullanim)
- [API Referansi](#api-referansi)
- [Konfigurasyon](#konfigurasyon)
- [Degerlendirme](#degerlendirme)
- [Dosya Yapisi](#dosya-yapisi)

---

## Genel Bakis

Bu sistem, Turk Isaret Dili (TID) gloss transkripsiyon dizilerini Turkce'ye cevirmek icin gelistirilmis bir RAG (Retrieval-Augmented Generation) sistemidir.

### Temel Ozellikler

| Ozellik | Aciklama |
|---------|----------|
| **Dual-Collection RAG** | Statik sozluk (TID_Sozluk) + Dinamik ceviri hafizasi (TID_Hafiza) |
| **TID Dilbilgisi Destegi** | Topic-Comment -> SOV donusumu, NMM cikarimi, zaman tespiti |
| **Hibrit Few-Shot** | Statik ornekler + dinamik Hafiza ornekleri |
| **Coklu Ceviri Alternatifleri** | Her transkripsiyon icin 3 alternatif ceviri |
| **Human-in-the-Loop** | Streamlit dashboard ile manuel onay ve duzeltme |
| **Akademik Metrikler** | BLEU ve BERTScore ile kantitatif degerlendirme |

### Teknik Ozellikler

| Bilesen | Teknoloji |
|---------|-----------|
| Vektor Veritabani | ChromaDB |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 (384 boyut) |
| Mesafe Metrigi | Cosine Similarity |
| LLM Saglayicilar | Google Gemini, OpenAI GPT-4 |
| Dashboard | Streamlit |

---

## Akademik Baglam

### Problem Tanimi

Turk Isaret Dili (TID), Turkce'den farkli bir dilbilgisi yapisina sahiptir:

1. **Soz Dizimi Farki**: TID Topic-Comment yapisi kullanirken, Turkce SOV (Ozne-Nesne-Yuklem) kullanir
2. **Morfolojik Fark**: TID'de ekler (iyelik, hal, zaman) yoktur
3. **NMM Eksikligi**: Gorsel modeller yuz mimiklerini (soru, olumsuzluk) yakalayamaz
4. **Gloss Boslugu**: Transkripsiyon sadece kok kelimeleri icerir

### Cozum Yaklasimi

**Iterative Dictionary-Augmented Generation with Feedback Loop**

```
Girdi: TID Transkripsiyon (gloss dizisi)
  |
  v
[Preprocessing] -> Dilbilgisi analizi (zaman, soru, olumsuzluk, tekrar)
  |
  v
[RAG Retrieval] -> Hafiza (benzer cumleler) + Sozluk (kelime bilgileri)
  |
  v
[Prompt Building] -> System instruction + Few-shot + RAG context
  |
  v
[LLM Generation] -> 3 alternatif ceviri
  |
  v
[Human Feedback] -> Onay/Duzeltme -> Hafiza'ya kayit
  |
  v
Cikti: Dogal Turkce cumle
```

### Literaturdeki Yeri

Bu sistem su yaklasimlardan esinlenmistir:

- **RAG**: Lewis et al., 2020 - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **Translation Memory**: Somers, 2003 - "Translation Memory Systems"
- **Human-in-the-Loop ML**: Monarch, 2021 - "Human-in-the-Loop Machine Learning"

---

## Sistem Mimarisi

### Genel Akis Diyagrami

```
                    +------------------+
                    |  Ham Transkripsiyon  |
                    |  "IKI ABLA VAR..."   |
                    +--------+---------+
                             |
                             v
              +-----------------------------+
              |      TIDPreprocessor        |
              |  - Zaman tespiti            |
              |  - Soru/olumsuzluk          |
              |  - Tekrar (pekistirme)      |
              |  - Ozel belirtec degisimi   |
              +-------------+---------------+
                            |
            +---------------+---------------+
            |                               |
            v                               v
+---------------------+         +---------------------+
|    TID_Hafiza       |         |    TID_Sozluk       |
|  (Dinamik Hafiza)   |         |  (Statik Sozluk)    |
|  2845 ceviri ornegi |         |  2867 kelime        |
+----------+----------+         +----------+----------+
           |                               |
           +---------------+---------------+
                           |
                           v
              +-----------------------------+
              |      Prompt Builder         |
              |  - System instruction       |
              |  - Few-shot ornekler        |
              |  - RAG context              |
              |  - Dilbilgisi ipuclari      |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              |      LLM (Gemini/OpenAI)    |
              |  System instruction ayri    |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              |      Response Parser        |
              |  Regex ile 3 alternatif     |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              |      TranslationResult      |
              |  - best_translation         |
              |  - alternatives[]           |
              |  - confidence scores        |
              +-----------------------------+
```

### Dual-Collection Stratejisi

```
+------------------------------------------------------------------+
|                        ChromaDB                                   |
|                                                                   |
|  +---------------------------+  +---------------------------+    |
|  |      TID_Sozluk           |  |      TID_Hafiza           |    |
|  |  (Statik Sozluk)          |  |  (Dinamik Ceviri Hafizasi)|    |
|  |---------------------------|  |---------------------------|    |
|  | Embedding: Kelime adi     |  | Embedding: Tam cumle      |    |
|  | Metadata:                 |  | Metadata:                 |    |
|  |   - kelime                |  |   - transkripsiyon        |    |
|  |   - tur (Ad/Eylem/...)    |  |   - ceviri                |    |
|  |   - aciklama              |  |   - provider              |    |
|  |   - ornek_transkripsiyon  |  |   - onay_tarihi           |    |
|  |   - ornek_ceviri          |  |                           |    |
|  +---------------------------+  +---------------------------+    |
|                                                                   |
|  Retrieval:                     Retrieval:                       |
|  - Exact match (kelime)         - Semantic similarity            |
|  - Partial match (regex)        - top_k=3, threshold=0.5         |
|  - top_k=1 per word                                               |
+------------------------------------------------------------------+
```

---

## Modul Detaylari

### 1. Preprocessing (`preprocessing/`)

#### TIDPreprocessor

TID'e ozgu dilbilgisi ozelliklerini tespit eder ve isler.

**Dosya:** `preprocessing/tid_preprocessor.py`

**Tespit Edilen Ozellikler:**

| Ozellik | Ornek | Islem |
|---------|-------|-------|
| Zaman (acik) | DUN, YARIN, SIMDI | `detected_tense = "past"/"future"/"present"` |
| Zaman (cikarim) | BITMEK, TAMAM | `_GECMIS_ZAMAN_` belirteci |
| Soru | NEREDE, NE, KIM | `is_question = True` |
| Olumsuzluk | DEGIL, YOK | `_NEGASYON_` belirteci |
| Tekrar | GEZMEK GEZMEK | `GEZMEK_TEKRAR` + `repetitions = {"GEZMEK": 2}` |
| Bilesik kelime | ARABA^SURMEK | `ARABA_SURMEK` |

**Kullanim:**

```python
from preprocessing import TIDPreprocessor

preprocessor = TIDPreprocessor()
result = preprocessor.preprocess("DUN OKUL GITMEK BITMEK")

print(result.processed)        # "DUN OKUL GITMEK _GECMIS_ZAMAN_"
print(result.detected_tense)   # "past"
print(result.tense_source)     # "explicit"
```

---

### 2. RAG Collections (`tid_collections/`)

#### SozlukCollection

Statik TID sozlugu. 2867 kelime iceren `TID_Sozluk_Verileri/` verilerinden olusturulur.

**Dosya:** `tid_collections/sozluk_collection.py`

**Ozellikler:**
- Tam esleme (exact match) oncelikli
- Kismi esleme (partial match) - regex tabanli ("Abi, Agabey" icinden "Abi" bulma)
- Semantik arama (fallback, varsayilan kapali)

**Kullanim:**

```python
from tid_collections import SozlukCollection

sozluk = SozlukCollection()
results = sozluk.query("ABLA")  # Tam esleme
# [{"metadata": {"kelime": "Abla", "tur": "Ad", "aciklama": "...", ...}, "match_type": "exact"}]
```

#### HafizaCollection

Dinamik ceviri hafizasi. Onaylanan ceviriler burada saklanir.

**Dosya:** `tid_collections/hafiza_collection.py`

**Baslangic Verisi:** TID_Sozluk_Verileri'ndeki ornek cumleler (2845 adet)

**Kullanim:**

```python
from tid_collections import HafizaCollection

hafiza = HafizaCollection()

# Benzer cumle ara
results = hafiza.query("OKUL GITMEK ISTEMEK", n_results=3)

# Yeni ceviri ekle
hafiza.add_translation(
    transkripsiyon="BEN OKUL GITMEK",
    ceviri="Okula gittim.",
    provider="gemini"
)
```

---

### 3. Retriever (`retriever/`)

#### DualRetriever

Iki asamali retrieval: Hafiza + Sozluk

**Dosya:** `retriever/dual_retriever.py`

**Akis:**

```
Transkripsiyon: "OKUL GITMEK ISTEMEK"
       |
       +---> [Level 1: Hafiza] -> Benzer cumleler (top_k=3)
       |
       +---> [Level 2: Sozluk] -> Her kelime icin bilgi (top_k=1)
       |
       v
RetrievalResult:
  - similar_translations: [{transkripsiyon, ceviri, similarity}, ...]
  - word_info: {OKUL: [...], GITMEK: [...], ISTEMEK: [...]}
```

**Kullanim:**

```python
from retriever import DualRetriever

retriever = DualRetriever()
result = retriever.retrieve("OKUL GITMEK ISTEMEK")

print(result.to_context_string())  # LLM promptu icin formatlanmis context
```

---

### 4. Prompt Builder (`prompt_builder/`)

#### System Instructions

LLM'e gonderilen TID dilbilgisi kurallari.

**Dosya:** `prompt_builder/system_instructions.py`

**Icerik:**
1. TID Sozdizimi (Topic-Comment -> SOV)
2. NMM Baglamsal Cikarim (Soru/Olumsuzluk)
3. Zaman Kurallari
4. Turkce Morfoloji
5. TID'e Ozgu Yapilar
6. Cikti Formati (3 alternatif)
7. Onemli Uyarilar (Halusinasyon, Belirsizlik, RAG)

#### FewShotBuilder

Dilbilgisi yapisina gore dinamik ornek secimi.

**Dosya:** `prompt_builder/few_shot_builder.py`

**Kategoriler:**
- `topic_comment`: Topic-Comment donusumu ornekleri
- `tense_past`: Gecmis zaman ornekleri
- `tense_future`: Gelecek zaman ornekleri
- `tense_present`: Simdiki zaman ornekleri
- `negation`: Olumsuzluk ornekleri
- `question`: Soru ornekleri
- `repetition`: Pekistirme ornekleri
- `compound`: Bilesik kelime ornekleri
- `general`: Genel ornekler

**Kullanim:**

```python
from prompt_builder import FewShotBuilder

builder = FewShotBuilder()
examples = builder.build_examples(
    detected_tense="past",
    is_question=False,
    is_negative=False,
    repetitions={"GEZMEK": 2},
    hafiza_results=[...]  # Dinamik ornekler
)
```

#### Templates

Kullanici prompt sablonu.

**Dosya:** `prompt_builder/templates.py`

**Cikti Formati:**

```
## ALTERNATIF 1
Ceviri: [ceviri]
Guven: [1-10]/10
Aciklama: [aciklama]

## ALTERNATIF 2
...

## ALTERNATIF 3
...
```

---

### 5. LLM Module (`llm/`)

#### LLMClient

Gemini ve OpenAI icin unified client.

**Dosya:** `llm/llm_client.py`

**Ozellikler:**
- System instruction ayri parametre olarak gonderilir
- Dinamik system instruction guncelleme
- TranslationResult dondurur

**Kullanim:**

```python
from llm import LLMClient

client = LLMClient(provider="gemini")
result = client.translate(user_prompt)

print(result.best.translation)    # En iyi ceviri
print(result.best.confidence)     # Guven puani
print(result.alternatives)        # Tum alternatifler
```

#### ResponseParser

Regex tabanli coklu alternatif parser.

**Dosya:** `llm/response_parser.py`

**Desteklenen Formatlar:**
- `## ALTERNATIF 1` / `## ALTERNATİF 1` / `## Alternatif 1`
- `Ceviri:` / `Çeviri:` / `CEVIRI:`
- `Guven:` / `Güven:` / `GUVEN:`
- `Aciklama:` / `Açıklama:` / `ACIKLAMA:`

**Fallback:** Format uyumsuzlugunda basit regex ile ceviri cikarimi

---

### 6. Pipeline (`pipeline/`)

#### TranslationPipeline

Tum bilesenleri birlestiren orchestrator.

**Dosya:** `pipeline/translation_pipeline.py`

**Akis:**

```python
TranslationPipeline.translate(transcription)
    |
    +---> 1. TIDPreprocessor.preprocess()
    +---> 2. DualRetriever.retrieve()
    +---> 3. FewShotBuilder.build_examples()
    +---> 4. build_dynamic_system_instruction()
    +---> 5. build_user_prompt()
    +---> 6. LLMClient.translate()
    +---> 7. ResponseParser.parse()
    |
    v
PipelineResult
```

**Kullanim:**

```python
from pipeline import TranslationPipeline

pipeline = TranslationPipeline(provider="gemini")
result = pipeline.translate("IKI ABLA VAR EVLENMEK GITMEK")

print(result.best_translation)     # "Iki ablam da evlendi."
print(result.confidence)           # 8
print(result.to_display_string())  # Formatlanmis cikti
```

---

### 7. Feedback (`feedback/`)

Human-in-the-Loop geri bildirim sistemi.

**Dosya:** `feedback/feedback_handler.py`

**Kullanim:**

```python
from feedback import FeedbackHandler

handler = FeedbackHandler()

# Ceviri olustur
feedback = handler.create_feedback(
    transkripsiyon="BEN OKUL GITMEK",
    llm_ceviri="Okula gidiyorum.",
    provider="gemini"
)

# Onayla ve Hafiza'ya kaydet
handler.approve_feedback(
    feedback_id=feedback.feedback_id,
    corrected_ceviri="Okula gittim."  # Duzeltme (opsiyonel)
)
```

---

### 8. Evaluation (`evaluation/`)

Akademik degerlendirme metrikleri.

**Dosyalar:**
- `evaluation/benchmark.py`: BLEU ve BERTScore hesaplama
- `evaluation/baseline.py`: Zero-shot baseline karsilastirma
- `evaluation/test_sets/test_glosses.json`: Test verisi

**Kullanim:**

```python
from evaluation import TranslationBenchmark

benchmark = TranslationBenchmark(rag_system, llm_client)
results = benchmark.run_benchmark(test_set)

print(results["rag"]["bleu"])           # RAG BLEU skoru
print(results["baseline"]["bleu"])      # Baseline BLEU skoru
print(results["rag"]["bertscore_f1"])   # RAG BERTScore
```

---

### 9. Integration (`integration/`)

Mevcut sistemlerle entegrasyon.

**Dosyalar:**
- `integration/input_adapter.py`: Farkli girdi formatlarini standartlastirma
- `integration/anlamlandirma_adapter.py`: `anlamlandirma-sistemi` entegrasyonu

---

## Kurulum

### Gereksinimler

- Python 3.9+
- Conda environment (onerilen)

### Adimlar

```bash
# 1. Environment olustur
conda create -n sign-transcriber python=3.10
conda activate sign-transcriber

# 2. Bagimliliklari yukle
cd TRANSKRIPSIYON-RAG-VDB
pip install -r requirements.txt

# 3. Sozluk koleksiyonunu olustur
python scripts/init_sozluk.py

# 4. Hafiza koleksiyonunu olustur
python scripts/init_hafiza.py

# 5. Testleri calistir
python scripts/test_rag.py
python scripts/test_linguistic.py
```

### API Anahtarlari

```bash
# Gemini icin
export GOOGLE_API_KEY="your-api-key"

# OpenAI icin
export OPENAI_API_KEY="your-api-key"
```

---

## Kullanim

### Basit Kullanim

```python
from pipeline import TranslationPipeline

# Pipeline olustur
pipeline = TranslationPipeline(provider="gemini")

# Ceviri yap
result = pipeline.translate("DUN ARKADAS BULUSMAK KAHVE ICMEK")

# Sonuclari goster
print(f"Ceviri: {result.best_translation}")
print(f"Guven: {result.confidence}/10")

# Tum alternatifler
for alt in result.translation_result.alternatives:
    print(f"  [{alt.confidence}/10] {alt.translation}")
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

Dashboard ozellikleri:
- Transkripsiyon girdisi
- RAG context gorsellestirme
- LLM ceviri
- Geri bildirim (onay/duzeltme)
- Sistem istatistikleri

### Benchmark Calistirma

```bash
python scripts/run_benchmark.py
```

---

## API Referansi

### TranslationPipeline

```python
class TranslationPipeline:
    def __init__(self, provider: str = "gemini", llm_config: LLMConfig = None)
    def translate(self, transcription: str) -> PipelineResult
    def translate_simple(self, transcription: str) -> str
    def get_stats(self) -> dict
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    translation_result: TranslationResult
    preprocessed: PreprocessedInput
    retrieval_result: RetrievalResult
    system_instruction: str
    user_prompt: str
    provider: str
    
    @property
    def best_translation(self) -> str
    @property
    def confidence(self) -> int
    @property
    def is_successful(self) -> bool
    def to_display_string(self) -> str
```

### TIDPreprocessor

```python
class TIDPreprocessor:
    def preprocess(self, transcription: str) -> PreprocessedInput
```

### PreprocessedInput

```python
@dataclass
class PreprocessedInput:
    original: str
    processed: str
    word_list: List[str]
    detected_tense: Optional[str]  # "past", "present", "future", None
    tense_source: Optional[str]    # "explicit", "inferred", None
    is_question: bool
    is_negative: bool
    repetitions: Dict[str, int]
    compound_words: List[str]
    linguistic_hints: Dict[str, any]
```

---

## Konfigurasyon

### config.py Parametreleri

| Parametre | Deger | Aciklama |
|-----------|-------|----------|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Turkce destekli multilingual model |
| `EMBEDDING_DIMENSION` | 384 | Vektor boyutu |
| `DISTANCE_METRIC` | `cosine` | Semantik benzerlik icin |
| `HAFIZA_TOP_K` | 3 | Hafiza'dan alinacak ornek sayisi |
| `SOZLUK_TOP_K` | 1 | Kelime basina sozluk sonucu |
| `SIMILARITY_THRESHOLD` | 0.5 | Minimum benzerlik esigi |
| `MAX_CONTEXT_TOKENS` | 2000 | Prompt icin maksimum token |

### Hyperparameter Justification

**Embedding Model Secimi:**
- `paraphrase-multilingual-MiniLM-L12-v2` secildi cunku:
  - 50+ dil destegi (Turkce dahil)
  - Sentence-level semantic similarity icin optimize
  - 384 boyut: hiz/kalite dengesi
- Alternatif: `multilingual-e5-large` (daha iyi ama yavas)

**Distance Metric:**
- Cosine similarity secildi cunku:
  - Vektor yonelimini olcer (semantik benzerlik)
  - Vektor buyuklugunden bagimsiz
  - NLP/embedding alaninda standart

---

## Degerlendirme

### Benchmark Sonuclari (Ocak 2026)

| Metrik | Deger | Aciklama |
|--------|-------|----------|
| **BLEU Score** | 25.51 | N-gram tabanli ceviri kalitesi |
| **BERTScore F1** | 0.847 | Semantik benzerlik |
| **Exact Match** | 66.7% | Birebir eslesen ceviriler |
| **Ortalama Guven** | 9.92/10 | LLM ozdegerlendirmesi |

### Test Sonuclari

```
Linguistic Tests: 31/31 (100%)
RAG Tests: 7/7 (100%)
Benchmark: 12 basarili ceviri / 50 test ornegi
```

### Detayli Raporlar

- **Turkce Rapor**: `evaluation/BENCHMARK_RAPORU.md`
- **English Report**: `evaluation/BENCHMARK_REPORT_EN.md`
- **Ham Veriler**: `evaluation/benchmark_results_corrected.json`

### Literatur Karsilastirmasi

| Calisma | Dil Cifti | BLEU |
|---------|-----------|------|
| Camgoz et al., 2018 | DGS->DE | 18.40 |
| Camgoz et al., 2020 | DGS->DE | 24.54 |
| Yin & Read, 2020 | ASL->EN | 21.80 |
| **Bu calisma** | **TID->TR** | **25.51** |

### Baseline Karsilastirma

Sistem, RAG olmadan (zero-shot) ve RAG ile karsilastirma yapabilir:

```python
from evaluation import TranslationBenchmark

benchmark = TranslationBenchmark(rag_system, llm_client)
results = benchmark.run_benchmark(test_set)

# RAG vs Baseline
print(f"RAG BLEU: {results['rag']['bleu']}")
print(f"Baseline BLEU: {results['baseline']['bleu']}")
```

---

## Dosya Yapisi

```
TRANSKRIPSIYON-RAG-VDB/
├── config.py                      # Konfigurasyon ve hyperparameter
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Bagimliliklar
├── README.md                      # Bu dosya
│
├── preprocessing/                 # On isleme modulu
│   ├── __init__.py
│   ├── cleaning.py               # Veri temizleme (HTML, boilerplate)
│   └── tid_preprocessor.py       # TID dilbilgisi analizi (442 satir)
│
├── tid_collections/              # ChromaDB koleksiyonlari
│   ├── __init__.py
│   ├── sozluk_collection.py      # Statik sozluk (356 satir)
│   └── hafiza_collection.py      # Dinamik hafiza (248 satir)
│
├── retriever/                    # RAG retrieval
│   ├── __init__.py
│   └── dual_retriever.py         # Iki asamali retrieval (321 satir)
│
├── prompt_builder/               # Prompt olusturma
│   ├── __init__.py
│   ├── system_instructions.py    # TID dilbilgisi kurallari (257 satir)
│   ├── few_shot_builder.py       # Dinamik ornek secimi (392 satir)
│   ├── templates.py              # Prompt sablonlari (235 satir)
│   └── augmented_prompt.py       # Legacy prompt builder (169 satir)
│
├── llm/                          # LLM modulu
│   ├── __init__.py
│   ├── llm_client.py             # Gemini/OpenAI client (301 satir)
│   └── response_parser.py        # Regex parser (279 satir)
│
├── pipeline/                     # Pipeline orchestrator
│   ├── __init__.py
│   └── translation_pipeline.py   # Tam pipeline (345 satir)
│
├── feedback/                     # Human-in-the-Loop
│   ├── __init__.py
│   └── feedback_handler.py       # Geri bildirim yonetimi (238 satir)
│
├── evaluation/                   # Akademik degerlendirme
│   ├── __init__.py
│   ├── benchmark.py              # BLEU/BERTScore (258 satir)
│   ├── baseline.py               # Zero-shot baseline (73 satir)
│   └── test_sets/
│       └── test_glosses.json     # Test verisi
│
├── integration/                  # Entegrasyon adapterleri
│   ├── __init__.py
│   ├── input_adapter.py          # Girdi standartlastirma (157 satir)
│   └── anlamlandirma_adapter.py  # Mevcut sistem entegrasyonu (172 satir)
│
├── scripts/                      # Yardimci scriptler
│   ├── init_sozluk.py            # Sozluk koleksiyonu olusturma
│   ├── init_hafiza.py            # Hafiza koleksiyonu olusturma
│   ├── test_rag.py               # RAG end-to-end testleri
│   ├── test_linguistic.py        # Dilbilgisi testleri
│   └── run_benchmark.py          # Benchmark calistirma
│
├── vectorstore/                  # ChromaDB persistent storage
│   └── chroma.sqlite3
│
├── archive/                      # Eski/deneysel kodlar
│
└── ornek_prompt.md               # Ornek prompt dokumantasyonu
```

### Toplam Kod Satiri

```
Toplam: ~4600 satir Python kodu
```

---

## Makale Yazimi Icin Notlar

### Metodoloji Bolumu

1. **Veri Kaynagi**: TID_Sozluk_Verileri (1933 kelime, web scraping)
2. **Preprocessing**: HTML temizleme, gloss normalizasyon, TID dilbilgisi analizi
3. **RAG Stratejisi**: Dual-collection (statik sozluk + dinamik hafiza)
4. **Prompt Engineering**: System instruction + few-shot + RAG context
5. **Human-in-the-Loop**: Manuel onay ile iteratif ogrenme

### Sonuclar Bolumu

1. Kantitatif metrikler: BLEU, BERTScore
2. RAG vs Baseline karsilastirma
3. Per-class analiz (kelime bazinda)
4. Hata analizi (confusion patterns)

### Tartisma Bolumu

1. TID'e ozgu zorluklar (NMM eksikligi, Topic-Comment)
2. RAG'in katkilari
3. Few-shot learning etkisi
4. Sistem limitleri ve gelecek calisma

---

## Lisans

Bu proje akademik arastirma amacli gelistirilmistir.

## Iletisim

Sorular ve oneriler icin GitHub Issues kullaniniz.
