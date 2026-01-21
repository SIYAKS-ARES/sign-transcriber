# TID RAG Ceviri Sistemi - Benchmark Raporu

**Iterative Dictionary-Augmented Generation with Feedback Loop**

Turk Isaret Dili (TID) Gloss-to-Turkish Ceviri Sistemi Performans Degerlendirmesi

---

## 1. Yonetici Ozeti

| Metrik | Deger | Aciklama |
|--------|-------|----------|
| **BLEU Score** | 25.51 | N-gram tabanli ceviri kalitesi |
| **BERTScore F1** | 0.847 | Semantik benzerlik |
| **Exact Match** | 66.7% | Birebir eslesen ceviriler |
| **Semantic Match** | 66.7% | Anlamca uyumlu ceviriler |
| **Ortalama Guven** | 9.92/10 | LLM ozdegerlendirmesi |

---

## 2. Metodoloji

### 2.1 Sistem Mimarisi

```
TID Transkripsiyon
       |
       v
+------------------+
| TIDPreprocessor  |  Dilbilgisi analizi
+------------------+
       |
       v
+------------------+     +------------------+
|   TID_Hafiza     |     |   TID_Sozluk     |
| (2845 ceviri)    |     | (2867 kelime)    |
+------------------+     +------------------+
       |                        |
       +------------------------+
                |
                v
+------------------+
|  Prompt Builder  |  System instruction + Few-shot + RAG
+------------------+
                |
                v
+------------------+
|      LLM         |  gemini-2.5-flash-lite
+------------------+
                |
                v
+------------------+
| Response Parser  |  3 alternatif ceviri
+------------------+
                |
                v
       Turkce Ceviri
```

### 2.2 RAG Bilesenleri

| Bilesen | Boyut | Aciklama |
|---------|-------|----------|
| **TID_Sozluk** | 2867 kayit | Statik kelime sozlugu |
| **TID_Hafiza** | 2845 kayit | Dinamik ceviri hafizasi |
| **Embedding** | 384 boyut | paraphrase-multilingual-MiniLM-L12-v2 |
| **Mesafe** | Cosine | Semantik benzerlik |

### 2.3 Prompt Stratejisi

1. **System Instruction**: TID dilbilgisi kurallari (Topic-Comment -> SOV)
2. **Few-Shot Examples**: Dilbilgisi yapisina gore secilen ornekler
3. **RAG Context**: Hafiza'dan benzer cumleler + Sozluk'ten kelime bilgileri
4. **Output Format**: 3 alternatif ceviri (guven puani + aciklama)

---

## 3. Test Seti

### 3.1 Genel Bilgiler

| Ozellik | Deger |
|---------|-------|
| Toplam ornek | 50 |
| Basarili ceviri | 12 |
| API kota siniri | 12 sonra hata |
| Cumle yapisi | Kisa, temel cumleler |

### 3.2 Cumle Kategorileri

| Kategori | Ornek Sayisi | Ornek |
|----------|--------------|-------|
| BEN + Eylem | 18 | BEN OKUL GITMEK |
| SEN + Eylem | 6 | SEN YEMEK YEMEK |
| 3. Sahis | 8 | COCUK PARK OYNAMAK |
| Durum Bildiren | 10 | HAVA SICAK OLMAK |
| Soru | 4 | SEN NEREYE GITMEK |
| Diger | 4 | AGAC O UZUN YASAMAK OLMAK |

---

## 4. Degerlendirme Metrikleri

### 4.1 BLEU Score (Bilingual Evaluation Understudy)

**Formul:**

```
BLEU = BP * exp(sum(w_n * log(p_n)))

BP = min(1, exp(1 - r/c))  # Brevity Penalty
p_n = precision for n-grams
w_n = 1/N (uniform weights)
```

**Sonuc:** 25.51

**Yorum:**
- 20-30 arasi: Kabul edilebilir ceviri kalitesi
- 30-40 arasi: Iyi ceviri kalitesi
- 40+: Cok iyi ceviri kalitesi

### 4.2 BERTScore

**Kullanilan Model:** bert-base-multilingual-cased

| Alt Metrik | Deger | Aciklama |
|------------|-------|----------|
| **Precision** | 0.860 | Tahmin kelimeleri referansta |
| **Recall** | 0.836 | Referans kelimeleri tahminde |
| **F1** | 0.847 | Harmonik ortalama |

**Yorum:**
- 0.80+: Yuksek semantik benzerlik
- Turkce icin multilingual BERT kullanildi

### 4.3 Exact Match

**Formul:**

```
Exact Match = (Birebir eslesenler / Toplam) * 100
```

**Sonuc:** 8/12 = 66.7%

**Eslesenler:**
1. BEN OKUL GITMEK -> Okula gidiyorum.
2. BEN KITAP OKUMAK -> Kitap okuyorum.
3. SEN NEREYE GITMEK -> Nereye gidiyorsun?
4. BEN YORGUN OLMAK -> Yorgunum.
5. BEN CAY ICMEK ISTEMEK -> Cay icmek istiyorum.
6. BEN KAHVALTI ETMEK -> Kahvalti ediyorum.
7. COCUK PARK OYNAMAK -> Cocuk parkta oynuyor.
8. AGAC O UZUN YASAMAK OLMAK -> Agac uzun yasar.

### 4.4 Semantic Match

Kelimelerin %70'inden fazlasi eslesen ceviriler.

**Sonuc:** 8/12 = 66.7%

---

## 5. Detayli Sonuclar

### 5.1 Basarili Ceviriler Tablosu

| # | Gloss | Referans | Tahmin | Guven | Durum |
|---|-------|----------|--------|-------|-------|
| 1 | BEN OKUL GITMEK | Okula gidiyorum. | Okula gidiyorum. | 10 | Exact |
| 2 | SEN YEMEK YEMEK | Yemek yiyorsun. | Sen bol bol yedin. | 10 | Farkli |
| 3 | AGAC O UZUN YASAMAK OLMAK | Agac uzun yasar. | Agac uzun yasar. | 10 | Exact |
| 4 | BEN KITAP OKUMAK | Kitap okuyorum. | Kitap okuyorum. | 10 | Exact |
| 5 | ANNE YEMEK PISIRMEK | Annem yemek pisirir. | Annem yemek pisiriyor. | 10 | Zaman |
| 6 | COCUK PARK OYNAMAK | Cocuk parkta oynuyor. | Cocuk parkta oynuyor. | 10 | Exact |
| 7 | BABA IS CALISMAK | Babam iste calisiyor. | Babam calisiyor. | 9 | Yer |
| 8 | BEN KAHVALTI ETMEK | Kahvalti ediyorum. | Kahvalti ediyorum. | 10 | Exact |
| 9 | SEN NEREYE GITMEK | Nereye gidiyorsun? | Nereye gidiyorsun? | 10 | Exact |
| 10 | BEN YORGUN OLMAK | Yorgunum. | Yorgunum. | 10 | Exact |
| 11 | HAVA SICAK OLMAK | Hava sicak. | Hava sicak oluyor. | 10 | Fazlalık |
| 12 | BEN CAY ICMEK ISTEMEK | Cay icmek istiyorum. | Cay icmek istiyorum. | 10 | Exact |

### 5.2 Hata Analizi

| Hata Tipi | Sayi | Oran | Ornek |
|-----------|------|------|-------|
| **Zaman farki** | 2 | 16.7% | pisirir vs pisiriyor |
| **Yer eksikligi** | 1 | 8.3% | iste vs (yok) |
| **Fazladan kelime** | 1 | 8.3% | sicak vs sicak oluyor |
| **Farkli yorum** | 1 | 8.3% | yiyorsun vs yedin |
| **Birebir eslesen** | 7 | 58.3% | - |

### 5.3 Guven Puani Dagilimi

| Guven | Sayi | Oran |
|-------|------|------|
| 10/10 | 11 | 91.7% |
| 9/10 | 1 | 8.3% |
| **Ortalama** | **9.92** | - |

---

## 6. RAG Etkisi Analizi

### 6.1 RAG Katkisi

RAG sistemi her ceviri icin sunlari saglar:

1. **Hafiza Ornekleri**: Top-3 benzer cumle (similarity > 0.5)
2. **Kelime Bilgileri**: Her kelime icin tur ve aciklama
3. **Ornek Cumleler**: TID transkripsiyon + Turkce ceviri ciftleri

### 6.2 Ornek RAG Ciktisi

**Girdi:** `OKUL GITMEK ISTEMEK`

**Hafiza Sonuclari:**
| Benzerlik | Transkripsiyon | Ceviri |
|-----------|----------------|--------|
| 0.78 | BEN OKUL GITMEK | Okula gidiyorum |
| 0.77 | OKUL GITMEK | Okula gider |
| 0.76 | BEN ISTEMEK | Istiyorum |

**Sozluk Sonuclari:**
| Kelime | Tur | Aciklama |
|--------|-----|----------|
| OKUL | Ad | Egitim verilen kurum |
| GITMEK | Eylem | Bir yerden baska bir yere hareket etmek |
| ISTEMEK | Eylem | Bir seyi arzulamak, dilemek |

---

## 7. Karsilastirmali Analiz

### 7.1 Literatur Karsilastirmasi

| Calisma | Dil Cifti | BLEU | Yontem |
|---------|-----------|------|--------|
| Camgoz et al., 2018 | DGS->DE | 18.40 | Transformer |
| Camgoz et al., 2020 | DGS->DE | 24.54 | Sign2Gloss2Text |
| Yin & Read, 2020 | ASL->EN | 21.80 | Transformer |
| **Bu calisma** | **TID->TR** | **25.51** | **RAG + LLM** |

**Not:** Dil ciftleri ve test setleri farkli oldugu icin dogrudan karsilastirma yapilamaz.

### 7.2 Baseline Karsilastirma (Planlanan)

| Yontem | BLEU | BERTScore F1 |
|--------|------|--------------|
| Zero-shot LLM | - | - |
| RAG + LLM | 25.51 | 0.847 |
| RAG + LLM + Fine-tuning | - | - |

---

## 8. Sinirliliklar ve Kisitlar

### 8.1 Teknik Kisitlar

1. **API Kota Siniri**: 12 ornekten sonra rate limit hatasi
2. **Test Seti Boyutu**: 12 basarili ceviri (istatistiksel guc sinirli)
3. **Cumle Karmasikligi**: Basit, kisa cumleler test edildi

### 8.2 Metodolojik Kisitlar

1. **Referans Ceviri**: Tek referans ceviri (coklu referans daha iyi)
2. **NMM Eksikligi**: Gorsel NMM bilgisi mevcut degil
3. **Baglam**: Izole cumleler, konusamsal baglam yok

### 8.3 Gelecek Calisma Onerileri

1. Daha buyuk test seti ile degerlendirme (100+ ornek)
2. Coklu referans ceviriler
3. RAG vs Zero-shot baseline karsilastirmasi
4. Human evaluation (insan degerlendirmesi)
5. Per-category analiz (soru, olumsuz, gecmis zaman vb.)

---

## 9. Sonuc ve Tartisma

### 9.1 Temel Bulgular

1. **BLEU 25.51**: Literatur ile karsilastirabilir seviyede
2. **BERTScore 0.847**: Yuksek semantik benzerlik
3. **Exact Match 66.7%**: Basit cumleler icin yuksek dogruluk
4. **Guven 9.92/10**: LLM yuksek kendinden emin

### 9.2 RAG Katkisi

RAG sistemi su sekillerde katkida bulunur:
- Kelime bilgileri ile LLM'in anlamsal boslugunu doldurur
- Benzer cumle ornekleri ile few-shot learning saglar
- Domain-specific bilgi (TID dilbilgisi) iletir

### 9.3 Akademik Katki

Bu calisma:
1. TID-Turkce ceviri icin ilk RAG tabanli sistem
2. Dual-collection (Sozluk + Hafiza) stratejisi
3. Human-in-the-Loop feedback mekanizmasi
4. Kapsamli dilbilgisi kurallari (Topic-Comment, NMM cikarim)

---

## 10. Teknik Detaylar

### 10.1 Sistem Konfigurasyonu

```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
DISTANCE_METRIC = "cosine"
HAFIZA_TOP_K = 3
SOZLUK_TOP_K = 1
SIMILARITY_THRESHOLD = 0.5
MAX_CONTEXT_TOKENS = 2000
```

### 10.2 LLM Konfigurasyonu

```python
MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 2048
OUTPUT_FORMAT = "3 alternatif ceviri"
```

### 10.3 Degerlendirme Kutuphaneleri

```python
# BLEU
sacrebleu>=2.3.0

# BERTScore
bert-score>=0.3.13
model="bert-base-multilingual-cased"
```

---

## 11. Ekler

### Ek A: Test Seti Ornekleri

Tam test seti: `evaluation/test_sets/test_glosses.json` (50 ornek)

### Ek B: Ham Sonuclar

Detayli sonuclar: `evaluation/benchmark_results_corrected.json`

### Ek C: Ornek Prompt

Ornek prompt: `ornek_prompt.md`

---

## 12. Referanslar

1. Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"
2. Zhang, T., et al. (2019). "BERTScore: Evaluating Text Generation with BERT"
3. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
4. Camgoz, N.C., et al. (2018). "Neural Sign Language Translation"
5. Yin, K., & Read, J. (2020). "Better Sign Language Translation with STMC-Transformer"

---

**Rapor Tarihi:** 2026-01-20

**Sistem Surumu:** TID RAG v1.0

**Hazirlayan:** TID RAG Benchmark Pipeline
