# LGS RAG Soru Uretim Sistemi - Kapsamli Teknik Plan

> **Versiyon:** 2.1 (Implementasyon Guncelleme)  
> **Tarih:** 2025-12-24  
> **Proje:** LGS-AR: Artirilmis Gerceklik Destekli Matematik Egitimi ve YZ Soru Analizi  
> **Sahip:** RAG Katmani Gelistirici  
> **Durum:** Core RAG Tamamlandi | Test/Deploy Bekliyor  
> **Tamamlanma:** ~40% (Core: 100%, Test: 0%, Deploy: 0%)

---

## Icerik Tablosu

**BOLUM A: GENEL BAKIS**
1. [Yonetici Ozeti](#1-yonetici-ozeti)
2. [Proje Baglami](#2-proje-baglami)
3. [Katmanlar Arasi Bagimlilik](#3-katmanlar-arasi-bagimlilik)

**BOLUM B: MODEL KATMANI (Upstream)**
4. [Model Katmani Taskleri](#4-model-katmani-taskleri)
5. [Model Ciktilari Sozlesmesi](#5-model-ciktilari-sozlesmesi)

**BOLUM C: RAG KATMANI (Bu Proje)**
6. [RAG Fonksiyonel Gereksinimler](#6-rag-fonksiyonel-gereksinimler)
7. [RAG Teknik Gereksinimler](#7-rag-teknik-gereksinimler)
8. [Sistem Mimarisi](#8-sistem-mimarisi)
9. [Veri Akisi](#9-veri-akisi)
10. [Bilesen Detaylari](#10-bilesen-detaylari)
11. [Veritabani ve Storage Tasarimi](#11-veritabani-ve-storage-tasarimi)
12. [LLM Entegrasyonu](#12-llm-entegrasyonu)
13. [Prompt Muhendisligi](#13-prompt-muhendisligi)
14. [Retrieval Stratejisi](#14-retrieval-stratejisi)
15. [Kalite Kontrol Mekanizmalari](#15-kalite-kontrol-mekanizmalari)
16. [Hata Yonetimi](#16-hata-yonetimi)
17. [API Tasarimi](#17-api-tasarimi)
17.5. [LangChain Entegrasyonu](#175-langchain-entegrasyonu)
17.6. [Manim Gorsel Uretim Modulu](#176-manim-gorsel-uretim-modulu)
18. [RAG Katmani Taskleri](#18-rag-katmani-taskleri)

**BOLUM D: OPERASYONEL**
19. [Test Stratejisi](#19-test-stratejisi)
20. [Deployment](#20-deployment)
21. [Guvenlik](#21-guvenlik)
22. [Performans ve Optimizasyon](#22-performans-ve-optimizasyon)
23. [Izleme ve Loglama](#23-izleme-ve-loglama)
24. [Maliyet Analizi](#24-maliyet-analizi)

**BOLUM E: PROJE YONETIMI**
25. [Zaman Cizelgesi](#25-zaman-cizelgesi)
26. [Riskler ve Azaltma Stratejileri](#26-riskler-ve-azaltma-stratejileri)
27. [Basari Kriterleri](#27-basari-kriterleri)
28. [Ekler](#28-ekler)
    - 28.7 [Implementasyon Durumu ve Eksik Dosyalar](#287-implementasyon-durumu-ve-eksik-dosyalar) ⭐ YENİ

---

## 1. Yonetici Ozeti

### 1.1 Amac

Bu dokuman, LGS (Liseye Gecis Sinavi) matematik sorulari uretmek icin tasarlanan RAG (Retrieval-Augmented Generation) sisteminin eksiksiz teknik planini icerir. Sistem, model katmanindan alinan kombinasyon skorlarini kullanarak, mevcut soru havuzundan benzer sorulari bulur ve LLM ile yeni, ozgun sorular uretir.

### 1.2 Kapsam

- **Dahil:** RAG pipeline, vector store, LLM entegrasyonu, API, kalite kontrol
- **Haric:** Model egitimi (ayri ekip), AR/Unity entegrasyonu (ayri ekip), frontend

### 1.3 Kritik Basari Faktorleri

| Faktor | Olcut | Hedef |
|--------|-------|-------|
| Soru Kalitesi | LGS formatina uygunluk | %95+ |
| Uretim Suresi | Soru basi sure | <10 saniye |
| Cesitlilik | Tekrar orani | <%5 |
| Dogrluk | Matematiksel hata orani | <%1 |

---

## 2. Proje Baglami

### 2.0 Problem ve Cozum

#### Problem

Ogrenciler soyut matematik kavramlarini (EBOB/EKOK) anlamakta zorlaniyor. Yeni nesil LGS sorulari, gorsel muhakeme ve elestirel dusunme gerektiriyor ancak geleneksel ogretim yontemleri bu ihtiyaci karsilamakta yetersiz kaliyor.

| Problem | Aciklama |
|---------|----------|
| Soyut kavramlarin somutlastirilamamasi | EBOB/EKOK gibi kavramlar gorsellestirilemeden ogretilemez |
| Yeni nesil sorularin karmasik yapisi | Cok adimli muhakeme, hikaye bazli sorular |
| Ogrenci motivasyon eksikligi | Geleneksel yontemler ilgi cekici degil |

#### Cozum

Artirilmis gerceklik teknolojisi ile matematiksel kavramlari somutlastiriyoruz, yapay zeka destegiyle soru tiplerini analiz ederek kisisellestirilmis ogrenme deneyimi sunuyoruz.

| Cozum | Aciklama |
|-------|----------|
| AR ile gorsellestirme ve etkilesim | Unity 3D + AR Foundation |
| YZ tabanli soru tipi tahmini | LGS Uygunluk Skoru (0-1) |
| Oyunlastirilmis ogrenme senaryolari | "Ciftcinin Cuvallari" gibi senaryolar |

### 2.1 Ust Proje Yapisi

```
ymgk-qa-ai/
├── 01-S-W-O-T-ANALIZ/          # SWOT analizleri
├── 02-SMART-HEDEFLER/          # SMART hedefler
├── 03-VIDEOLAR/                # Egitim videolari
├── 04-DOCS/                    # Proje dokumanlari
│   └── rag.md                  # Model-RAG sozlesmesi
├── LGS-AR/                     # Unity AR projesi
├── lgs-model/                  # Model katmani
│   ├── data/
│   │   ├── raw/                # Ham veri
│   │   └── processed/          # Islenmis veri
│   │       ├── dataset_model_final.csv
│   │       └── dataset_ocr_li.csv
│   ├── outputs/                # Model ciktilari (beklenen)
│   │   ├── configs.json        # [ZORUNLU] Kombinasyon skorlari
│   │   ├── combination_scores.csv
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── notebooks/
└── lgs-rag/                    # [OLUSTURULACAK] RAG sistemi
```

### 2.1.1 Veri Seti Olusturma: Ekip Calismasi

Veri seti olusturma sureci tum ekip uyelerinin katilimiyla gerceklestirildi.

| Asama | Aciklama |
|-------|----------|
| **Veri Toplama** | LGS matematik sorulari sistematik olarak toplandi ve kategorize edildi |
| **Etiketleme Kriterleri** | Zorluk seviyesi, gorsel tip ve konu bazli detayli siniflandirma |
| **Cift Yonlu Kullanim** | Veri seti hem YZ modelinin egitiminde hem de AR senaryolarinin tasariminda kullanildi |

Bu yaklasim, projenin butunlugunu ve tutarliligini sagladi.

### 2.2 Mevcut Veri Analizi

**Toplam Soru Sayisi:** 176

**Alt Konu Dagilimi:**
| Alt Konu | Sayi | Oran |
|----------|------|------|
| ebob_ekok | 101 | %57.4 |
| carpanlar | 42 | %23.9 |
| aralarinda_asal | 33 | %18.7 |

**Zorluk Dagilimi:**
| Zorluk | Sayi | Oran |
|--------|------|------|
| 1 | 20 | %11.4 |
| 2 | 43 | %24.4 |
| 3 | 59 | %33.5 |
| 4 | 47 | %26.7 |
| 5 | 7 | %4.0 |

**Gorsel Tipi Dagilimi:**
| Gorsel Tipi | Sayi | Oran |
|-------------|------|------|
| yok | 83 | %47.2 |
| sematik | 46 | %26.1 |
| resimli | 29 | %16.5 |
| geometrik_sekil | 17 | %9.7 |
| tablo | 1 | %0.5 |

**Kaynak Tipi Dagilimi:**
| Kaynak | Sayi | is_LGS | Agirlik |
|--------|------|--------|---------|
| baslangic | 87 | 0 | 0.3 |
| ornek | 75 | 1 | 0.8 |
| cikmis | 14 | 1 | 1.0 |

**LGS Profili Sorular:** 89 (ornek + cikmis)

---

## 3. Katmanlar Arasi Bagimlilik

### 3.1 Sistem Katmanlari

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROJE MIMARISI                                  │
│                 LGS-AR: Artirilmis Gerceklik + Yapay Zeka                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   MODEL KATMANI     │    │    RAG KATMANI      │    │    AR KATMANI       │
│   (Upstream)        │───►│    (Bu Proje)       │───►│    (Downstream)     │
│                     │    │                     │    │                     │
│ - Veri Analizi      │    │ - Retrieval         │    │ - Unity 3D          │
│ - Preprocessing     │    │ - LLM Entegrasyon   │    │ - AR Foundation     │
│ - Model Egitimi     │    │ - Soru Uretimi      │    │ - XR Origin         │
│ - Skor Uretimi      │    │ - Kalite Kontrol    │    │ - Low Poly Tasarim  │
│                     │    │                     │    │ - Level Kit         │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
   configs.json              generated_questions.json      AR Experience
   model.pkl                 API endpoints                 "Ciftcinin Cuvallari"
   combination_scores.csv                                  Oyunlastirilmis Senaryo
```

### 3.1.1 AR Modulu Teknoloji Altyapisi

| Teknoloji | Aciklama |
|-----------|----------|
| **Unity 3D** | Ana oyun motoru ve gelistirme platformu |
| **AR Foundation** | XR Origin ve duzlem algilama teknolojileri |
| **Mobil Platform** | Android ve iOS cihaz destegi |
| **Low Poly Tasarim** | Optimum performans icin cartoon stil |

### 3.1.2 AR Yazilim Mimarisi

| Mimari Yaklasim | Aciklama |
|-----------------|----------|
| **Moduler Tasarim (Level Kit)** | Kod yazmadan senaryo tasarimi |
| **ScriptableObjects** | Kod degisikligi gerektirmeden icerik yonetimi |
| **Game Manager / UI Manager** | Singleton pattern ile merkezi yonetim |

### 3.1.3 Ornek AR Senaryo: "Ciftcinin Cuvallari"

**Konu:** EBOB (En Buyuk Ortak Bolen)

Ogrenciler, masa yuzeyinde belirlenen AR duzleminde bugdaylar ve cuvallarla fiziksel olarak etkilesime girer. Senaryo, hasat sonrasi urunlerin esit sekilde dagitilmasi problemini ele alir.

```
[1] Duzlem Algilama     → Kamera ile masa taranir
[2] Senaryo Baslatma    → 3D nesneler yerlestirilir
[3] Etkilesim           → Cuvallari silolara dagitma
[4] Cozum Kontrolu      → Anlik geri bildirim
```

### 3.2 Bagimlilik Matrisi

| Kaynak | Hedef | Dosya/Arayuz | Kritiklik |
|--------|-------|--------------|-----------|
| Model | RAG | `configs.json` | ZORUNLU |
| Model | RAG | `dataset_ocr_li.csv` | ZORUNLU |
| Model | RAG | `combination_scores.csv` | OPSIYONEL |
| Model | RAG | `model.pkl` | OPSIYONEL |
| Model | RAG | `metadata.json` | OPSIYONEL |
| RAG | AR | REST API `/api/v1/generate` | ZORUNLU |
| RAG | AR | JSON soru formati | ZORUNLU |

### 3.3 Kritik Yol

```
[Model TASK 1-6] ──► [Model TASK 7-8] ──► [Model TASK 9-11] ──► [RAG Baslangic]
     Veri Hazirlik       Model Egitimi       Skor Uretimi        configs.json hazir
```

**RAG Baslamak Icin Minimum Gereksinim:**
- `configs.json` (kombinasyonlar + skorlar)
- `dataset_ocr_li.csv` (soru metinleri + metadata)

---

## 4. Model Katmani Taskleri

> **Not:** Bu bolum Model ekibinin sorumlulugundadir. RAG ekibi olarak bu tasklerin ciktilarini bekliyoruz.

### 4.1 Task Ozet Tablosu

| Task ID | Task Adi | Girdi | Cikti | Bagimlilk |
|---------|----------|-------|-------|-----------|
| M-01 | Veri Seti Ilk Analizi | `sorular.csv` | Analiz raporu | - |
| M-02 | Veri Temizleme Scripti | `sorular.csv` | Temiz CSV | M-01 |
| M-03 | is_LGS Etiketinin Olusturulmasi | Temiz CSV | Etiketli CSV | M-02 |
| M-04 | Agirlik Uretimi (sample_weight) | Etiketli CSV | Agirlikli CSV | M-03 |
| M-05 | Feature Engineering | Agirlikli CSV | Feature CSV | M-04 |
| M-06 | Temiz Dataset Ciktisi | Feature CSV | `cleaned_dataset.csv` | M-05 |
| M-07 | Model Pipeline Kurulumu | Pipeline kodu | Pipeline.py | M-06 |
| M-08 | Model Egitimi | `cleaned_dataset.csv` | `model.pkl` | M-07 |
| M-09 | Kombinasyon Taramasi | `model.pkl` | Skor listesi | M-08 |
| M-10 | Threshold Uygulamasi | Skor listesi | Filtrelenmis liste | M-09 |
| M-11 | JSON Uretimi | Filtrelenmis liste | `configs.json` | M-10 |

### 4.2 Task Detaylari

#### M-01: Veri Seti Ilk Analizi

**Amac:** CSV kolonlarini inceleme, eksik/bozuk satirlari tespit etme, alt konu-zorluk-gorsel tipi dagilimlarini cikarma ve veri kalitesi analizi yapma.

**Beklenen Ciktilar:**
```python
# Analiz raporu icerigi
{
    "toplam_satir": 176,
    "eksik_deger_sayisi": {"kolon_adi": sayi},
    "dagilimlar": {
        "alt_konu": {"carpanlar": 42, "ebob_ekok": 101, "aralarinda_asal": 33},
        "zorluk": {1: 20, 2: 43, 3: 59, 4: 47, 5: 7},
        "gorsel_tipi": {"yok": 83, "sematik": 46, ...}
    },
    "veri_kalitesi_skoru": 0.95
}
```

**Kabul Kriterleri:**
- [ ] Tum kolonlar dokumante edildi
- [ ] Eksik degerler raporlandi
- [ ] Dagilim grafikleri olusturuldu
- [ ] Anomaliler tespit edildi

---

#### M-02: Veri Temizleme Scripti

**Amac:** Gorsel Yok/Yok birlestirme, LGS Ornek Soru/Sorusu normalize etme, zorluklari integer'a cevirme, bos satirlari temizleme ve whitespace duzenleme.

**Beklenen Islemler:**
```python
# Gorsel tipi normalizasyonu
gorsel_mapping = {
    "Gorsel Yok": "yok",
    "Yok": "yok",
    "Resimli": "resimli",
    "Geometrik Sekil": "geometrik_sekil",
    "Sematik": "sematik",
    "Tablo": "tablo"
}

# Kaynak tipi normalizasyonu
kaynak_mapping = {
    "Baslangic Sorusu": "baslangic",
    "LGS Ornek Sorusu": "ornek",
    "LGS Ornek Soru": "ornek",
    "Cikmis Soru": "cikmis"
}

# Zorluk donusumu
df["Zorluk"] = df["Zorluk"].astype(str).str.strip().astype(int)
```

**Kabul Kriterleri:**
- [ ] Tum gorsel tipleri normalize edildi (5 kategori)
- [ ] Tum kaynak tipleri normalize edildi (3 kategori)
- [ ] Zorluk degerleri 1-5 arasi integer
- [ ] Bos/null satir yok
- [ ] Whitespace temizlendi

---

#### M-03: is_LGS Etiketinin Olusturulmasi

**Amac:** Cikmis + LGS Ornek = 1, Baslangic Sorusu = 0 olacak sekilde etiket fonksiyonunu olusturma ve veri setine uygulama.

**Beklenen Implementasyon:**
```python
def create_is_lgs_label(kaynak_tipi: str) -> int:
    """
    LGS profili etiketi olustur
    
    Args:
        kaynak_tipi: "cikmis", "ornek", veya "baslangic"
    
    Returns:
        1: LGS profili (cikmis veya ornek)
        0: LGS profili degil (baslangic)
    """
    return 1 if kaynak_tipi in ["cikmis", "ornek"] else 0

df["is_LGS"] = df["Kaynak_Tipi"].apply(create_is_lgs_label)
```

**Beklenen Dagilim:**
| is_LGS | Kaynak | Sayi |
|--------|--------|------|
| 1 | cikmis | 14 |
| 1 | ornek | 75 |
| 0 | baslangic | 87 |

**Kabul Kriterleri:**
- [ ] is_LGS kolonu eklendi
- [ ] Degerler sadece 0 veya 1
- [ ] Toplam is_LGS=1: 89 satir

---

#### M-04: Agirlik Uretimi (sample_weight)

**Amac:** Cikmis sorulara 1.0, LGS ornek sorulara 0.8, baslangic sorularina 0.3 agirligi veren sample_weight fonksiyonunu yazma.

**Beklenen Implementasyon:**
```python
def get_sample_weight(kaynak_tipi: str) -> float:
    """
    Egitim agirligi hesapla
    
    Mantik:
    - Cikmis sorular en degerli (gercek LGS)
    - Ornek sorular degerli (MEB ornekleri)
    - Baslangic sorular referans (egitim amacli)
    """
    weights = {
        "cikmis": 1.0,
        "ornek": 0.8,
        "baslangic": 0.3
    }
    return weights.get(kaynak_tipi, 0.5)

df["egitim_agirligi"] = df["Kaynak_Tipi"].apply(get_sample_weight)
```

**Kabul Kriterleri:**
- [ ] egitim_agirligi kolonu eklendi
- [ ] Degerler: 1.0, 0.8, 0.3
- [ ] Null deger yok

---

#### M-05: Feature Engineering

**Amac:** Alt konu ve gorsel tipi kategorilerini normalize ederek one-hot encoding'e hazirlama, zorluk degerini numerik isleme uygun hale getirme.

**Beklenen Islemler:**
```python
# Kategorik normalizasyon (one-hot icin)
alt_konu_normalized = {
    "Bir Dogal Sayinin Carpanlari": "carpanlar",
    "EBOB ve EKOK": "ebob_ekok",
    "Aralarinda Asal Sayilar": "aralarinda_asal"
}

# OCR-based feature'lar (opsiyonel ama onerilen)
df["ocr_kelime_sayisi"] = df["Soru_MetniOCR"].str.split().str.len()
df["ocr_karakter_sayisi"] = df["Soru_MetniOCR"].str.len()
df["ocr_rakam_sayisi"] = df["Soru_MetniOCR"].str.count(r"\d")

# Cok adimli soru tespiti
reasoning_keywords = ["en az", "en cok", "buna gore", "son durumda"]
df["ocr_cok_adimli"] = df["Soru_MetniOCR"].str.contains(
    "|".join(reasoning_keywords), case=False
).astype(int)
```

**Cikti Kolonlari:**
| Kolon | Tip | Aciklama |
|-------|-----|----------|
| Alt_Konu | str | Normalize edilmis (3 kategori) |
| Zorluk | int | 1-5 arasi |
| Gorsel_Tipi | str | Normalize edilmis (5 kategori) |
| ocr_kelime_sayisi | int | Soru kelime sayisi |
| ocr_karakter_sayisi | int | Soru karakter sayisi |
| ocr_rakam_sayisi | int | Sorudaki rakam sayisi |
| ocr_cok_adimli | int | Cok adimli mi (0/1) |

**Kabul Kriterleri:**
- [ ] Tum kategoriler normalize edildi
- [ ] OCR feature'lari eklendi
- [ ] Null deger yok
- [ ] Tipler dogru

---

#### M-06: Temiz Dataset Ciktisinin Uretilmesi

**Amac:** Tum preprocessing adimlarini pipeline haline getirip cleaned_dataset.csv/pkl olusturma ve modeli egitmeye hazir veri seti cikarma.

**Beklenen Ciktilar:**

1. `cleaned_dataset.csv` - Tum kolonlar (OCR dahil)
2. `dataset_model_final.csv` - Model egitimi icin (OCR haric)
3. `dataset_ocr_li.csv` - RAG retrieval icin (OCR dahil)

**Pipeline Yapisi:**
```python
def preprocessing_pipeline(raw_csv_path: str) -> pd.DataFrame:
    """
    Tam preprocessing pipeline
    
    Adimlar:
    1. CSV oku
    2. Kolon adlarini temizle
    3. Gorsel tipi normalize et
    4. Kaynak tipi normalize et
    5. Alt konu normalize et
    6. Zorluk integer'a cevir
    7. is_LGS etiketi ekle
    8. sample_weight ekle
    9. OCR feature'lari ekle
    10. Bos satirlari temizle
    """
    pass
```

**Kabul Kriterleri:**
- [ ] Pipeline tek fonksiyon olarak calisir
- [ ] 3 cikti dosyasi uretildi
- [ ] Dosyalar UTF-8 encoded
- [ ] Satir sayisi: 176 (temizlik sonrasi)

---

#### M-07: Model Pipeline Kurulumu

**Amac:** OneHotEncoder + LogisticRegression iceren ML pipeline tasarimi, weighted training destegi ekleme, encoding -> model akisi olusturma.

**Beklenen Implementasyon:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Feature kolonlari
categorical_features = ["Alt_Konu", "Gorsel_Tipi"]
numerical_features = ["Zorluk", "ocr_kelime_sayisi", "ocr_rakam_sayisi", "ocr_cok_adimli"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# Full pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",  # Dengesiz siniflar icin
        max_iter=1000,
        random_state=42
    ))
])
```

**Kabul Kriterleri:**
- [ ] Pipeline olusturuldu
- [ ] Categorical encoding calisiyor
- [ ] Numerical features dahil
- [ ] Weighted training destegi var

---

#### M-08: Model Egitimi

**Amac:** Weighted Logistic Regression modelini egitme, skor dagilimini kontrol etme ve egitilmis modeli model.pkl olarak kaydetme.

**Beklenen Islemler:**
```python
import joblib

# Egitim
X = df[categorical_features + numerical_features]
y = df["is_LGS"]
sample_weights = df["egitim_agirligi"]

model_pipeline.fit(X, y, classifier__sample_weight=sample_weights)

# Skor kontrolu
y_proba = model_pipeline.predict_proba(X)[:, 1]
print(f"Skor dagilimi: min={y_proba.min():.2f}, max={y_proba.max():.2f}")
print(f"Ortalama skor: {y_proba.mean():.2f}")

# Kaydet
joblib.dump(model_pipeline, "outputs/model.pkl")
```

**Beklenen Metrikler:**
| Metrik | Beklenen Aralik |
|--------|-----------------|
| Accuracy | >0.80 |
| Precision (is_LGS=1) | >0.75 |
| Recall (is_LGS=1) | >0.80 |
| AUC-ROC | >0.85 |

**Kabul Kriterleri:**
- [ ] Model egitildi
- [ ] Metrikler beklenen aralikta
- [ ] model.pkl kaydedildi
- [ ] Skor dagilimi mantikli (0-1 arasi)

---

#### M-09: Kombinasyon Taramasi

**Amac:** Alt konu x zorluk x gorsel tipi tum kombinasyonlarini uretme ve her biri icin LGS_skor ihtimalini hesaplayan tarama scripti hazirlama.

**Beklenen Implementasyon:**
```python
from itertools import product

# Tum olasi degerler
alt_konular = ["carpanlar", "ebob_ekok", "aralarinda_asal"]
zorluklar = [1, 2, 3, 4, 5]
gorsel_tipleri = ["yok", "resimli", "geometrik_sekil", "sematik", "tablo"]

# Tum kombinasyonlar
combinations = list(product(alt_konular, zorluklar, gorsel_tipleri))
# Toplam: 3 x 5 x 5 = 75 kombinasyon

# Her kombinasyon icin skor hesapla
results = []
for alt_konu, zorluk, gorsel_tipi in combinations:
    # Dummy OCR features (ortalama degerler)
    features = {
        "Alt_Konu": alt_konu,
        "Zorluk": zorluk,
        "Gorsel_Tipi": gorsel_tipi,
        "ocr_kelime_sayisi": 15,  # Ortalama
        "ocr_rakam_sayisi": 3,    # Ortalama
        "ocr_cok_adimli": 0       # Default
    }
    
    X_comb = pd.DataFrame([features])
    lgs_skor = model_pipeline.predict_proba(X_comb)[0, 1]
    
    results.append({
        "alt_konu": alt_konu,
        "zorluk": zorluk,
        "gorsel_tipi": gorsel_tipi,
        "lgs_skor": round(lgs_skor, 4)
    })

# CSV olarak kaydet
pd.DataFrame(results).to_csv("outputs/combination_scores.csv", index=False)
```

**Beklenen Cikti (combination_scores.csv):**
| alt_konu | zorluk | gorsel_tipi | lgs_skor | rank |
|----------|--------|-------------|----------|------|
| ebob_ekok | 4 | sematik | 0.92 | 1 |
| carpanlar | 4 | geometrik_sekil | 0.89 | 2 |
| ... | ... | ... | ... | ... |

**Kabul Kriterleri:**
- [ ] 75 kombinasyon taramasi yapildi
- [ ] Her kombinasyon icin skor hesaplandi
- [ ] combination_scores.csv olusturuldu
- [ ] Skorlar 0-1 araliginda

---

#### M-10: Threshold Uygulamasi

**Amac:** Belirlenen threshold (or. 0.70) uzerindeki kombinasyonlari filtreleme ve aday konfigurasyon listesini olusturma.

**Beklenen Implementasyon:**
```python
THRESHOLD = 0.70  # Ayarlanabilir

# Threshold ustu kombinasyonlar
df_scores = pd.read_csv("outputs/combination_scores.csv")
df_filtered = df_scores[df_scores["lgs_skor"] >= THRESHOLD]

# Siralama ve rank
df_filtered = df_filtered.sort_values("lgs_skor", ascending=False)
df_filtered["rank"] = range(1, len(df_filtered) + 1)
df_filtered["threshold_pass"] = 1

print(f"Threshold: {THRESHOLD}")
print(f"Toplam kombinasyon: {len(df_scores)}")
print(f"Threshold ustu: {len(df_filtered)}")
print(f"Oran: {len(df_filtered)/len(df_scores)*100:.1f}%")
```

**Beklenen Cikti:**
```
Threshold: 0.70
Toplam kombinasyon: 75
Threshold ustu: ~25-35
Oran: ~33-47%
```

**Kabul Kriterleri:**
- [ ] Threshold degeri belirlendi
- [ ] Filtreleme calisiyor
- [ ] En az 15 kombinasyon threshold ustu
- [ ] Rank siralaması eklendi

---

#### M-11: JSON Uretimi

**Amac:** model_version, threshold ve configs[] alanlarini iceren configs.json dosyasini olusturma ve kaydetme.

**Beklenen Cikti (configs.json):**
```json
{
  "schema_version": "1.0",
  "model_version": "lgs-struct-score-v1.0",
  "created_at": "2025-12-24T10:00:00Z",
  "threshold": 0.70,
  "total_combinations": 75,
  "filtered_combinations": 28,
  "selection_policy": {
    "mode": "weighted_sampling",
    "temperature": 1.0,
    "alpha": 2.0,
    "top_k": 20
  },
  "combinations": [
    {
      "alt_konu": "ebob_ekok",
      "zorluk": 4,
      "gorsel_tipi": "sematik",
      "lgs_skor": 0.92,
      "rank": 1
    },
    {
      "alt_konu": "carpanlar",
      "zorluk": 4,
      "gorsel_tipi": "geometrik_sekil",
      "lgs_skor": 0.89,
      "rank": 2
    }
  ]
}
```

**Kabul Kriterleri:**
- [ ] JSON formati gecerli
- [ ] Tum zorunlu alanlar mevcut
- [ ] Kombinasyonlar sirali (skor azalan)
- [ ] UTF-8 encoded
- [ ] `lgs-model/outputs/configs.json` konumunda

### 4.3 Model Cikti Dosyalari Ozeti

| Dosya | Konum | Zorunlu | Aciklama |
|-------|-------|---------|----------|
| `configs.json` | `lgs-model/outputs/` | EVET | RAG'in ana giris dosyasi |
| `combination_scores.csv` | `lgs-model/outputs/` | HAYIR | Debug/analiz icin |
| `model.pkl` | `lgs-model/outputs/` | HAYIR | Yeniden skor uretimi icin |
| `metadata.json` | `lgs-model/outputs/` | HAYIR | Surumleme icin |
| `dataset_ocr_li.csv` | `lgs-model/data/processed/` | EVET | RAG retrieval icin |
| `cleaned_dataset.csv` | `lgs-model/data/processed/` | HAYIR | Referans |

---

## 5. Model Ciktilari Sozlesmesi

> **Kaynak:** `04-DOCS/rag.md` - Model Katmani Ciktilari ve RAG Soru Uretim Mantigi Teknik Sozlesmesi (v1.0)

### 5.1 Sozlesme Amaci

Bu sozlesme, Model (offline) katmaninin uretecegi ciktilari ve bu ciktilarla calisan RAG + LLM (online) katmaninin isleyis mantigini tanimlar.

**Hedef:** RAG tarafinin, model detaylarini bilmeden yalnizca model ciktilariyla sistem kurabilmesi.

### 5.2 Temel Tanimlar

| Terim | Tanim |
|-------|-------|
| **Kombinasyon** | (alt_konu, zorluk, gorsel_tipi) uclusu |
| **LGS Uygunluk Skoru (lgs_skor)** | Kombinasyonun "LGS soru yapisina benzerlik derecesi"ni ifade eden 0-1 arasi deger |
| **Offline** | Model egitimi + skor uretimi + cikti dosyalarinin uretilmesi |
| **Online** | configs.json kullanilarak RAG ile soru uretimi (LLM cagrilari) |

### 5.3 Model Katmaninin Rolu

Model katmani dogrudan soru uretmez ve "bu soru LGS'de kesin cikar" seklinde bir tahmin iddiasi tasimaz.

**Modelin gorevi:**
1. Veri setinden "LGS sorularinin yapisal profilini" ogrenmek
2. Tum olasi kombinasyonlar icin 0-1 arasi LGS Uygunluk Skoru uretmek
3. Bu skorlari RAG'in kullanacagi sekilde disa aktarmak

### 5.3.1 YZ Modelinin Calisma Prensibi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    YZ MODELI CALISMA PRENSIBI                                │
└─────────────────────────────────────────────────────────────────────────────┘

[1] PROFIL OGRENME
    Model, veri setindeki ornek ve cikmis LGS sorularindan bir "yapisal profil" ogrenir.
    
    Bu profil sunlardan olusur:
    ├── Zorluk seviyesi
    ├── Kelime uzunlugu (ocr_kelime_sayisi)
    ├── Gorsel kullanimi (gorsel_tipi)
    ├── Alt konu dagilimi
    └── Soru kurgusu (ocr_cok_adimli)

[2] SKOR HESAPLAMA
    Yeni bir soru yapisi modele verildiginde, model bu yapinin
    LGS sorularinin profilini ne kadar andidigini hesaplar.
    
    Hesap sonucunda 0-1 arasi "LGS Uygunluk Skoru" uretilir:
    
    0.0 ────────────────────────────────────────── 1.0
    │                                              │
    LGS yapisindan uzak              LGS yapisina cok yakin

[3] SKOR KULLANIMI
    Bu skor, olusturulacak yeni sorularin LGS formatina
    uygunlugunu yonlendirmek icin kullanilir.
    
    Yuksek skorlu kombinasyonlar → Oncelikli soru uretimi
```

### 5.3.2 Ozellik Cikarimi (Feature Engineering)

| Ozellik | Aciklama | Kullanim |
|---------|----------|----------|
| `ocr_kelime_sayisi` | Sorunun uzunluk yapisi | Karmasiklik gostergesi |
| `ocr_karakter_sayisi` | Soru metninin karmasiklik duzeyi | Detay seviyesi |
| `ocr_rakam_sayisi` | Islemsel yogunluk gostergesi | Matematiksel zorluk |
| `ocr_cok_adimli` | "buna gore, en az, daha sonra..." ifadeleri | Cok asamali muhakeme bayragi |

### 5.4 Kesin Ciktilar (RAG'in Bagimli Oldugu Dosyalar)

| Dosya | Zorunluluk | Aciklama |
|-------|------------|----------|
| `configs.json` | **ZORUNLU** | RAG tarafinin ana sozlesme dosyasi |
| `combination_scores.csv` | Opsiyonel (Onerilir) | Debug/analiz icin tum skorlar |
| `model.pkl` | Opsiyonel | Yeniden skor uretmek/guncelleme icin |
| `metadata.json` | Opsiyonel | Surumleme/izlenebilirlik icin |

### 5.5 configs.json Semasi (ZORUNLU)

RAG tarafinin ana sozlesme dosyasidir.

**Icerik:**
- model surumu
- skor esigi (threshold)
- kombinasyon listesi (her satir: alt_konu, zorluk, gorsel_tipi, lgs_skor)
- onerilen secim politikasi parametreleri (opsiyonel)

**Ornek:**

```json
{
  "schema_version": "1.0",
  "model_version": "lgs-struct-score-1.0",
  "created_at": "2025-12-23",
  "threshold": 0.75,
  "selection_policy": {
    "mode": "weighted_sampling",
    "temperature": 1.0,
    "top_k": 20
  },
  "combinations": [
    {
      "alt_konu": "carpanlar",
      "zorluk": 4,
      "gorsel_tipi": "tablo",
      "lgs_skor": 0.84
    },
    {
      "alt_konu": "ebob_ekok",
      "zorluk": 3,
      "gorsel_tipi": "sematik",
      "lgs_skor": 0.81
    }
  ]
}
```

**JSON Schema:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["schema_version", "model_version", "threshold", "combinations"],
  "properties": {
    "schema_version": {"type": "string", "example": "1.0"},
    "model_version": {"type": "string", "example": "lgs-struct-score-1.0"},
    "created_at": {"type": "string", "format": "date"},
    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
    "selection_policy": {
      "type": "object",
      "properties": {
        "mode": {"type": "string", "enum": ["weighted_sampling", "top_k", "random"]},
        "temperature": {"type": "number", "default": 1.0},
        "top_k": {"type": "integer", "default": 20}
      }
    },
    "combinations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["alt_konu", "zorluk", "gorsel_tipi", "lgs_skor"],
        "properties": {
          "alt_konu": {"type": "string", "enum": ["carpanlar", "ebob_ekok", "aralarinda_asal"]},
          "zorluk": {"type": "integer", "minimum": 1, "maximum": 5},
          "gorsel_tipi": {"type": "string", "enum": ["yok", "resimli", "geometrik_sekil", "sematik", "tablo"]},
          "lgs_skor": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    }
  }
}
```

### 5.6 combination_scores.csv Semasi (Opsiyonel, Onerilir)

Tum kombinasyonlarin (yalnizca threshold ustu degil) skorlarini icerir. RAG bunu sart kosmaz; ancak analiz, raporlama ve hata ayiklama icin faydalidir.

| Kolon | Tip | Aciklama |
|-------|-----|----------|
| alt_konu | string | carpanlar/ebob_ekok/aralarinda_asal |
| zorluk | int | 1-5 |
| gorsel_tipi | string | yok/resimli/geometrik_sekil/sematik/tablo |
| lgs_skor | float | 0-1 arasi skor |
| rank | int | Skor siralama (1 = en yuksek) |
| threshold_pass | int | 0/1 (threshold ustu mu?) |

### 5.7 metadata.json Semasi (Opsiyonel)

Surumleme ve izlenebilirlik icin:

```json
{
  "training_dataset": "dataset_model_final.csv",
  "dataset_hash": "sha256:abc123...",
  "features": ["Alt_Konu", "Zorluk", "Gorsel_Tipi", "ocr_*"],
  "training_params": {
    "model_type": "LogisticRegression",
    "class_weight": "balanced",
    "max_iter": 1000
  },
  "metrics": {
    "accuracy": 0.85,
    "auc_roc": 0.89
  },
  "random_seed": 42,
  "created_at": "2025-12-24T10:00:00Z"
}
```

### 5.8 dataset_ocr_li.csv Semasi (RAG Retrieval Icin)

| Kolon | Tip | Zorunlu | Aciklama |
|-------|-----|---------|----------|
| Soru_MetniOCR | string | EVET | Soru tam metni |
| Alt_Konu | string | EVET | carpanlar/ebob_ekok/aralarinda_asal |
| Zorluk | int | EVET | 1-5 |
| Gorsel_Tipi | string | EVET | yok/resimli/geometrik_sekil/sematik/tablo |
| Kaynak_Tipi | string | EVET | baslangic/ornek/cikmis |
| egitim_agirligi | float | EVET | 0.3/0.8/1.0 |
| is_LGS | int | EVET | 0/1 |
| ocr_kelime_sayisi | int | HAYIR | Kelime sayisi |
| ocr_karakter_sayisi | int | HAYIR | Karakter sayisi |
| ocr_rakam_sayisi | int | HAYIR | Rakam sayisi |
| ocr_cok_adimli | int | HAYIR | 0/1 |

### 5.9 Repo Konum Yapisi

```
ymgk-qa-ai/
└── lgs-model/
    ├── data/processed/
    │   ├── dataset_model_final.csv
    │   └── dataset_ocr_li.csv
    ├── outputs/
    │   ├── configs.json          # [ZORUNLU] RAG'in ana girdisi
    │   ├── combination_scores.csv # [OPSIYONEL] Debug/analiz
    │   ├── model.pkl             # [OPSIYONEL] Model dosyasi
    │   └── metadata.json         # [OPSIYONEL] Surumleme
    └── notebooks/                 # Colab ipynb dosyalari
```

### 5.10 Teslimat Notu

| Teslimat Paketi | Icerik |
|-----------------|--------|
| **Minimum** | `configs.json` |
| **Onerilen** | `configs.json` + `combination_scores.csv` + `metadata.json` |
| **Tam** | Tum dosyalar + `model.pkl` |

---

## 6. RAG Fonksiyonel Gereksinimler

> **Kaynak:** `04-DOCS/rag.md` - RAG Katmani Mantigi (Online)

### 6.0 RAG Isleyis Adimlari (Sozlesme)

RAG tarafi su adimlarla calisir:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RAG KATMANI ISLEYIS ADIMLARI                            │
└─────────────────────────────────────────────────────────────────────────────┘

[1] configs.json yuklenir
         │
         ▼
[2] Kombinasyon secilir (Top-K veya olasiliksal secim)
         │
         │   Onerilen Politika:
         │   - Once threshold ile filtrele
         │   - Kalanlar arasindan weighted sampling: agirlik = (lgs_skor^alpha), orn. alpha=2
         │   - Istege bagli sicaklik (temperature) ile dagilimi yumusat
         │
         ▼
[3] Secilen kombinasyona uygun ornek sorular veri deposundan cekilir (retrieval)
         │
         │   Strateji:
         │   - ayni alt_konu
         │   - yakin zorluk (+-1)
         │   - ayni/benzer gorsel_tipi
         │   - gerekiyorsa metin benzerligi (embedding) ile siralama
         │
         ▼
[4] LLM promptu; kombinasyon + ornekler + cikti formati ile hazirlanir
         │
         │   LLM'e verilen paket:
         │   - hedef kombinasyon (alt_konu, zorluk, gorsel_tipi)
         │   - 3-8 ornek soru (kisa icerik / sablon)
         │   - cikti formati (JSON)
         │   - kisitlar (LGS tarzi, tek dogru secenek, cozum adimlari vb.)
         │
         ▼
[5] LLM'den yeni soru uretilir
         │
         ▼
[6] (Opsiyonel) Kalite kontrol / format dogrulama / tekrar uretim dongusu
         │
         │   Hizli kontroller:
         │   - 4 secenek var mi?
         │   - dogru cevap seceneklerde mi?
         │   - alt_konu ile uyumlu mu?
         │   - zorluk: adim sayisi / islem yogunlugu ile tutarli mi?
         │
         │   Basarisizsa: ayni kombinasyonla yeniden uretim
         │   veya bir alt siradaki kombinasyona gecis
         │
         ▼
   [CIKTI] Uretilmis LGS Sorusu (JSON)
```

### 6.1 Temel Islevler

| ID | Islev | Aciklama | Oncelik |
|----|-------|----------|---------|
| FR-01 | Kombinasyon Yukleme | configs.json'dan kombinasyonlari oku | P0 |
| FR-02 | Kombinasyon Secimi | Threshold ve weighted sampling ile sec | P0 |
| FR-03 | Soru Retrieval | Benzer sorulari vector store'dan bul | P0 |
| FR-04 | Soru Uretimi | LLM ile yeni soru uret | P0 |
| FR-05 | Kalite Kontrol | Uretilen soruyu dogrula | P0 |
| FR-06 | Tekrar Uretim | Basarisiz sorular icin yeniden dene | P1 |
| FR-07 | Batch Uretim | Toplu soru uretimi | P1 |
| FR-08 | Soru Kaydi | Uretilen sorulari sakla | P1 |
| FR-09 | API Erisimi | REST API ile erisim | P1 |
| FR-10 | Istatistik | Uretim istatistikleri | P2 |

### 6.2 Kullanim Senaryolari

**US-01: Tekil Soru Uretimi**
```
Aktor: Sistem/API Kullanicisi
On Kosul: configs.json mevcut
Akis:
  1. Kombinasyon secimi iste
  2. Sistem weighted sampling ile kombinasyon secer
  3. Benzer sorular retrieval edilir (3-8 adet)
  4. LLM'e prompt gonderilir
  5. Soru uretilir ve dogrulanir
  6. JSON formatta soru dondurulur
Son Kosul: Gecerli LGS formatlı soru
```

**US-02: Batch Soru Uretimi**
```
Aktor: Sistem/API Kullanicisi
Akis:
  1. Istenilen soru sayisi ve filtreler alinir
  2. Her soru icin US-01 tekrarlanir
  3. Cesitlilik kontrolu yapilir
  4. Sonuclar toplu dondurulur
```

**US-03: Belirli Kombinasyon ile Uretim**
```
Aktor: API Kullanicisi
Akis:
  1. Spesifik alt_konu, zorluk, gorsel_tipi verilir
  2. Bu kombinasyona uygun sorular retrieval edilir
  3. Soru uretilir
```

### 6.3 Cikti Formati

```json
{
  "id": "uuid-v4",
  "alt_konu": "carpanlar",
  "zorluk": 4,
  "gorsel_tipi": "tablo",
  "hikaye": "Bir fabrikada uretilen urunler kutulara...",
  "soru": "Buna gore, en az kac kutu gereklidir?",
  "gorsel_aciklama": "3 sutun 4 satirlik tablo, basliklar: Urun, Miktar, Kutu",
  "secenekler": {
    "A": "12",
    "B": "15",
    "C": "18",
    "D": "24"
  },
  "dogru_cevap": "C",
  "cozum": [
    "Adim 1: Urun miktarlarini belirleyelim",
    "Adim 2: EBOB hesaplayalim",
    "Adim 3: Toplam kutu sayisini bulalim"
  ],
  "kontroller": {
    "tek_dogru_mu": true,
    "format_ok": true,
    "secenek_sayisi": 4
  },
  "metadata": {
    "created_at": "2025-12-24T10:30:00Z",
    "model_version": "gpt-4",
    "retrieval_count": 5,
    "generation_attempt": 1
  }
}
```

---

## 7. RAG Teknik Gereksinimler

### 7.1 Yazilim Gereksinimleri

| Kategori | Teknoloji | Versiyon | Amac |
|----------|-----------|----------|------|
| Runtime | Python | 3.11+ | Ana dil |
| Web Framework | FastAPI | 0.104+ | REST API |
| Vector DB | ChromaDB / FAISS | 0.4+ | Embedding storage |
| Embedding | sentence-transformers | 2.2+ | Turkce embedding |
| LLM Client | openai / anthropic | latest | LLM API |
| **RAG Framework** | **LangChain** | **0.1+** | **Basit RAG pipeline** |
| **Gorsel Uretim** | **Manim** | **0.18+** | **Matematik animasyonu** |
| Validation | Pydantic | 2.0+ | Veri dogrulama |
| Async | asyncio / httpx | - | Async islemler |
| Config | python-dotenv | - | Ortam degiskenleri |
| Logging | structlog | - | Yapisal loglama |
| Testing | pytest + pytest-asyncio | - | Test framework |

### 7.1.1 LangChain Bagimliliklari

```bash
pip install langchain langchain-openai langchain-community faiss-cpu
```

### 7.1.2 Manim Bagimliliklari

```bash
pip install manim

# macOS ek bagimliliklar
brew install ffmpeg cairo pango

# Ubuntu ek bagimliliklar
sudo apt install ffmpeg libcairo2-dev libpango1.0-dev
```

### 28.2 Donanim Gereksinimleri

| Ortam | CPU | RAM | Disk | GPU |
|-------|-----|-----|------|-----|
| Development | 4 core | 8GB | 20GB | Opsiyonel |
| Production | 8 core | 16GB | 50GB | Opsiyonel |

### 28.3 Dis Bagimliliklar

| Servis | Kullanim | Zorunlu |
|--------|----------|---------|
| OpenAI API | GPT-4 soru uretimi | Evet* |
| Anthropic API | Claude alternatif | Hayir |
| HuggingFace | Turkce embedding modeli | Evet |

*En az bir LLM API zorunlu

### 28.4 Ortam Degiskenleri

```bash
# .env.example
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Secimi
LLM_PROVIDER=openai  # openai | anthropic | Gemini | local
LLM_MODEL=gpt-4-turbo-preview

# Embedding
EMBEDDING_MODEL=emrecan/bert-base-turkish-cased-mean-nli-stsb-tr

# Paths
CONFIGS_PATH=../lgs-model/outputs/configs.json
QUESTIONS_CSV_PATH=../lgs-model/data/processed/dataset_ocr_li.csv
VECTORSTORE_PATH=./vectorstore

# RAG Settings
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.7
MAX_GENERATION_ATTEMPTS=3

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 8. Sistem Mimarisi

### 8.1 Yuksek Seviye Mimari

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ /generate│  │ /batch   │  │ /health  │  │ /stats               │ │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────────────────┘ │
└───────┼─────────────┼───────────────────────────────────────────────┘
        │             │
        ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Orchestration Layer                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    QuestionGenerationPipeline                  │ │
│  │  1. Load Config → 2. Select Combo → 3. Retrieve → 4. Generate  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐
│ Config Loader │  │ Combination     │  │ Retriever                   │
│               │  │ Selector        │  │ ┌─────────┐  ┌────────────┐ │
│ configs.json  │  │                 │  │ │Embedding│  │ChromaDB    │ │
│ questions.csv │  │ weighted_sample │  │ │ Model   │  │Vector Store│ │
└───────────────┘  └─────────────────┘  │ └─────────┘  └────────────┘ │
                                        └─────────────────────────────┘
                                                      │
                                                      ▼
                          ┌─────────────────────────────────────────────┐
                          │              Generation Layer               │
                          │  ┌─────────────┐  ┌──────────────────────┐  │
                          │  │ Prompt      │  │ LLM Client           │  │
                          │  │ Builder     │  │ (OpenAI/Anthropic)   │  │
                          │  └─────────────┘  └──────────────────────┘  │
                          └─────────────────────────────────────────────┘
                                                      │
                                                      ▼
                          ┌─────────────────────────────────────────────┐
                          │            Quality Control Layer            │
                          │  ┌─────────────┐  ┌──────────────────────┐  │
                          │  │ Format      │  │ Content              │  │
                          │  │ Validator   │  │ Validator            │  │
                          │  └─────────────┘  └──────────────────────┘  │
                          └─────────────────────────────────────────────┘
                                                      │
                                                      ▼
                          ┌─────────────────────────────────────────────┐
                          │              Storage Layer                  │
                          │  ┌─────────────┐  ┌──────────────────────┐  │
                          │  │ Question    │  │ Generation           │  │
                          │  │ Repository  │  │ Logs                 │  │
                          │  └─────────────┘  └──────────────────────┘  │
                          └─────────────────────────────────────────────┘
```

### 8.2 Dizin Yapisi

```
06-RAG-WITH-LANGCHAIN/
├── .env.example                    # Ortam degiskenleri sablonu
├── .env                            # [GIT IGNORE] Gercek degerler
├── .gitignore
├── requirements.txt                # Python bagimliliklari
├── requirements-dev.txt            # Gelistirme bagimliliklari
├── pyproject.toml                  # Proje konfigurasyonu
├── README.md                       # Proje dokumantasyonu
│
├── config/
│   ├── __init__.py
│   ├── settings.py                 # Pydantic Settings
│   └── logging_config.py           # Logging konfigurasyonu
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/                     # Pydantic modeller
│   │   ├── __init__.py
│   │   ├── combination.py          # Kombinasyon modeli
│   │   ├── question.py             # Soru modeli
│   │   ├── config_schema.py        # configs.json semasi
│   │   └── api_models.py           # Request/Response modelleri
│   │
│   ├── services/                   # Is mantigi
│   │   ├── __init__.py
│   │   ├── config_loader.py        # Config ve veri yukleme
│   │   ├── combination_selector.py # Kombinasyon secimi
│   │   ├── embedding_service.py    # Embedding islemleri
│   │   ├── retriever.py            # Vector search
│   │   ├── prompt_builder.py       # Prompt olusturma
│   │   ├── llm_client.py           # LLM API istemcisi
│   │   ├── question_generator.py   # Soru uretim orkestrasyon
│   │   └── quality_checker.py      # Kalite kontrol
│   │
│   ├── repositories/               # Veri erisim katmani
│   │   ├── __init__.py
│   │   ├── question_repository.py  # Soru CRUD
│   │   └── vector_repository.py    # Vector store islemleri
│   │
│   ├── api/                        # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── generate.py         # Uretim endpointleri
│   │   │   ├── health.py           # Saglik kontrol
│   │   │   └── stats.py            # Istatistikler
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── error_handler.py    # Hata yakalama
│   │   │   └── logging.py          # Request loglama
│   │   └── dependencies.py         # FastAPI dependencies
│   │
│   └── utils/                      # Yardimci fonksiyonlar
│       ├── __init__.py
│       ├── text_utils.py           # Metin islemleri
│       └── math_utils.py           # Matematik dogrulama
│
├── vectorstore/                    # ChromaDB verileri
│   └── .gitkeep
│
├── data/                           # Uretilen veriler
│   ├── generated_questions/        # Uretilen sorular
│   └── logs/                       # Uretim loglari
│
├── scripts/
│   ├── init_vectorstore.py         # Vector store baslat
│   ├── test_generation.py          # Manuel test
│   └── benchmark.py                # Performans testi
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/
│   │   ├── test_combination_selector.py
│   │   ├── test_prompt_builder.py
│   │   ├── test_quality_checker.py
│   │   └── test_retriever.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   └── fixtures/
│       ├── sample_configs.json
│       └── sample_questions.csv
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 9. Veri Akisi

### 27.1 Soru Uretim Akisi (Detayli)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SORU URETIM AKISI                                  │
└─────────────────────────────────────────────────────────────────────────────┘

[1] BASLATMA
     │
     ▼
┌─────────────────┐
│ configs.json    │──────────────────────────────────────────────────┐
│ yukle           │                                                  │
└────────┬────────┘                                                  │
         │                                                           │
         ▼                                                           │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ Threshold       │     │ Filtreler:                              │  │
│ uygula          │◄────│  - lgs_skor >= threshold (0.75)         │  │
│ (default: 0.75) │     │  - alt_konu filtresi (opsiyonel)        │  │
└────────┬────────┘     │  - zorluk filtresi (opsiyonel)          │  │
         │              │  - gorsel_tipi filtresi (opsiyonel)     │  │
         │              └─────────────────────────────────────────┘  │
         ▼                                                           │
[2] KOMBINASYON SECIMI                                               │
     │                                                               │
     ▼                                                               │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ Weighted        │     │ Agirlik = lgs_skor ^ alpha              │  │
│ Sampling        │◄────│ alpha = 2 (yuksek skorlar oncelikli)    │  │
│                 │     │ temperature = 1.0 (cesitlilik)          │  │
└────────┬────────┘     └─────────────────────────────────────────┘  │
         │                                                           │
         │  Secilen: {alt_konu: "ebob_ekok", zorluk: 4,              │
         │            gorsel_tipi: "sematik", lgs_skor: 0.87}        │
         ▼                                                           │
[3] RETRIEVAL                                                        │
     │                                                               │
     ▼                                                               │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ Filtre          │     │ Sabit Filtreler:                        │  │
│ Kriterleri      │◄────│  - ayni alt_konu                        │  │
│ Olustur         │     │  - zorluk: +-1 (3,4,5 icin 4)           │  │
└────────┬────────┘     │  - tercihen ayni/benzer gorsel_tipi     │  │
         │              └─────────────────────────────────────────┘  │
         ▼                                                           │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ Vector          │     │ Siralama Kriterleri:                    │  │
│ Similarity      │◄────│  1. Kaynak onceligi: cikmis > ornek     │  │
│ Search          │     │  2. Embedding similarity                │  │
└────────┬────────┘     │  3. Top-K: 5 (configurable)             │  │
         │              └─────────────────────────────────────────┘  │
         │                                                           │
         │  Bulunan: 5 adet benzer soru                              │
         ▼                                                           │
[4] PROMPT OLUSTURMA                                                 │
     │                                                               │
     ▼                                                               │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ System          │     │ Rol: LGS matematik soru yazari          │  │
│ Prompt          │◄────│ Kurallar: LGS formati, 4 secenek, vb.   │  │
└────────┬────────┘     └─────────────────────────────────────────┘  │
         │                                                           │
         ▼                                                           │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ User            │     │ Hedef Kombinasyon                       │  │
│ Prompt          │◄────│ Ornek Sorular (5 adet)                  │  │
│                 │     │ Cikti Formati (JSON)                    │  │
└────────┬────────┘     └─────────────────────────────────────────┘  │
         │                                                           │
         ▼                                                           │
[5] LLM CAGRI                                                        │
     │                                                               │
     ▼                                                               │
┌─────────────────┐     ┌─────────────────────────────────────────┐  │
│ API Call        │     │ Model: gpt-4-turbo-preview              │  │
│ (OpenAI/Claude) │◄────│ Temperature: 0.7                        │  │
│                 │     │ Max Tokens: 2000                        │  │
└────────┬────────┘     └─────────────────────────────────────────┘  │
         │  LLM Response (JSON string)                               │
         │                                                           │
         ▼                                                           │
[6] PARSE & VALIDATE                                                 │
     │                                                               │
     ▼                                                               │
┌─────────────────┐                                                  │
│ JSON Parse      │──── Basarisiz ────┐                              │
└────────┬────────┘                   │                              │
         │ Basarili                   │                              │
         ▼                            │                              │
┌─────────────────┐                   │                              │
│ Schema          │──── Basarisiz ────┤                              │
│ Validation      │                   │                              │
└────────┬────────┘                   │                              │
         │ Basarili                   │                              │
         ▼                            ▼                              │
     │                        │ Retry Logic   │                      │
[7] KALITE KONTROL            ┌───────────────┐                      │
     ▼                        │ (max 3 kez)   │                      │
┌─────────────────┐           └───────┬───────┘                      │
│ Format          │                   │                              │
│ Kontrolu        │◄──────────────────┘                              │
│ - 4 secenek?    │                                                  │
│ - dogru_cevap   │                                                  │
│   seceneklerde? │                                                  │
└────────┬────────┘                                                  │
         │                                                           │
         ▼                                                           │
┌─────────────────┐                                                  │
│ Icerik          │                                                  │
│ Kontrolu        │                                                  │
│ - alt_konu      │                                                  │
│   uyumlu mu?    │                                                  │
│ - zorluk        │                                                  │
│   mantikli mi?  │                                                  │
└────────┬────────┘                                                  │
         │                                                           │
         ▼                                                           │
[8] CIKTI                                                            │
     │                                                               │
     ▼                                                               │
┌─────────────────┐                                                  │
│ Final JSON      │                                                  │
│ + Metadata      │                                                  │
└─────────────────┘                                                  │
```

### 27.2 Vector Store Baslangic Akisi

```
[INIT] Sistem Baslangici
         │
         ▼
┌─────────────────┐
│ dataset_ocr_li  │
│ .csv yukle      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Her satir icin: │
│ - Soru metni al │
│ - Metadata al   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │
│ Model ile       │
│ vektorlestir    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB'ye     │
│ kaydet          │
│ + metadata      │
└─────────────────┘
```

---

## 10. Bilesen Detaylari

### 10.1 Config Loader

**Dosya:** `src/services/config_loader.py`

**Sorumluluklar:**
- `configs.json` okuma ve dogrulama
- `dataset_ocr_li.csv` okuma
- Singleton pattern ile tek yuklem

```python
# Pseudo-kod
class ConfigLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_configs(self, path: str) -> ConfigSchema:
        """configs.json yukle ve dogrula"""
        
    def load_questions(self, path: str) -> List[Question]:
        """CSV'den sorulari yukle"""
        
    def get_combinations(self) -> List[Combination]:
        """Threshold ustu kombinasyonlari don"""
```

**Hata Durumlari:**
| Hata | Aksiyon |
|------|---------|
| Dosya bulunamadi | ConfigFileNotFoundError raise |
| JSON parse hatasi | InvalidConfigFormatError raise |
| Schema uyumsuz | ConfigValidationError raise |

### 10.2 Combination Selector

> **Kaynak:** `04-DOCS/rag.md` - Kombinasyon Secimi Onerilen Politika

**Dosya:** `src/services/combination_selector.py`

**Onerilen Politika (Sozlesme):**

Deterministik tekrarlari azaltmak icin:
1. Once threshold ile filtrele
2. Kalanlar arasindan weighted sampling: agirlik = (lgs_skor^alpha), orn. alpha=2
3. Istege bagli sicaklik (temperature) ile dagilimi yumusat

Boylece hem yuksek skorlu yapilar oncelikli olur hem de cesitlilik korunur.

**Sorumluluklar:**
- Threshold filtreleme
- Weighted sampling (alpha=2)
- Temperature ile yumusatma
- Filtre uygulama (alt_konu, zorluk, gorsel_tipi)

```python
class CombinationSelector:
    def __init__(self, combinations: List[Combination], config: SelectionPolicy):
        self.combinations = combinations
        self.config = config
    
    def select(
        self,
        filters: Optional[CombinationFilters] = None,
        exclude: Optional[List[Combination]] = None
    ) -> Combination:
        """
        Weighted sampling ile kombinasyon sec
        
        Args:
            filters: alt_konu, zorluk, gorsel_tipi filtreleri
            exclude: Haric tutulacak kombinasyonlar (cesitlilik icin)
        
        Returns:
            Secilen Combination
        """
        
    def _calculate_weights(self, combinations: List[Combination]) -> List[float]:
        """Agirlik = skor ^ alpha"""
        
    def _apply_temperature(self, weights: List[float]) -> List[float]:
        """Softmax with temperature"""
```

**Algoritma Detayi:**
```
1. Threshold filtresi: skor >= 0.75
2. Kullanici filtreleri uygula
3. Haric tutulacaklari cikar
4. Her kombinasyon icin agirlik = skor^2
5. Temperature ile softmax: p_i = exp(w_i/T) / sum(exp(w_j/T))
6. Olasiliksal secim: numpy.random.choice(weights=p)
```

### 10.3 Embedding Service

**Dosya:** `src/services/embedding_service.py`

**Sorumluluklar:**
- Turkce text embedding
- Batch embedding desteği
- Cache mekanizmasi

```python
class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._cache = {}
    
    def embed(self, text: str) -> np.ndarray:
        """Tek metin icin embedding"""
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Toplu embedding"""
        
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity"""
```

**Model Secimi:**
```
Birincil: emrecan/bert-base-turkish-cased-mean-nli-stsb-tr
Alternatif: dbmdz/bert-base-turkish-cased
Fallback: paraphrase-multilingual-MiniLM-L12-v2
```

### 10.4 Retriever

**Dosya:** `src/services/retriever.py`

**Sorumluluklar:**
- ChromaDB uzerinde arama
- Metadata filtreleme
- Sonuc siralama

```python
class Retriever:
    def __init__(self, collection: chromadb.Collection, embedding_service: EmbeddingService):
        self.collection = collection
        self.embedding_service = embedding_service
    
    def retrieve(
        self,
        combination: Combination,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[RetrievedQuestion]:
        """
        Kombinasyona uygun sorulari getir
        
        Strateji:
        1. alt_konu = hedef
        2. zorluk in [hedef-1, hedef, hedef+1]
        3. gorsel_tipi = hedef (tercihen)
        4. kaynak_tipi onceligi: cikmis > ornek > baslangic
        """
        
    def _build_where_filter(self, combination: Combination) -> Dict:
        """ChromaDB where filtresi olustur"""
        
    def _rerank(self, results: List, combination: Combination) -> List:
        """Kaynak tipine gore yeniden sirala"""
```

**Arama Stratejisi:**
```
Seviye 1: Tam eslesme
  - alt_konu = hedef
  - zorluk = hedef
  - gorsel_tipi = hedef
  
Seviye 2: Zorluk genisletme
  - alt_konu = hedef
  - zorluk in [hedef-1, hedef+1]
  - gorsel_tipi = hedef
  
Seviye 3: Gorsel tipi genisletme
  - alt_konu = hedef
  - zorluk in [hedef-1, hedef, hedef+1]
  - gorsel_tipi = herhangi
```

### 10.5 Prompt Builder

**Dosya:** `src/services/prompt_builder.py`

**Sorumluluklar:**
- System prompt olusturma
- User prompt olusturma
- Ornek formatlama

```python
class PromptBuilder:
    def __init__(self):
        self.system_template = self._load_system_template()
        self.user_template = self._load_user_template()
    
    def build_system_prompt(self) -> str:
        """LGS soru yazari rolu tanimla"""
        
    def build_user_prompt(
        self,
        combination: Combination,
        examples: List[RetrievedQuestion]
    ) -> str:
        """Hedef + ornekler + format"""
        
    def format_examples(self, examples: List[RetrievedQuestion]) -> str:
        """Ornekleri okunabilir formata cevir"""
```

### 10.6 LLM Client

**Dosya:** `src/services/llm_client.py`

**Sorumluluklar:**
- OpenAI/Anthropic API cagrisi
- Retry mekanizmasi
- Rate limiting

```python
class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self._init_client(api_key)
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """LLM API cagrisi"""
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    async def _call_api(self, messages: List[Dict]) -> str:
        """Retry ile API cagrisi"""
```

**Provider Implementasyonlari:**
```python
class OpenAIClient(LLMClient):
    """OpenAI GPT modelleri"""
    
class AnthropicClient(LLMClient):
    """Anthropic Claude modelleri"""
    
class LLMClientFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> LLMClient:
        """Factory pattern ile client olustur"""
```

### 10.7 Question Generator

**Dosya:** `src/services/question_generator.py`

**Sorumluluklar:**
- Pipeline orkestrasyon
- Retry logic
- Sonuc birlestirme

```python
class QuestionGenerator:
    def __init__(
        self,
        config_loader: ConfigLoader,
        combination_selector: CombinationSelector,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        quality_checker: QualityChecker
    ):
        # Dependency injection
        
    async def generate(
        self,
        filters: Optional[CombinationFilters] = None,
        combination: Optional[Combination] = None
    ) -> GeneratedQuestion:
        """
        Tek soru uret
        
        Args:
            filters: Kombinasyon filtreleri
            combination: Spesifik kombinasyon (opsiyonel)
        """
        
    async def generate_batch(
        self,
        count: int,
        filters: Optional[CombinationFilters] = None,
        ensure_diversity: bool = True
    ) -> List[GeneratedQuestion]:
        """Toplu uretim"""
        
    async def _generate_with_retry(
        self,
        combination: Combination,
        examples: List[RetrievedQuestion],
        max_attempts: int = 3
    ) -> GeneratedQuestion:
        """Retry ile uretim"""
```

### 10.8 Quality Checker

**Dosya:** `src/services/quality_checker.py`

**Sorumluluklar:**
- Format dogrulama
- Icerik dogrulama
- Matematiksel tutarlilik kontrolu

```python
class QualityChecker:
    def check(self, question: GeneratedQuestion, combination: Combination) -> QualityResult:
        """Tum kontrolleri calistir"""
        
    def check_format(self, question: GeneratedQuestion) -> List[str]:
        """
        Format kontrolleri:
        - 4 secenek var mi?
        - dogru_cevap A/B/C/D mi?
        - dogru_cevap seceneklerde mi?
        - cozum adimlari var mi?
        """
        
    def check_content(self, question: GeneratedQuestion, combination: Combination) -> List[str]:
        """
        Icerik kontrolleri:
        - alt_konu uyumlu mu?
        - zorluk mantikli mi? (adim sayisi, islem yogunlugu)
        - gorsel_aciklama gerekli ve mantikli mi?
        """
        
    def check_math(self, question: GeneratedQuestion) -> List[str]:
        """
        Matematiksel kontroller:
        - Sayilar tutarli mi?
        - Cozum adimlari dogru mu? (basit kontrol)
        """
```

**Kalite Skoru:**
```python
class QualityResult:
    passed: bool
    score: float  # 0-1 arasi
    errors: List[str]
    warnings: List[str]
```

---

## 11. Veritabani ve Storage Tasarimi

### 25.1 ChromaDB Collection Schema

```python
# Collection: lgs_questions
{
    "id": "uuid-string",
    "embedding": [float, ...],  # 768 boyut (BERT)
    "document": "Soru metni...",
    "metadata": {
        "alt_konu": "ebob_ekok",
        "zorluk": 4,
        "gorsel_tipi": "sematik",
        "kaynak_tipi": "cikmis",
        "is_lgs": 1,
        "egitim_agirligi": 1.0,
        "ocr_kelime_sayisi": 25,
        "ocr_karakter_sayisi": 180,
        "ocr_rakam_sayisi": 5,
        "ocr_cok_adimli": 1
    }
}
```

### 25.2 Uretilen Sorular Storage

**Dosya Formati:** JSON Lines (`.jsonl`)

```json
{"id": "uuid", "question": {...}, "metadata": {...}, "created_at": "..."}
{"id": "uuid", "question": {...}, "metadata": {...}, "created_at": "..."}
```

**Dizin Yapisi:**
```
data/
├── generated_questions/
│   ├── 2025-12-24.jsonl
│   ├── 2025-12-25.jsonl
│   └── ...
└── logs/
    ├── generation_2025-12-24.log
    └── ...
```

### 25.3 Cache Yapisi

```python
# Redis veya in-memory cache
cache_schema = {
    "embedding:{text_hash}": embedding_vector,
    "combination_scores": serialized_combinations,
    "recent_generations": List[question_id]  # Son 100 uretim
}
```

---

## 12. LLM Entegrasyonu

### 27.1 Provider Karsilastirmasi

| Ozellik | OpenAI GPT-4 | Claude 3.5 | GPT-3.5 |
|---------|--------------|------------|---------|
| Kalite | Cok Yuksek | Cok Yuksek | Orta |
| Turkce | Iyi | Iyi | Orta |
| Maliyet | Yuksek | Yuksek | Dusuk |
| Hiz | Orta | Iyi | Cok Iyi |
| Oneri | Uretim | Alternatif | Test |

### 27.2 Model Parametreleri

```python
GENERATION_CONFIG = {
    "openai": {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.95,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "max_tokens": 2000
    }
}
```

### 27.3 Rate Limiting

```python
RATE_LIMITS = {
    "openai": {
        "requests_per_minute": 60,
        "tokens_per_minute": 90000
    },
    "anthropic": {
        "requests_per_minute": 50,
        "tokens_per_minute": 100000
    }
}
```

### 27.4 Fallback Stratejisi

```
1. Birincil: OpenAI GPT-4
2. Rate limit asildiysa: Anthropic Claude
3. Her iki API de basarisiz: Queue'ya ekle, sonra dene
4. 3 basarisiz deneme: Manuel inceleme icin logla
```

---

## 13. Prompt Muhendisligi

### 28.1 System Prompt

```markdown
Sen, LGS (Liseye Gecis Sinavi) matematik sorulari yazan uzman bir egitimcisin.

## Rol
- 8. sinif ogrencilerine yonelik matematik sorulari yaziyorsun
- Sorular MEB mufredat ve LGS formatlarina uygun olmali
- Her soru tek dogru cevapli, 4 secenekli olmali

## Konu Alanlari
- Carpanlar ve Katlar
  - Bir dogal sayinin carpanlari
  - EBOB ve EKOK
  - Aralarinda asal sayilar

## LGS Soru Ozellikleri
1. Gercek hayat senaryolari (hikayeli sorular)
2. Cok adimli cozum gerektiren yapida
3. Yaniltici secenekler (dikkat gerektiren)
4. Gorsel destek (tablo, sema, sekil) kullanimi

## Zorluk Seviyeleri
1: Temel kavram, tek islem
2: Iki islemli, basit uygulama
3: Uc+ islem, orta karmasiklik
4: Cok adimli, ileri duzey
5: Olimpiyat tarzi, en zor

## Kurallar
- Turkce dil bilgisi kurallarina uy
- Matematiksel notasyon dogru kullan
- Secenekler mantikli ve tutarli olmali
- Cozum adimlari acik ve anlasilir olmali
```

### 13.2 User Prompt Sablonu

```markdown
Asagidaki kombinasyona uygun yeni bir LGS matematik sorusu uret.

## Hedef Kombinasyon
- Alt Konu: {alt_konu}
- Zorluk: {zorluk}/5
- Gorsel Tipi: {gorsel_tipi}

## Ornek Sorular
Bu kombinasyona benzer gecmis LGS sorulari:

{formatted_examples}

## Cikti Formati
Asagidaki JSON formatinda yanit ver:

```json
{
  "alt_konu": "carpanlar",
  "zorluk": 4,
  "gorsel_tipi": "tablo",
  "hikaye": "Soru baglami/hikayesi...",
  "soru": "Asil soru metni...",
  "gorsel_aciklama": "Gorselin detayli aciklamasi (gorsel gerekliyse)",
  "secenekler": {
    "A": "Secenek A",
    "B": "Secenek B", 
    "C": "Secenek C",
    "D": "Secenek D"
  },
  "dogru_cevap": "B",
  "cozum": [
    "Adim 1: ...",
    "Adim 2: ...",
    "Adim 3: ..."
  ],
  "kontroller": {
    "tek_dogru_mu": true,
    "format_ok": true
  }
}
```

## Onemli Notlar
- Ornek sorulardan FARKLI, ozgun bir soru uret
- Sayilari ve senaryoyu degistir
- Zorluk seviyesine uygun karmasiklikta ol
- {gorsel_tipi} tipinde gorsel icin detayli aciklama ver
- kontroller alanini doldur (tek dogru secenek ve format kontrolu)
```

### 13.3 Ornek Formatlama

```markdown
### Ornek {i} (Zorluk: {zorluk}, Kaynak: {kaynak})
**Soru:** {soru_metni}

**Alt Konu:** {alt_konu}
**Gorsel:** {gorsel_tipi}

---
```

---

## 14. Retrieval Stratejisi

> **Kaynak:** `04-DOCS/rag.md` - Retrieval Beklenen Girdi/Cikti

### 14.0 Sozlesme Tanimi

**Girdi:** Secilen kombinasyon (alt_konu, zorluk, gorsel_tipi)

**Cikti:** 3-8 adet benzer ornek soru (tercihen cikmis + ornek)

**Strateji (oneri):**
- ayni alt_konu
- yakin zorluk (+-1)
- ayni/benzer gorsel_tipi
- gerekiyorsa metin benzerligi (embedding) ile siralama

### 14.1 Arama Algoritma

```python
def retrieve_strategy(combination: Combination, top_k: int = 5) -> List[Question]:
    """
    Katmanli arama stratejisi
    """
    results = []
    
    # Katman 1: Tam eslesme + cikmis sorular
    layer1 = search(
        alt_konu=combination.alt_konu,
        zorluk=combination.zorluk,
        gorsel_tipi=combination.gorsel_tipi,
        kaynak_tipi="cikmis"
    )
    results.extend(layer1)
    
    # Katman 2: Tam eslesme + ornek sorular
    if len(results) < top_k:
        layer2 = search(
            alt_konu=combination.alt_konu,
            zorluk=combination.zorluk,
            gorsel_tipi=combination.gorsel_tipi,
            kaynak_tipi="ornek"
        )
        results.extend(layer2)
    
    # Katman 3: Zorluk +-1, ayni gorsel
    if len(results) < top_k:
        layer3 = search(
            alt_konu=combination.alt_konu,
            zorluk__in=[combination.zorluk - 1, combination.zorluk + 1],
            gorsel_tipi=combination.gorsel_tipi,
            kaynak_tipi__in=["cikmis", "ornek"]
        )
        results.extend(layer3)
    
    # Katman 4: Sadece alt_konu eslesme
    if len(results) < top_k:
        layer4 = search(
            alt_konu=combination.alt_konu,
            kaynak_tipi__in=["cikmis", "ornek"]
        )
        results.extend(layer4)
    
    # Tekrarlari kaldir ve sirala
    results = deduplicate(results)
    results = rerank_by_relevance(results, combination)
    
    return results[:top_k]
```

### 14.2 Siralama Kriterleri

| Oncelik | Kriter | Agirlik |
|---------|--------|---------|
| 1 | kaynak_tipi = cikmis | 1.0 |
| 2 | kaynak_tipi = ornek | 0.8 |
| 3 | zorluk = hedef | +0.2 |
| 4 | gorsel_tipi = hedef | +0.15 |
| 5 | Embedding similarity | x0.5 |

### 14.3 Minimum Sonuc Garantisi

```python
MIN_RETRIEVAL_COUNT = 3
MAX_RETRIEVAL_COUNT = 8

if len(results) < MIN_RETRIEVAL_COUNT:
    # Baslangic sorularini da dahil et
    fallback = search(
        alt_konu=combination.alt_konu,
        kaynak_tipi="baslangic"
    )
    results.extend(fallback)
```

---

## 15. Kalite Kontrol Mekanizmalari

> **Kaynak:** `04-DOCS/rag.md` - Kalite Kontrol Onerileri

### 15.0 Hizli Kontroller (Sozlesme)

Belgede tanimlanan hizli kontroller:

| Kontrol | Soru | Basarisizlik Aksiyonu |
|---------|------|----------------------|
| Secenek sayisi | 4 secenek var mi? | Retry |
| Dogru cevap | dogru cevap seceneklerde mi? | Retry |
| Konu uyumu | alt_konu ile uyumlu mu? (basit kural kontrolu) | Warning + Retry |
| Zorluk tutarliligi | zorluk: adim sayisi / islem yogunlugu ile tutarli mi? | Warning |

**Basarisizlik Stratejisi:**
1. Ayni kombinasyonla yeniden uretim
2. Veya bir alt siradaki kombinasyona gecis

### 15.1 Kontrol Matrisi (Detayli)

| Kontrol | Tip | Basarisizlik Aksiyonu |
|---------|-----|----------------------|
| JSON parse | Format | Retry |
| Schema validation | Format | Retry |
| 4 secenek | Format | Retry |
| dogru_cevap gecerli (A/B/C/D) | Format | Retry |
| dogru_cevap seceneklerde | Format | Retry |
| alt_konu uyumu | Icerik | Warning + Log |
| Zorluk tutarliligi | Icerik | Warning |
| Gorsel gereklilik (gorsel_tipi != yok) | Icerik | Warning |
| Tekrar kontrolu | Cesitlilik | Retry with exclusion |
| kontroller.tek_dogru_mu | Self-check | Warning |
| kontroller.format_ok | Self-check | Warning |

### 15.2 Tekrar Tespit

```python
class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.85):
        self.recent_questions = []  # Son 100 soru
        self.threshold = similarity_threshold
    
    def is_duplicate(self, new_question: str) -> bool:
        new_embedding = embed(new_question)
        
        for recent in self.recent_questions:
            similarity = cosine_similarity(new_embedding, recent.embedding)
            if similarity > self.threshold:
                return True
        
        return False
    
    def add(self, question: str):
        self.recent_questions.append({
            "text": question,
            "embedding": embed(question)
        })
        
        # FIFO: 100'den fazlaysa eskiyi sil
        if len(self.recent_questions) > 100:
            self.recent_questions.pop(0)
```

### 15.3 Kalite Skoru Hesaplama

```python
def calculate_quality_score(question: GeneratedQuestion, checks: Dict) -> float:
    """
    Toplam kalite skoru (0-1)
    """
    weights = {
        "format_valid": 0.3,
        "content_relevant": 0.25,
        "difficulty_appropriate": 0.2,
        "solution_complete": 0.15,
        "not_duplicate": 0.1
    }
    
    score = sum(
        weights[check] * (1.0 if passed else 0.0)
        for check, passed in checks.items()
    )
    
    return score
```

---

## 16. Hata Yonetimi

### 28.1 Hata Siniflandirmasi

```python
# exceptions.py

class RAGException(Exception):
    """Base exception"""
    
class ConfigError(RAGException):
    """Konfigürasyon hatalari"""
    
class ConfigFileNotFoundError(ConfigError):
    pass
    
class InvalidConfigFormatError(ConfigError):
    pass

class RetrievalError(RAGException):
    """Retrieval hatalari"""
    
class InsufficientResultsError(RetrievalError):
    pass

class LLMError(RAGException):
    """LLM API hatalari"""
    
class RateLimitError(LLMError):
    pass
    
class APIConnectionError(LLMError):
    pass

class GenerationError(RAGException):
    """Uretim hatalari"""
    
class InvalidOutputFormatError(GenerationError):
    pass
    
class QualityCheckFailedError(GenerationError):
    pass
```

### 28.2 Retry Politikasi

```python
RETRY_CONFIG = {
    "max_attempts": 3,
    "initial_wait": 1,  # saniye
    "max_wait": 30,
    "exponential_base": 2,
    "retryable_errors": [
        RateLimitError,
        APIConnectionError,
        InvalidOutputFormatError
    ]
}
```

### 28.3 Graceful Degradation

```
Senaryo: LLM API kullanilmaz
Aksiyon:
  1. Fallback provider'a gec
  2. Queue'ya ekle, sonra dene
  3. Mevcut soru havuzundan rastgele sec (son care)
```

---

## 17. API Tasarimi

### 17.1 Endpoint Listesi

| Method | Endpoint | Aciklama |
|--------|----------|----------|
| POST | /api/v1/generate | Tek soru uret |
| POST | /api/v1/generate/batch | Toplu uretim |
| GET | /api/v1/combinations | Mevcut kombinasyonlar |
| GET | /api/v1/health | Saglik kontrolu |
| GET | /api/v1/stats | Istatistikler |

### 17.2 Request/Response Semalari

**POST /api/v1/generate**

Request:
```json
{
  "filters": {
    "alt_konu": "ebob_ekok",
    "zorluk": 4,
    "gorsel_tipi": "sematik"
  },
  "specific_combination": false
}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "alt_konu": "ebob_ekok",
    "zorluk": 4,
    "gorsel_tipi": "sematik",
    "hikaye": "...",
    "soru": "...",
    "gorsel_aciklama": "...",
    "secenekler": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "dogru_cevap": "B",
    "cozum": ["..."],
    "metadata": {
      "generation_time_ms": 2500,
      "retrieval_count": 5,
      "quality_score": 0.95
    }
  }
}
```

**POST /api/v1/generate/batch**

Request:
```json
{
  "count": 10,
  "filters": {
    "alt_konu": "carpanlar"
  },
  "ensure_diversity": true
}
```

Response:
```json
{
  "success": true,
  "data": {
    "questions": [...],
    "stats": {
      "requested": 10,
      "generated": 10,
      "failed": 0,
      "duplicates_avoided": 2
    }
  }
}
```

### 17.3 Hata Yanit Formati

```json
{
  "success": false,
  "error": {
    "code": "GENERATION_FAILED",
    "message": "Soru uretimi basarisiz",
    "details": "LLM API baglanti hatasi",
    "retry_after": 60
  }
}
```

---

## 17.5 LangChain Entegrasyonu

> **Not:** LangChain entegrasyonu basit tutulacak. Temel RAG pipeline'i icin yeterli.

### 17.5.1 LangChain Nedir?

LangChain, buyuk dil modelleri (LLM) ile uygulama gelistirmeyi kolaylastiran bir Python framework'udur. RAG sistemleri icin:
- Document loaders
- Text splitters
- Embeddings
- Vector stores
- Chains (RetrievalQA)

### 17.5.2 Kurulum

```bash
pip install langchain langchain-openai langchain-community faiss-cpu chromadb
```

### 17.5.3 Basit LangChain RAG Pipeline

```python
# src/langchain_rag.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd

class LangChainRAG:
    """
    LangChain tabanli basit RAG sistemi
    """
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            api_key=openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.vectorstore = None
    
    def load_questions(self, csv_path: str):
        """
        CSV'den sorulari yukle ve vectorstore olustur
        """
        df = pd.read_csv(csv_path)
        
        # Dokuman metinleri olustur
        documents = []
        for _, row in df.iterrows():
            text = f"""
            Soru: {row['Soru_MetniOCR']}
            Alt Konu: {row['Alt_Konu']}
            Zorluk: {row['Zorluk']}
            Gorsel Tipi: {row['Gorsel_Tipi']}
            Kaynak: {row['Kaynak_Tipi']}
            """
            documents.append(text)
        
        # Text splitter (kisa sorular icin gerekli degil ama buyuk dokumanlarda lazim)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # FAISS vectorstore olustur
        self.vectorstore = FAISS.from_texts(
            documents, 
            self.embeddings
        )
        
        print(f"Yuklendi: {len(documents)} soru")
    
    def create_qa_chain(self):
        """
        Soru-cevap zinciri olustur
        """
        # LGS soru uretim promptu
        prompt_template = """
        Sen LGS matematik sorulari ureten uzman bir egitimcisin.
        
        Asagidaki ornek sorulari referans alarak yeni bir soru uret:
        
        {context}
        
        Hedef Kombinasyon: {question}
        
        JSON formatinda yanit ver:
        {{
            "hikaye": "...",
            "soru": "...",
            "secenekler": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
            "dogru_cevap": "...",
            "cozum": ["Adim 1...", "Adim 2..."]
        }}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def generate_question(self, combination: dict) -> str:
        """
        Kombinasyona gore soru uret
        """
        query = f"Alt Konu: {combination['alt_konu']}, Zorluk: {combination['zorluk']}, Gorsel: {combination['gorsel_tipi']}"
        result = self.qa_chain.invoke({"query": query})
        return result["result"]
```

### 17.5.4 Kullanim Ornegi

```python
# LangChain RAG kullanimi
from langchain_rag import LangChainRAG

# Baslat
rag = LangChainRAG(openai_api_key="sk-...")

# Sorulari yukle
rag.load_questions("lgs-model/data/processed/dataset_ocr_li.csv")

# QA zinciri olustur
rag.create_qa_chain()

# Soru uret
question = rag.generate_question({
    "alt_konu": "ebob_ekok",
    "zorluk": 4,
    "gorsel_tipi": "sematik"
})

print(question)
```

### 17.5.5 LangChain vs Custom RAG

| Ozellik | LangChain | Custom RAG |
|---------|-----------|------------|
| Kurulum | Hizli | Orta |
| Esneklik | Orta | Yuksek |
| Kontrol | Sinirli | Tam |
| Debugging | Zor | Kolay |
| Performans | Iyi | Optimize edilebilir |

**Oneri:** Prototip icin LangChain, uretim icin Custom RAG.

---

## 17.6 Manim Gorsel Uretim Modulu

> **Not:** Manim, matematiksel kavramlari animasyonlarla gorsellestiren Python kutuphanesidir (3Blue1Brown).

### 17.6.1 Manim Nedir?

Manim (Mathematical Animation Engine), Grant Sanderson (3Blue1Brown) tarafindan gelistirilen, matematik egitimi icin animasyon ureten bir Python kutuphanesidir.

**Kullanim Alanlari:**
- EBOB/EKOK gorsellestirme
- Carpan agaclari animasyonu
- Soru cozum adimlari animasyonu
- Gorsel soru uretimi

### 17.6.2 Kurulum

```bash
# Manim Community Edition
pip install manim

# Ek bagimliliklar (macOS)
brew install ffmpeg cairo pango

# Ek bagimliliklar (Ubuntu)
sudo apt install ffmpeg libcairo2-dev libpango1.0-dev
```

### 17.6.3 Temel Manim Ornekleri

#### EBOB Gorsellestirme

```python
# src/manim_visuals/ebob_visualization.py

from manim import *

class EBOBVisualization(Scene):
    """
    EBOB kavramini gorsel olarak aciklar
    """
    
    def construct(self):
        # Baslik
        title = Text("EBOB Nedir?", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Iki sayi
        num1 = Text("24", font_size=72, color=BLUE)
        num2 = Text("36", font_size=72, color=GREEN)
        nums = VGroup(num1, num2).arrange(RIGHT, buff=2)
        
        self.play(FadeIn(nums))
        self.wait(1)
        
        # Carpanlar
        factors1 = Text("24 = 2 x 2 x 2 x 3", font_size=36).next_to(num1, DOWN)
        factors2 = Text("36 = 2 x 2 x 3 x 3", font_size=36).next_to(num2, DOWN)
        
        self.play(Write(factors1), Write(factors2))
        self.wait(2)
        
        # Ortak carpanlar vurgula
        common = Text("Ortak: 2 x 2 x 3 = 12", font_size=42, color=YELLOW)
        common.to_edge(DOWN)
        
        self.play(Write(common))
        
        # EBOB sonucu
        result = Text("EBOB(24, 36) = 12", font_size=48, color=RED)
        result.next_to(common, UP)
        
        self.play(
            FadeIn(result, scale=1.5),
            Flash(result, color=RED)
        )
        self.wait(2)


class CarpanAgaci(Scene):
    """
    Bir sayinin carpan agacini gosterir
    """
    
    def construct(self):
        # 60 sayisinin carpan agaci
        title = Text("60'in Asal Carpanlari", font_size=42)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Agac yapisi
        #        60
        #       /  \
        #      2   30
        #         /  \
        #        2   15
        #           /  \
        #          3    5
        
        nodes = {
            "60": Dot(ORIGIN + UP*2, color=WHITE),
            "2a": Dot(LEFT*2 + UP, color=RED),
            "30": Dot(RIGHT*2 + UP, color=WHITE),
            "2b": Dot(RIGHT + DOWN*0, color=RED),
            "15": Dot(RIGHT*3 + DOWN*0, color=WHITE),
            "3": Dot(RIGHT*2 + DOWN*1.5, color=RED),
            "5": Dot(RIGHT*4 + DOWN*1.5, color=RED),
        }
        
        labels = {
            "60": Text("60", font_size=32).next_to(nodes["60"], UP),
            "2a": Text("2", font_size=32, color=RED).next_to(nodes["2a"], LEFT),
            "30": Text("30", font_size=32).next_to(nodes["30"], RIGHT),
            "2b": Text("2", font_size=32, color=RED).next_to(nodes["2b"], LEFT),
            "15": Text("15", font_size=32).next_to(nodes["15"], RIGHT),
            "3": Text("3", font_size=32, color=RED).next_to(nodes["3"], DOWN),
            "5": Text("5", font_size=32, color=RED).next_to(nodes["5"], DOWN),
        }
        
        # Animasyonla goster
        self.play(Create(nodes["60"]), Write(labels["60"]))
        self.wait(0.5)
        
        # 60 = 2 x 30
        line1 = Line(nodes["60"].get_center(), nodes["2a"].get_center())
        line2 = Line(nodes["60"].get_center(), nodes["30"].get_center())
        self.play(Create(line1), Create(line2))
        self.play(Create(nodes["2a"]), Write(labels["2a"]))
        self.play(Create(nodes["30"]), Write(labels["30"]))
        
        # Devam...
        self.wait(2)
        
        # Sonuc
        result = Text("60 = 2 x 2 x 3 x 5 = 2² x 3 x 5", font_size=36)
        result.to_edge(DOWN)
        self.play(Write(result))
        self.wait(2)
```

### 17.6.4 LGS Soru Gorseli Uretimi

```python
# src/manim_visuals/question_visual.py

from manim import *

class LGSSoruGorseli(Scene):
    """
    LGS sorusu icin gorsel uretir
    """
    
    def __init__(self, soru_data: dict, **kwargs):
        super().__init__(**kwargs)
        self.soru = soru_data
    
    def construct(self):
        # Soru metni
        soru_text = Text(
            self.soru.get("soru", "Soru metni"),
            font_size=28,
            line_spacing=1.2
        ).to_edge(UP)
        
        self.play(Write(soru_text))
        
        # Gorsel tipi: tablo
        if self.soru.get("gorsel_tipi") == "tablo":
            self._create_table()
        elif self.soru.get("gorsel_tipi") == "sematik":
            self._create_schema()
        
        # Secenekler
        self._show_options()
    
    def _create_table(self):
        """Tablo gorseli olustur"""
        table = Table(
            [["Urun", "Miktar"],
             ["Elma", "24"],
             ["Armut", "36"]],
            include_outer_lines=True
        ).scale(0.6)
        
        self.play(Create(table))
    
    def _create_schema(self):
        """Sematik gorsel olustur"""
        # Ok ve kutu ile sema
        box1 = Rectangle(width=2, height=1).shift(LEFT*3)
        box2 = Rectangle(width=2, height=1).shift(RIGHT*3)
        arrow = Arrow(box1.get_right(), box2.get_left())
        
        self.play(Create(box1), Create(box2), Create(arrow))
    
    def _show_options(self):
        """Secenekleri goster"""
        options = VGroup(
            Text("A) 12", font_size=24),
            Text("B) 18", font_size=24),
            Text("C) 24", font_size=24),
            Text("D) 36", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN)
        
        self.play(FadeIn(options))


# Render fonksiyonu
def render_question_visual(soru_data: dict, output_path: str):
    """
    Soru icin gorsel render et
    
    Args:
        soru_data: Soru JSON verisi
        output_path: Cikti dosya yolu
    """
    from manim import config
    
    config.output_file = output_path
    config.pixel_width = 1920
    config.pixel_height = 1080
    config.frame_rate = 30
    
    scene = LGSSoruGorseli(soru_data)
    scene.render()
```

### 17.6.5 RAG + Manim Entegrasyonu

```python
# src/services/visual_generator.py

import json
import subprocess
from pathlib import Path

class VisualGenerator:
    """
    Uretilen sorular icin Manim gorselleri olusturur
    """
    
    def __init__(self, output_dir: str = "generated_visuals"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_visual(self, question: dict) -> str:
        """
        Soru icin gorsel uret
        
        Args:
            question: Uretilen soru JSON'i
        
        Returns:
            Gorsel dosya yolu
        """
        gorsel_tipi = question.get("gorsel_tipi", "yok")
        
        if gorsel_tipi == "yok":
            return None
        
        # Manim scene dosyasi olustur
        scene_file = self._create_scene_file(question)
        
        # Render et
        output_file = self.output_dir / f"soru_{question.get('id', 'temp')}.mp4"
        
        subprocess.run([
            "manim", "render",
            "-ql",  # Low quality (hizli)
            scene_file,
            "LGSSoruGorseli",
            "-o", str(output_file)
        ])
        
        return str(output_file)
    
    def _create_scene_file(self, question: dict) -> str:
        """Gecici scene dosyasi olustur"""
        # ...
        pass
```

### 17.6.6 Manim Cikti Formatlari

| Format | Komut | Kullanim |
|--------|-------|----------|
| MP4 Video | `manim render -ql scene.py` | Animasyonlu gorsel |
| GIF | `manim render -ql --format gif` | Web icin |
| PNG | `manim render -s` | Statik gorsel |
| SVG | `--format svg` | Vektorel |

### 17.6.7 AR Entegrasyonu Icin Gorsel Export

```python
def export_for_unity(visual_path: str, unity_assets_path: str):
    """
    Manim gorselini Unity Assets klasorune kopyala
    
    Bu gorsel, AR uygulamasinda soru gosterimi icin kullanilabilir.
    """
    import shutil
    
    dest = Path(unity_assets_path) / "GeneratedVisuals"
    dest.mkdir(exist_ok=True)
    
    shutil.copy(visual_path, dest)
```

---

## 18. RAG Katmani Taskleri

> **Not:** Bu bolum RAG katmaninin 7 ana taskini detayli olarak dokumante eder.

### 18.1 Task Ozet Tablosu

| Task ID | Task Adi | Girdi | Cikti | Bagimlilk |
|---------|----------|-------|-------|-----------|
| R-01 | Embedding Pipeline Kurulumu | `dataset_ocr_li.csv` | FAISS index | M-06 |
| R-02 | Filtreleme Sistemi | `configs.json`, metadata | Filtrelenmis sorular | M-11, R-01 |
| R-03 | Benzer Soru Getirme Modulu | Kombinasyon, index | Top-K sorular | R-01, R-02 |
| R-04 | RAG Ciktisinin Formatlenmasi | Top-K sorular | Formatli paket | R-03 |
| R-05 | Prompt Tasarimi | Sablonlar | Prompt template'leri | - |
| R-06 | Soru Uretim Pipeline'i | Config, examples | Uretilmis soru JSON | R-03, R-04, R-05 |
| R-07 | Cesitlilik Mekanizmasi | Pipeline | Varyasyonlu uretim | R-06 |
| R-08 | LangChain Entegrasyonu | Soru CSV | LangChain RAG | R-01 |
| R-09 | Manim Gorsel Uretimi | Uretilen soru | MP4/PNG gorsel | R-06 |

### 18.2 Task Akis Diyagrami

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG TASK AKISI                                      │
└─────────────────────────────────────────────────────────────────────────────┘

MODEL CIKTILARI                          RAG PIPELINE
═══════════════                          ════════════

┌─────────────────┐
│ dataset_ocr_li  │
│ .csv            │────────┐
└─────────────────┘        │
                           ▼
                    ┌──────────────────┐
                    │ R-01: Embedding  │
                    │ Pipeline         │
                    │                  │
                    │ - Model secimi   │
                    │ - Vektorlestirme │
                    │ - FAISS index    │
                    └────────┬─────────┘
                             │
┌─────────────────┐          │
│ configs.json    │──────────┼──────────────┐
└─────────────────┘          │              │
                             ▼              ▼
                    ┌──────────────────┐   ┌──────────────────┐
                    │ R-02: Filtreleme │   │ R-05: Prompt     │
                    │ Sistemi          │   │ Tasarimi         │
                    │                  │   │                  │
                    │ - Alt konu       │   │ - System prompt  │
                    │ - Zorluk         │   │ - User prompt    │
                    │ - Gorsel tipi    │   │ - Format rules   │
                    └────────┬─────────┘   └────────┬─────────┘
                             │                      │
                             ▼                      │
                    ┌──────────────────┐            │
                    │ R-03: Benzer     │            │
                    │ Soru Getirme     │            │
                    │                  │            │
                    │ - retrieve_      │            │
                    │   examples()     │            │
                    │ - Top-K secim    │            │
                    └────────┬─────────┘            │
                             │                      │
                             ▼                      │
                    ┌──────────────────┐            │
                    │ R-04: RAG        │            │
                    │ Cikti Formati    │            │
                    │                  │            │
                    │ - JSON/Text      │            │
                    │ - Metadata       │            │
                    └────────┬─────────┘            │
                             │                      │
                             └──────────┬───────────┘
                                        │
                                        ▼
                             ┌──────────────────┐
                             │ R-06: Soru       │
                             │ Uretim Pipeline  │
                             │                  │
                             │ - generate_      │
                             │   question()     │
                             │ - LLM cagri      │
                             └────────┬─────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │ R-07: Cesitlilik │
                             │ Mekanizmasi      │
                             │                  │
                             │ - Randomness     │
                             │ - Temperature    │
                             │ - Varyasyonlar   │
                             └────────┬─────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │ URETILMIS SORU   │
                             │ (JSON)           │
                             └──────────────────┘
```

### 18.3 Task Detaylari

---

#### R-01: Embedding Pipeline Kurulumu

**Amac:** Embedding modeli secme, tum soru metinlerini embedding'leme ve FAISS gibi bir index sistemi kurarak arama yapisini hazirlama.

**Alt Gorevler:**
1. Turkce embedding modeli secimi ve benchmark
2. Soru metinlerini vektorlestirme
3. FAISS index olusturma ve kaydetme
4. Index yukleme ve arama fonksiyonlari

**Beklenen Implementasyon:**

```python
# src/services/embedding_service.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class EmbeddingPipeline:
    def __init__(self, model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
        """
        Turkce embedding modeli yukle
        
        Alternatif modeller:
        - dbmdz/bert-base-turkish-cased
        - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.id_to_metadata = {}
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Metin listesini vektorlestir"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Cosine similarity icin
        )
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        FAISS index olustur
        
        Args:
            embeddings: (N, D) numpy array
            metadata: Her embedding icin metadata listesi
        """
        # Flat index (kucuk veri seti icin yeterli)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine icin normalize edilmis)
        
        # Embedding'leri ekle
        self.index.add(embeddings.astype('float32'))
        
        # Metadata mapping
        self.id_to_metadata = {i: meta for i, meta in enumerate(metadata)}
        
        print(f"Index olusturuldu: {self.index.ntotal} vektor")
    
    def save_index(self, index_path: str, metadata_path: str):
        """Index ve metadata'yi kaydet"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
    
    def load_index(self, index_path: str, metadata_path: str):
        """Index ve metadata'yi yukle"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.id_to_metadata = pickle.load(f)
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        """
        Benzer sorulari ara
        
        Returns:
            [(score, metadata), ...] listesi
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Gecerli sonuc
                results.append((float(score), self.id_to_metadata[idx]))
        
        return results
```

**Kullanim Ornegi:**

```python
# Index olusturma (bir kez)
import pandas as pd

df = pd.read_csv("lgs-model/data/processed/dataset_ocr_li.csv")

pipeline = EmbeddingPipeline()

# Embedding
texts = df["Soru_MetniOCR"].tolist()
embeddings = pipeline.embed_texts(texts)

# Metadata hazirla
metadata = df.to_dict('records')

# Index olustur ve kaydet
pipeline.build_index(embeddings, metadata)
pipeline.save_index("vectorstore/faiss.index", "vectorstore/metadata.pkl")
```

**Cikti Dosyalari:**
| Dosya | Konum | Aciklama |
|-------|-------|----------|
| `faiss.index` | `lgs-rag/vectorstore/` | FAISS index dosyasi |
| `metadata.pkl` | `lgs-rag/vectorstore/` | Soru metadata'lari |

**Kabul Kriterleri:**
- [x] ✅ Turkce embedding modeli secildi ve yuklendi
- [x] ✅ 176 soru vektorlestirildi (CSV'den)
- [x] ✅ FAISS index olusturuldu
- [x] ✅ Index kaydetme/yukleme calisiyor
- [x] ✅ Arama fonksiyonu calisiyor

---

#### R-02: Filtreleme Sistemi

**Amac:** Alt konu, zorluk ve gorsel tipine gore RAG oncesi daraltici bir filtreleme katmani olusturma.

**Beklenen Implementasyon:**

```python
# src/services/filter_service.py

from typing import Optional
from dataclasses import dataclass

@dataclass
class FilterCriteria:
    alt_konu: Optional[str] = None
    zorluk: Optional[int] = None
    zorluk_range: Optional[tuple[int, int]] = None  # (min, max)
    gorsel_tipi: Optional[str] = None
    kaynak_tipi: Optional[list[str]] = None  # ["cikmis", "ornek"]
    is_lgs: Optional[int] = None

class FilterService:
    def __init__(self, metadata_list: list[dict]):
        """
        Metadata listesi ile basla
        
        Args:
            metadata_list: [{alt_konu, zorluk, gorsel_tipi, ...}, ...]
        """
        self.all_metadata = metadata_list
    
    def filter(self, criteria: FilterCriteria) -> list[dict]:
        """
        Kriterlere gore metadata'lari filtrele
        
        Returns:
            Filtrelenmis metadata listesi
        """
        results = self.all_metadata.copy()
        
        if criteria.alt_konu:
            results = [m for m in results if m.get("Alt_Konu") == criteria.alt_konu]
        
        if criteria.zorluk:
            results = [m for m in results if m.get("Zorluk") == criteria.zorluk]
        
        if criteria.zorluk_range:
            min_z, max_z = criteria.zorluk_range
            results = [m for m in results if min_z <= m.get("Zorluk", 0) <= max_z]
        
        if criteria.gorsel_tipi:
            results = [m for m in results if m.get("Gorsel_Tipi") == criteria.gorsel_tipi]
        
        if criteria.kaynak_tipi:
            results = [m for m in results if m.get("Kaynak_Tipi") in criteria.kaynak_tipi]
        
        if criteria.is_lgs is not None:
            results = [m for m in results if m.get("is_LGS") == criteria.is_lgs]
        
        return results
    
    def filter_indices(self, criteria: FilterCriteria) -> list[int]:
        """
        Filtrelenmis metadata'larin index'lerini don
        (FAISS ile kullanmak icin)
        """
        indices = []
        for i, meta in enumerate(self.all_metadata):
            if self._matches(meta, criteria):
                indices.append(i)
        return indices
    
    def _matches(self, meta: dict, criteria: FilterCriteria) -> bool:
        """Tek metadata'nin kriterlere uyup uymadigini kontrol et"""
        if criteria.alt_konu and meta.get("Alt_Konu") != criteria.alt_konu:
            return False
        if criteria.zorluk and meta.get("Zorluk") != criteria.zorluk:
            return False
        if criteria.zorluk_range:
            z = meta.get("Zorluk", 0)
            if not (criteria.zorluk_range[0] <= z <= criteria.zorluk_range[1]):
                return False
        if criteria.gorsel_tipi and meta.get("Gorsel_Tipi") != criteria.gorsel_tipi:
            return False
        if criteria.kaynak_tipi and meta.get("Kaynak_Tipi") not in criteria.kaynak_tipi:
            return False
        if criteria.is_lgs is not None and meta.get("is_LGS") != criteria.is_lgs:
            return False
        return True
```

**Filtreleme Stratejisi:**

```python
def create_retrieval_filter(combination: dict) -> FilterCriteria:
    """
    Kombinasyona gore optimal filtre olustur
    
    Strateji:
    1. Ayni alt_konu (zorunlu)
    2. Zorluk: hedef +-1 (esneklik)
    3. Gorsel tipi: tercih edilir ama zorunlu degil
    4. Kaynak: cikmis > ornek > baslangic (siralama icin)
    """
    return FilterCriteria(
        alt_konu=combination["alt_konu"],
        zorluk_range=(
            max(1, combination["zorluk"] - 1),
            min(5, combination["zorluk"] + 1)
        ),
        gorsel_tipi=None,  # Esneklik icin
        kaynak_tipi=["cikmis", "ornek"],  # LGS profili oncelikli
        is_lgs=1
    )
```

**Kabul Kriterleri:**
- [x] ✅ FilterCriteria dataclass olusturuldu
- [x] ✅ Tum filtre kriterleri calisiyor
- [x] ✅ Index-based filtreleme calisiyor
- [x] ✅ Bos sonuc durumu handle ediliyor
- [ ] ⏳ Performans: <10ms filtreleme (benchmark gerekli)

---

#### R-03: Benzer Soru Getirme Modulu

**Amac:** `retrieve_examples(config)` fonksiyonunu yazarak konfig'e uygun top-3/top-5 benzer soruyu metadata ile birlikte dondurme.

**Beklenen Implementasyon:**

```python
# src/services/retriever.py

from typing import Optional
from dataclasses import dataclass

@dataclass
class RetrievedQuestion:
    soru_metni: str
    alt_konu: str
    zorluk: int
    gorsel_tipi: str
    kaynak_tipi: str
    similarity_score: float
    metadata: dict

class QuestionRetriever:
    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        filter_service: FilterService
    ):
        self.embedding = embedding_pipeline
        self.filter = filter_service
    
    def retrieve_examples(
        self,
        combination: dict,
        top_k: int = 5,
        fallback_top_k: int = 8
    ) -> list[RetrievedQuestion]:
        """
        Kombinasyona uygun ornek sorulari getir
        
        Args:
            combination: {alt_konu, zorluk, gorsel_tipi, lgs_skor}
            top_k: Dondurelecek soru sayisi
            fallback_top_k: Yeterli sonuc yoksa genisletilmis arama
        
        Returns:
            RetrievedQuestion listesi (sirali)
        
        Strateji:
        1. Oncelikle tam eslesme ara
        2. Yeterli degilse zorluk araligini genislet
        3. Hala yeterli degilse gorsel tipi kisitini kaldir
        """
        
        # Seviye 1: Tam eslesme + LGS profili
        filter_criteria = FilterCriteria(
            alt_konu=combination["alt_konu"],
            zorluk=combination["zorluk"],
            gorsel_tipi=combination["gorsel_tipi"],
            is_lgs=1
        )
        results = self._search_with_filter(filter_criteria, top_k)
        
        if len(results) >= top_k:
            return self._rerank(results, combination)[:top_k]
        
        # Seviye 2: Zorluk +-1
        filter_criteria = FilterCriteria(
            alt_konu=combination["alt_konu"],
            zorluk_range=(
                max(1, combination["zorluk"] - 1),
                min(5, combination["zorluk"] + 1)
            ),
            gorsel_tipi=combination["gorsel_tipi"],
            is_lgs=1
        )
        results = self._search_with_filter(filter_criteria, fallback_top_k)
        
        if len(results) >= top_k:
            return self._rerank(results, combination)[:top_k]
        
        # Seviye 3: Sadece alt_konu + LGS
        filter_criteria = FilterCriteria(
            alt_konu=combination["alt_konu"],
            is_lgs=1
        )
        results = self._search_with_filter(filter_criteria, fallback_top_k)
        
        if len(results) >= 3:  # Minimum 3 ornek
            return self._rerank(results, combination)[:top_k]
        
        # Seviye 4: Sadece alt_konu (baslangic dahil)
        filter_criteria = FilterCriteria(
            alt_konu=combination["alt_konu"]
        )
        results = self._search_with_filter(filter_criteria, fallback_top_k)
        
        return self._rerank(results, combination)[:top_k]
    
    def _search_with_filter(
        self,
        criteria: FilterCriteria,
        top_k: int
    ) -> list[RetrievedQuestion]:
        """Filtreli arama yap"""
        
        # Filtre uygula
        valid_indices = self.filter.filter_indices(criteria)
        
        if not valid_indices:
            return []
        
        # Filtrelenmis metadata'lardan temsili sorgu olustur
        # (veya tum indeksler uzerinde arama yap)
        filtered_metadata = [
            self.embedding.id_to_metadata[i] 
            for i in valid_indices
        ]
        
        # En yuksek skorlu sorulari sec
        results = []
        for meta in filtered_metadata[:top_k]:
            results.append(RetrievedQuestion(
                soru_metni=meta.get("Soru_MetniOCR", ""),
                alt_konu=meta.get("Alt_Konu", ""),
                zorluk=meta.get("Zorluk", 0),
                gorsel_tipi=meta.get("Gorsel_Tipi", ""),
                kaynak_tipi=meta.get("Kaynak_Tipi", ""),
                similarity_score=1.0,  # Filtreleme-based
                metadata=meta
            ))
        
        return results
    
    def _rerank(
        self,
        results: list[RetrievedQuestion],
        combination: dict
    ) -> list[RetrievedQuestion]:
        """
        Sonuclari yeniden sirala
        
        Siralama kriterleri:
        1. Kaynak tipi: cikmis (1.0) > ornek (0.8) > baslangic (0.3)
        2. Zorluk yakinligi
        3. Gorsel tipi eslesmesi
        """
        def score(q: RetrievedQuestion) -> float:
            s = 0.0
            
            # Kaynak agirligi
            kaynak_weights = {"cikmis": 1.0, "ornek": 0.8, "baslangic": 0.3}
            s += kaynak_weights.get(q.kaynak_tipi, 0.5) * 0.4
            
            # Zorluk yakinligi (0-1 arasi, 0 = ayni)
            zorluk_diff = abs(q.zorluk - combination["zorluk"])
            s += (1 - zorluk_diff / 4) * 0.3
            
            # Gorsel tipi eslesmesi
            if q.gorsel_tipi == combination["gorsel_tipi"]:
                s += 0.3
            
            return s
        
        return sorted(results, key=score, reverse=True)
```

**Kabul Kriterleri:**
- [x] ✅ `retrieve_examples()` fonksiyonu calisiyor
- [x] ✅ 4 seviyeli fallback stratejisi implemente edildi
- [x] ✅ Reranking calisiyor
- [x] ✅ Minimum 3 ornek garantisi var
- [x] ✅ Bos sonuc durumu handle ediliyor

---

#### R-04: RAG Ciktisinin Formatlenmasi

**Amac:** LLM'e verilecek ornek soru paketini temiz ve duzenli bir formatta hazirlama (JSON veya text).

**Beklenen Implementasyon:**

```python
# src/services/output_formatter.py

from typing import Literal

class RAGOutputFormatter:
    def __init__(self, format_type: Literal["json", "markdown", "text"] = "markdown"):
        self.format_type = format_type
    
    def format_examples(
        self,
        examples: list[RetrievedQuestion],
        combination: dict
    ) -> str:
        """
        Ornekleri LLM icin formatla
        
        Args:
            examples: Retrieve edilen sorular
            combination: Hedef kombinasyon
        
        Returns:
            Formatli string
        """
        if self.format_type == "json":
            return self._format_as_json(examples, combination)
        elif self.format_type == "markdown":
            return self._format_as_markdown(examples, combination)
        else:
            return self._format_as_text(examples, combination)
    
    def _format_as_markdown(
        self,
        examples: list[RetrievedQuestion],
        combination: dict
    ) -> str:
        """Markdown formatinda cikti"""
        
        output = []
        output.append("## Hedef Kombinasyon")
        output.append(f"- **Alt Konu:** {combination['alt_konu']}")
        output.append(f"- **Zorluk:** {combination['zorluk']}/5")
        output.append(f"- **Gorsel Tipi:** {combination['gorsel_tipi']}")
        output.append(f"- **LGS Skoru:** {combination.get('lgs_skor', 'N/A')}")
        output.append("")
        output.append("## Ornek Sorular")
        output.append("")
        
        for i, ex in enumerate(examples, 1):
            output.append(f"### Ornek {i}")
            output.append(f"**Kaynak:** {ex.kaynak_tipi.upper()}")
            output.append(f"**Zorluk:** {ex.zorluk}/5")
            output.append(f"**Gorsel:** {ex.gorsel_tipi}")
            output.append("")
            output.append(f"**Soru:**")
            output.append(f"> {ex.soru_metni}")
            output.append("")
            output.append("---")
            output.append("")
        
        return "\n".join(output)
    
    def _format_as_json(
        self,
        examples: list[RetrievedQuestion],
        combination: dict
    ) -> str:
        """JSON formatinda cikti"""
        import json
        
        data = {
            "hedef_kombinasyon": combination,
            "ornek_sorular": [
                {
                    "soru_metni": ex.soru_metni,
                    "alt_konu": ex.alt_konu,
                    "zorluk": ex.zorluk,
                    "gorsel_tipi": ex.gorsel_tipi,
                    "kaynak_tipi": ex.kaynak_tipi
                }
                for ex in examples
            ]
        }
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _format_as_text(
        self,
        examples: list[RetrievedQuestion],
        combination: dict
    ) -> str:
        """Duz text formatinda cikti"""
        
        lines = []
        lines.append(f"HEDEF: {combination['alt_konu']} | Zorluk {combination['zorluk']} | {combination['gorsel_tipi']}")
        lines.append("")
        lines.append("ORNEK SORULAR:")
        lines.append("=" * 50)
        
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n[{i}] ({ex.kaynak_tipi}, Z:{ex.zorluk})")
            lines.append(ex.soru_metni)
            lines.append("-" * 30)
        
        return "\n".join(lines)
```

**Kabul Kriterleri:**
- [x] ✅ 3 format tipi destekleniyor (JSON, Markdown, Text)
- [x] ✅ Hedef kombinasyon bilgisi dahil
- [x] ✅ Ornek sorular temiz formatlanmis
- [x] ✅ Turkce karakterler dogru handle ediliyor

---

#### R-05: Prompt Tasarimi

**Amac:** LGS tarzi, gorsel aciklamali, hikayeli soru uretimi icin prompt sablonlari olusturma ve stil transferi ayarlari yapma.

**Beklenen Implementasyon:**

```python
# src/services/prompt_builder.py

from typing import Optional

class PromptBuilder:
    def __init__(self):
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """LGS soru yazari sistem promptu"""
        return """Sen, LGS (Liseye Gecis Sinavi) matematik sorulari yazan uzman bir egitimcisin.

## Rol ve Gorev
- 8. sinif ogrencilerine yonelik matematik sorulari yaziyorsun
- Sorular MEB mufredat ve LGS formatlarina tam uygun olmali
- Her soru tek dogru cevapli, 4 secenekli (A, B, C, D) olmali

## Konu Alani: Carpanlar ve Katlar
Asagidaki alt konularda soru yazabilirsin:
1. **carpanlar**: Bir dogal sayinin carpanlari, asal carpanlar, carpan sayisi
2. **ebob_ekok**: En buyuk ortak bolen, en kucuk ortak kat, problemler
3. **aralarinda_asal**: Aralarinda asal sayilar, ozellikler

## LGS Soru Ozellikleri
1. **Hikaye/Baglam**: Gercek hayat senaryolari (alisveris, paylasim, zaman, geometri)
2. **Cok Adimli**: 2-4 adim cozum gerektiren yapilar
3. **Yaniltici Secenekler**: Dikkat gerektiren, mantikli yanlis secenekler
4. **Gorsel Destek**: Gerektiginde tablo, sema, sekil aciklamasi

## Zorluk Seviyeleri
| Seviye | Aciklama | Adim Sayisi |
|--------|----------|-------------|
| 1 | Temel kavram, dogrudan uygulama | 1 |
| 2 | Iki islemli, basit problem | 2 |
| 3 | Orta karmasiklik, mantik gerektiren | 2-3 |
| 4 | Ileri duzey, cok adimli | 3-4 |
| 5 | Olimpiyat tarzi, en zor | 4+ |

## Gorsel Tipi Kurallari
- **yok**: Gorsel aciklama verme
- **tablo**: Satirlar, sutunlar ve basliklar iceren tablo tanimla
- **sematik**: Ok, kutu, iliskiler iceren sema tanimla
- **geometrik_sekil**: Uzunluk, alan, cevre iceren sekil tanimla
- **resimli**: Gercek hayat objelerini icerecek sekilde tanimla

## Cikti Kurallari
1. JSON formatinda yanit ver
2. Turkce dil kurallarina uy
3. Matematiksel notasyon dogru kullan (^, sqrt, x, vb.)
4. Secenekler arasi mantikli aralik birak
5. Cozum adimlarini acik yaz"""
    
    def build_user_prompt(
        self,
        combination: dict,
        formatted_examples: str,
        additional_instructions: Optional[str] = None
    ) -> str:
        """Kullanici promptu olustur"""
        
        prompt_parts = []
        
        # Gorev tanimi
        prompt_parts.append("Asagidaki kombinasyona uygun YENI ve OZGUN bir LGS matematik sorusu uret.")
        prompt_parts.append("")
        
        # Ornekler
        prompt_parts.append(formatted_examples)
        prompt_parts.append("")
        
        # Ozel talimatlar
        if additional_instructions:
            prompt_parts.append("## Ek Talimatlar")
            prompt_parts.append(additional_instructions)
            prompt_parts.append("")
        
        # Cikti formati
        prompt_parts.append("## Beklenen JSON Ciktisi")
        prompt_parts.append("""
```json
{
  "hikaye": "Sorunun gercek hayat baglami/hikayesi...",
  "soru": "Asil soru metni (Buna gore... ile bitebilir)",
  "gorsel_aciklama": "Gorsel tipi 'yok' degilse detayli gorsel tanimi",
  "secenekler": {
    "A": "Secenek A degeri",
    "B": "Secenek B degeri",
    "C": "Secenek C degeri",
    "D": "Secenek D degeri"
  },
  "dogru_cevap": "A/B/C/D",
  "cozum": [
    "Adim 1: Ilk islem aciklamasi",
    "Adim 2: Ikinci islem aciklamasi",
    "Adim 3: Sonuc"
  ]
}
```""")
        
        prompt_parts.append("")
        prompt_parts.append("## Onemli Uyarilar")
        prompt_parts.append("- Orneklerden FARKLI, tamamen yeni bir soru uret")
        prompt_parts.append("- Sayilari, hikayeyi ve baglami degistir")
        prompt_parts.append(f"- Zorluk seviyesi {combination['zorluk']}/5 olmali")
        
        if combination["gorsel_tipi"] != "yok":
            prompt_parts.append(f"- {combination['gorsel_tipi']} tipinde gorsel icin DETAYLI aciklama ver")
        else:
            prompt_parts.append("- gorsel_aciklama alanini bos birak veya null yap")
        
        prompt_parts.append("- Sadece JSON ciktisi ver, baska aciklama ekleme")
        
        return "\n".join(prompt_parts)
    
    def get_style_variations(self) -> list[str]:
        """
        Cesitlilik icin stil varyasyonlari
        """
        return [
            "Hikayede bir ogrenci senaryosu kullan.",
            "Hikayede bir is yeri/fabrika senaryosu kullan.",
            "Hikayede bir spor/oyun senaryosu kullan.",
            "Hikayede bir aile/ev senaryosu kullan.",
            "Hikayede bir okul/sinif senaryosu kullan.",
            "Sayilari 100'den buyuk sec.",
            "Sayilari 20'den kucuk sec.",
            "Negatif bir durum iceren senaryo kullan (eksik, kayip, vb.).",
            "Karsilastirma iceren bir senaryo kullan.",
            "Zaman iceren bir senaryo kullan (gun, saat, dakika)."
        ]
```

**Prompt Sablonu Ornegi:**

```markdown
## Hedef Kombinasyon
- Alt Konu: ebob_ekok
- Zorluk: 4/5
- Gorsel Tipi: sematik

## Ornek Sorular
[5 adet ornek soru...]

## Ek Talimatlar
Hikayede bir fabrika senaryosu kullan.

## Beklenen JSON Ciktisi
[Format...]
```

**Kabul Kriterleri:**
- [x] ✅ System prompt LGS formatina uygun
- [x] ✅ User prompt dinamik olarak olusturuluyor
- [x] ✅ Stil varyasyonlari mevcut
- [x] ✅ JSON cikti formati tanimli
- [x] ✅ Turkce karakterler dogru

---

#### R-06: Soru Uretim Pipeline'i

**Amac:** `generate_question(config, examples)` fonksiyonunu gelistirerek JSON -> RAG -> Prompt -> LLM akisini gerceklestiren pipeline'i kurma.

**Beklenen Implementasyon:**

```python
# src/services/question_generator.py

import json
from typing import Optional
from dataclasses import dataclass

@dataclass
class GeneratedQuestion:
    hikaye: str
    soru: str
    gorsel_aciklama: Optional[str]
    secenekler: dict[str, str]
    dogru_cevap: str
    cozum: list[str]
    combination: dict
    metadata: dict

class QuestionGenerator:
    def __init__(
        self,
        retriever: QuestionRetriever,
        formatter: RAGOutputFormatter,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient
    ):
        self.retriever = retriever
        self.formatter = formatter
        self.prompt_builder = prompt_builder
        self.llm = llm_client
    
    async def generate_question(
        self,
        combination: dict,
        style_instruction: Optional[str] = None,
        max_attempts: int = 3
    ) -> GeneratedQuestion:
        """
        Tek soru uret
        
        Pipeline:
        1. Benzer sorulari retrieve et
        2. Ornekleri formatla
        3. Prompt olustur
        4. LLM cagri
        5. Parse ve validate
        """
        
        # Step 1: Retrieval
        examples = self.retriever.retrieve_examples(combination, top_k=5)
        
        if len(examples) < 3:
            raise InsufficientExamplesError(
                f"Yetersiz ornek: {len(examples)} bulundu, minimum 3 gerekli"
            )
        
        # Step 2: Format
        formatted_examples = self.formatter.format_examples(examples, combination)
        
        # Step 3: Prompt
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(
            combination=combination,
            formatted_examples=formatted_examples,
            additional_instructions=style_instruction
        )
        
        # Step 4 & 5: Generate with retry
        for attempt in range(max_attempts):
            try:
                # LLM cagri
                response = await self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7 + (attempt * 0.1)  # Her denemede artir
                )
                
                # Parse JSON
                question_data = self._parse_response(response)
                
                # Validate
                self._validate_question(question_data, combination)
                
                # Return
                return GeneratedQuestion(
                    hikaye=question_data["hikaye"],
                    soru=question_data["soru"],
                    gorsel_aciklama=question_data.get("gorsel_aciklama"),
                    secenekler=question_data["secenekler"],
                    dogru_cevap=question_data["dogru_cevap"],
                    cozum=question_data["cozum"],
                    combination=combination,
                    metadata={
                        "attempt": attempt + 1,
                        "retrieval_count": len(examples),
                        "style_instruction": style_instruction
                    }
                )
            
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_attempts - 1:
                    raise GenerationFailedError(f"Max attempts reached: {e}")
                continue
        
        raise GenerationFailedError("Unexpected error in generation loop")
    
    def _parse_response(self, response: str) -> dict:
        """LLM cevabindan JSON parse et"""
        
        # JSON blogu bul
        response = response.strip()
        
        # ```json ... ``` formatini handle et
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end]
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end]
        
        return json.loads(response)
    
    def _validate_question(self, data: dict, combination: dict):
        """Uretilen soruyu dogrula"""
        
        errors = []
        
        # Zorunlu alanlar
        required_fields = ["hikaye", "soru", "secenekler", "dogru_cevap", "cozum"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Eksik alan: {field}")
        
        # Secenek kontrolu
        if "secenekler" in data:
            if len(data["secenekler"]) != 4:
                errors.append(f"4 secenek olmali, {len(data['secenekler'])} var")
            
            expected_keys = {"A", "B", "C", "D"}
            if set(data["secenekler"].keys()) != expected_keys:
                errors.append("Secenekler A, B, C, D olmali")
        
        # Dogru cevap kontrolu
        if "dogru_cevap" in data:
            if data["dogru_cevap"] not in ["A", "B", "C", "D"]:
                errors.append(f"Gecersiz dogru_cevap: {data['dogru_cevap']}")
            
            if "secenekler" in data and data["dogru_cevap"] not in data["secenekler"]:
                errors.append("dogru_cevap seceneklerde yok")
        
        # Cozum kontrolu
        if "cozum" in data:
            if not isinstance(data["cozum"], list) or len(data["cozum"]) < 1:
                errors.append("cozum en az 1 adim icermeli")
        
        # Gorsel kontrolu
        if combination["gorsel_tipi"] != "yok":
            if not data.get("gorsel_aciklama"):
                errors.append(f"gorsel_aciklama gerekli ({combination['gorsel_tipi']})")
        
        if errors:
            raise ValidationError(", ".join(errors))
```

**Kullanim Ornegi:**

```python
# Tekil uretim
question = await generator.generate_question(
    combination={
        "alt_konu": "ebob_ekok",
        "zorluk": 4,
        "gorsel_tipi": "sematik",
        "lgs_skor": 0.87
    },
    style_instruction="Hikayede bir fabrika senaryosu kullan."
)

print(question.hikaye)
print(question.soru)
print(question.secenekler)
```

**Kabul Kriterleri:**
- [x] ✅ `generate_question()` async fonksiyon calisiyor
- [x] ✅ 5 adimli pipeline implemente edildi
- [x] ✅ Retry mekanizmasi calisiyor
- [x] ✅ JSON parsing robust
- [x] ✅ Validation kapsamli

---

#### R-07: Cesitlilik (Randomness) Mekanizmasi

**Amac:** Ayni konfigurasyondan farkli sorular uretmek icin randomness, temperature ve prompt varyasyonlari gelistirme.

**Beklenen Implementasyon:**

```python
# src/services/diversity_service.py

import random
from typing import Optional

class DiversityService:
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
        self.used_styles = []  # Son kullanilan stiller
        self.recent_questions = []  # Son uretilen sorular
    
    def get_random_style(self, exclude_recent: int = 3) -> str:
        """
        Rastgele stil secimi
        
        Args:
            exclude_recent: Son N stili haric tut
        """
        all_styles = self.prompt_builder.get_style_variations()
        
        # Son kullanilanlari haric tut
        available = [s for s in all_styles if s not in self.used_styles[-exclude_recent:]]
        
        if not available:
            available = all_styles
        
        selected = random.choice(available)
        self.used_styles.append(selected)
        
        return selected
    
    def get_temperature(self, attempt: int = 1, base: float = 0.7) -> float:
        """
        Dinamik temperature hesapla
        
        Args:
            attempt: Deneme sayisi
            base: Baslangic temperature
        
        Returns:
            0.5 - 1.0 arasi temperature
        """
        # Her denemede biraz artir
        temp = base + (attempt - 1) * 0.1
        
        # Rastgele varyasyon ekle
        temp += random.uniform(-0.05, 0.05)
        
        # Sinirlarda tut
        return max(0.5, min(1.0, temp))
    
    def add_prompt_variation(self, base_prompt: str) -> str:
        """
        Prompt'a rastgele varyasyon ekle
        """
        variations = [
            "\n\nNot: Sayilari orijinal tut, hikayeyi degistir.",
            "\n\nNot: Farkli bir senaryo kullan.",
            "\n\nNot: Daha kisa ve oz bir hikaye yaz.",
            "\n\nNot: Daha detayli bir hikaye yaz.",
            "\n\nNot: Gunluk hayattan ornek kullan.",
        ]
        
        return base_prompt + random.choice(variations)
    
    def is_duplicate(self, new_question: str, threshold: float = 0.85) -> bool:
        """
        Tekrar kontrolu
        
        Args:
            new_question: Yeni soru metni
            threshold: Benzerlik esigi
        """
        # Basit kelime bazli benzerlik
        new_words = set(new_question.lower().split())
        
        for recent in self.recent_questions[-10:]:  # Son 10 soru
            recent_words = set(recent.lower().split())
            
            # Jaccard similarity
            intersection = len(new_words & recent_words)
            union = len(new_words | recent_words)
            
            if union > 0 and intersection / union > threshold:
                return True
        
        return False
    
    def add_to_history(self, question: str):
        """Soru gecmisine ekle"""
        self.recent_questions.append(question)
        
        # Maksimum 100 soru tut
        if len(self.recent_questions) > 100:
            self.recent_questions.pop(0)
    
    def shuffle_examples(self, examples: list, keep_first: int = 2) -> list:
        """
        Orneklerin sirasini karistir (ilk N haric)
        
        Args:
            examples: Ornek listesi
            keep_first: Ilk N tanesini yerinde tut
        """
        if len(examples) <= keep_first:
            return examples
        
        fixed = examples[:keep_first]
        shuffled = examples[keep_first:]
        random.shuffle(shuffled)
        
        return fixed + shuffled


class DiverseQuestionGenerator:
    """Cesitlilik destekli soru uretici"""
    
    def __init__(
        self,
        generator: QuestionGenerator,
        diversity: DiversityService
    ):
        self.generator = generator
        self.diversity = diversity
    
    async def generate_diverse_batch(
        self,
        combination: dict,
        count: int = 5,
        max_duplicates: int = 2
    ) -> list[GeneratedQuestion]:
        """
        Cesitlilik garantili toplu uretim
        
        Args:
            combination: Hedef kombinasyon
            count: Uretilecek soru sayisi
            max_duplicates: Maksimum tekrar denemesi
        """
        results = []
        duplicate_count = 0
        
        while len(results) < count:
            # Rastgele stil
            style = self.diversity.get_random_style()
            
            # Temperature
            temp = self.diversity.get_temperature()
            
            try:
                # Uret
                question = await self.generator.generate_question(
                    combination=combination,
                    style_instruction=style
                )
                
                # Tekrar kontrolu
                if self.diversity.is_duplicate(question.soru):
                    duplicate_count += 1
                    if duplicate_count >= max_duplicates:
                        # Farkli kombinasyon dene veya hata ver
                        raise TooManyDuplicatesError()
                    continue
                
                # Basarili
                self.diversity.add_to_history(question.soru)
                results.append(question)
                duplicate_count = 0  # Reset
                
            except GenerationFailedError:
                continue
        
        return results
```

**Cesitlilik Parametreleri:**

| Parametre | Varsayilan | Aciklama |
|-----------|------------|----------|
| `temperature` | 0.7 | LLM randomness (0.5-1.0) |
| `style_count` | 10 | Farkli stil sayisi |
| `exclude_recent` | 3 | Son N stili haric tut |
| `duplicate_threshold` | 0.85 | Tekrar tespit esigi |
| `max_history` | 100 | Gecmis soru sayisi |

**Kabul Kriterleri:**
- [x] ✅ Rastgele stil secimi calisiyor
- [x] ✅ Dinamik temperature calisiyor
- [x] ✅ Tekrar tespiti calisiyor
- [x] ✅ Ornek siralama karistirma calisiyor
- [x] ✅ Toplu uretimde cesitlilik garantisi var

### 18.4 RAG Bagimliliklari

RAG katmani baslamadan once Model katmanindan beklenen ciktilar:

| Dosya | Durum | Aciklama | Bagimlilk |
|-------|-------|----------|-----------|
| `dataset_ocr_li.csv` | MEVCUT | Soru metinleri + metadata | M-06 |
| `configs.json` | BEKLENIYOR | Kombinasyon skorlari | M-11 |

---

#### R-08: LangChain Entegrasyonu

**Amac:** LangChain framework'u ile basit ve hizli RAG pipeline'i olusturma.

**Alt Gorevler:**
1. LangChain bagimliliklerini yukle
2. OpenAI/HuggingFace embeddings entegrasyonu
3. FAISS vectorstore olusturma
4. RetrievalQA chain kurulumu
5. Soru uretim promptu hazirlama

**Beklenen Dosya:** `src/langchain_rag.py`

**Bagimlliklar:**
```bash
pip install langchain langchain-openai langchain-community faiss-cpu
```

**Kabul Kriterleri:**
- [x] ✅ LangChain yuklu ve calisiyor
- [x] ✅ Sorular vectorstore'a yuklendi
- [x] ✅ RetrievalQA chain calisiyor
- [ ] ⏳ Soru uretimi basarili (API key ile test gerekli)

**Not:** LangChain, prototipleme icin kullanilacak. Uretim icin custom RAG tercih edilebilir.

---

#### R-09: Manim Gorsel Uretimi

**Amac:** Uretilen sorular icin Manim kutuphanesi ile matematiksel gorseller/animasyonlar uretme.

**Alt Gorevler:**
1. Manim Community Edition kurulumu
2. EBOB/EKOK gorsellestirme sablonlari
3. Tablo/sematik gorsel uretimi
4. Carpan agaci animasyonu
5. AR icin export fonksiyonu

**Beklenen Dosyalar:**
- `src/manim_visuals/ebob_visualization.py`
- `src/manim_visuals/question_visual.py`
- `src/services/visual_generator.py`

**Bagimlliklar:**
```bash
pip install manim

# macOS
brew install ffmpeg cairo pango

# Ubuntu
sudo apt install ffmpeg libcairo2-dev libpango1.0-dev
```

**Gorsel Tipleri:**

| Gorsel Tipi | Manim Scene | Cikti |
|-------------|-------------|-------|
| tablo | TableScene | PNG |
| sematik | SchemaScene | PNG/MP4 |
| geometrik_sekil | GeometryScene | PNG/MP4 |
| resimli | IllustrationScene | PNG |

**Kabul Kriterleri:**
- [x] ✅ Manim scene siniflari tanimli (EBOBVisualization, CarpanAgaci, vb.)
- [x] ✅ EBOB/EKOK animasyon kodu yazildi
- [x] ✅ Tablo gorseli scene'i mevcut
- [x] ✅ Unity Assets'e export fonksiyonu tanimli
- [ ] ⏳ Render testi (Manim kurulumu gerekli)

---

### 18.5 RAG Cikti Dosyalari

| Dosya | Konum | Aciklama |
|-------|-------|----------|
| `faiss.index` | `lgs-rag/vectorstore/` | FAISS vektor index |
| `metadata.pkl` | `lgs-rag/vectorstore/` | Soru metadata cache |
| `generated_questions/` | `lgs-rag/data/` | Uretilen sorular |
| `generation_logs/` | `lgs-rag/data/` | Uretim loglari |

---

## 19. Test Stratejisi

### 27.1 Test Piramidi

```
                    ┌─────────┐
                    │   E2E   │  5%
                    ├─────────┤
                    │Integr.  │  20%
                    ├─────────┤
                    │  Unit   │  75%
                    └─────────┘
```

### 27.2 Unit Testler

```python
# test_combination_selector.py

def test_threshold_filtering():
    """Threshold alti kombinasyonlar filtrelenmeli"""
    
def test_weighted_sampling_distribution():
    """Yuksek skorlu kombinasyonlar daha sik secilmeli"""
    
def test_filter_application():
    """Filtreler dogru uygulanmali"""

# test_quality_checker.py

def test_format_validation_four_options():
    """4 secenek kontrolu"""
    
def test_correct_answer_in_options():
    """dogru_cevap seceneklerde olmali"""
    
def test_content_relevance():
    """alt_konu uyumu kontrolu"""
```

### 27.3 Integration Testler

```python
# test_pipeline.py

async def test_full_generation_pipeline():
    """Tam uretim akisi"""
    
async def test_retrieval_to_generation():
    """Retrieval sonuclari LLM'e gecmeli"""
    
async def test_retry_on_failure():
    """Basarisiz uretimde retry calismali"""
```

### 27.4 E2E Testler

```python
# test_api.py

async def test_generate_endpoint():
    """API uzerinden soru uretimi"""
    
async def test_batch_generation():
    """Toplu uretim API'si"""
```

### 27.5 Test Fixture'lari

```python
# conftest.py

@pytest.fixture
def sample_configs():
    return {
        "schema_version": "1.0",
        "threshold": 0.75,
        "combinations": [...]
    }

@pytest.fixture
def mock_llm_response():
    return {
        "hikaye": "Test hikayesi",
        "soru": "Test sorusu?",
        ...
    }
```

---

## 20. Deployment

### 28.1 Gelistirme Ortami

```bash
# Kurulum
conda create ......
conda activate ....
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Vector store baslat
python scripts/init_vectorstore.py

# Sunucu calistir
uvicorn src.api.main:app --reload --port 8000
```

### 28.2 Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Vector store init
RUN python scripts/init_vectorstore.py

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=openai
    volumes:
      - ./vectorstore:/app/vectorstore
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 28.3 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ --cov=src
      
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install ruff
      - run: ruff check src/
```

---

## 21. Guvenlik

### 25.1 API Guvenligi

| Onlem | Uygulama |
|-------|----------|
| API Key | Header: X-API-Key |
| Rate Limiting | 100 req/min per key |
| Input Validation | Pydantic models |
| Output Sanitization | JSON encode |

### 25.2 Secret Yonetimi

```bash
# Asla commit etme:
.env
*.pem
*_key.json

# Guvenli saklama:
# - Environment variables
# - Secret manager (AWS Secrets Manager, Vault)
```

### 25.3 LLM Guvenligi

- Prompt injection korumasI: Input sanitization
- Output validation: Schema enforcement
- Cost limiting: Max token limits

---

## 22. Performans ve Optimizasyon

### 22.1 Hedef Metrikler

| Metrik | Hedef | Olcum |
|--------|-------|-------|
| Latency (p50) | <3s | Soru basi |
| Latency (p99) | <10s | Soru basi |
| Throughput | 10 req/s | Batch |
| Memory | <2GB | Runtime |

### 22.2 Optimizasyon Stratejileri

**Embedding Cache:**
```python
@lru_cache(maxsize=1000)
def get_embedding(text_hash: str) -> np.ndarray:
    return model.encode(text)
```

**Async LLM Calls:**
```python
async def generate_batch(combinations: List) -> List:
    tasks = [generate_one(c) for c in combinations]
    return await asyncio.gather(*tasks)
```

**Connection Pooling:**
```python
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100)
)
```

### 22.3 Benchmark Script

```python
# scripts/benchmark.py
async def run_benchmark():
    times = []
    for _ in range(100):
        start = time.time()
        await generate_question()
        times.append(time.time() - start)
    
    print(f"p50: {np.percentile(times, 50):.2f}s")
    print(f"p99: {np.percentile(times, 99):.2f}s")
```

---

## 23. Izleme ve Loglama

### 27.1 Log Formati

```python
# structlog kullanimi
{
    "timestamp": "2025-12-24T10:30:00Z",
    "level": "INFO",
    "event": "question_generated",
    "combination": {"alt_konu": "ebob_ekok", "zorluk": 4},
    "retrieval_count": 5,
    "generation_time_ms": 2500,
    "quality_score": 0.95,
    "request_id": "uuid"
}
```

### 27.2 Metrikler

```python
# Prometheus metrikleri
generation_requests_total = Counter('generation_requests_total')
generation_duration_seconds = Histogram('generation_duration_seconds')
retrieval_results_count = Histogram('retrieval_results_count')
quality_score = Histogram('quality_score')
llm_api_errors_total = Counter('llm_api_errors_total')
```

### 27.3 Alertler

| Alert | Kosul | Aksiyon |
|-------|-------|---------|
| High Error Rate | >5% hata | Sayfa |
| Slow Response | p99 >15s | Uyari |
| LLM API Down | 3 basarisiz | Sayfa |

---

## 24. Maliyet Analizi

### 28.1 LLM API Maliyetleri

| Model | Input | Output | Tahmini/Soru |
|-------|-------|--------|--------------|
| GPT-4 Turbo | $0.01/1K | $0.03/1K | ~$0.05 |
| GPT-3.5 | $0.0005/1K | $0.0015/1K | ~$0.002 |
| Claude 3.5 | $0.003/1K | $0.015/1K | ~$0.03 |
| Gemini 3 | $/K | $/K | ~$ |


### 28.2 Aylik Maliyet Tahmini

| Senaryo | Soru/Gun | Aylik Maliyet |
|---------|----------|---------------|
| Dusuk | 100 | ~$150 |
| Orta | 500 | ~$750 |
| Yuksek | 2000 | ~$3000 |

### 28.3 Maliyet Optimizasyonu

- Daha kucuk model kullanimi (GPT-3.5 test icin)
- Prompt uzunlugu optimizasyonu
- Basarili uretim oranini artirma
- Cache ile tekrar uretim azaltma

---

## 25. Zaman Cizelgesi

### 25.1 Proje Genel Bakis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJE ZAMAN CIZELGESI                               │
└─────────────────────────────────────────────────────────────────────────────┘

MODEL KATMANI (Upstream)
═════════════════════════════════════════════════════════════════════════════
Hafta 1-2: [M-01 → M-06] Veri Hazirlama
           ├── M-01: Veri Analizi
           ├── M-02: Veri Temizleme
           ├── M-03: is_LGS Etiketi
           ├── M-04: Agirlik Uretimi
           ├── M-05: Feature Engineering
           └── M-06: Dataset Ciktisi ──► dataset_ocr_li.csv (RAG'e hazir)

Hafta 3:   [M-07 → M-08] Model Egitimi
           ├── M-07: Pipeline Kurulumu
           └── M-08: Model Egitimi ──► model.pkl

Hafta 4:   [M-09 → M-11] Skor Uretimi
           ├── M-09: Kombinasyon Taramasi
           ├── M-10: Threshold Uygulamasi
           └── M-11: JSON Uretimi ──► configs.json (RAG BASLANGIC NOKTASI)

RAG KATMANI (Bu Proje)
═════════════════════════════════════════════════════════════════════════════
Hafta 4*:  [R-01 → R-XX] RAG Hazirlik (Model ile paralel baslayabilir)
           ├── Proje yapisi olusturma
           ├── Mock configs.json ile gelistirme
           └── Embedding service kurulumu

Hafta 5:   [R-XX → R-XX] Core Gelistirme
           ├── Config loader
           ├── Combination selector
           ├── Vector store kurulumu
           └── Retriever implementasyonu

Hafta 6:   [R-XX → R-XX] LLM Entegrasyonu
           ├── Prompt builder
           ├── LLM client (OpenAI/Anthropic)
           ├── Question generator
           └── Quality checker

Hafta 7:   [R-XX → R-XX] API ve Test
           ├── FastAPI endpoints
           ├── Error handling
           ├── Unit tests
           └── Integration tests

Hafta 8:   [R-XX → R-XX] Finalizasyon
           ├── Performance tuning
           ├── Documentation
           └── Deployment

AR KATMANI (Downstream)
═════════════════════════════════════════════════════════════════════════════
Hafta 8+:  RAG API hazir olduktan sonra baslayabilir
```

*Not: Mock config kullanilarak Model ile paralel gelistirme yapilabilir.

### 25.2 Model Taskleri Milestone'lari

| Hafta | Task | Durum | Cikti |
|-------|------|-------|-------|
| 1 | M-01: Veri Analizi | [ ] | Analiz raporu |
| 1 | M-02: Veri Temizleme | [ ] | Temiz CSV |
| 1 | M-03: is_LGS Etiketi | [ ] | Etiketli CSV |
| 2 | M-04: Agirlik Uretimi | [ ] | Agirlikli CSV |
| 2 | M-05: Feature Engineering | [ ] | Feature CSV |
| 2 | M-06: Dataset Ciktisi | [ ] | `cleaned_dataset.csv`, `dataset_ocr_li.csv` |
| 3 | M-07: Pipeline Kurulumu | [ ] | Pipeline.py |
| 3 | M-08: Model Egitimi | [ ] | `model.pkl` |
| 4 | M-09: Kombinasyon Taramasi | [ ] | `combination_scores.csv` |
| 4 | M-10: Threshold Uygulamasi | [ ] | Filtrelenmis liste |
| 4 | M-11: JSON Uretimi | [ ] | `configs.json` |

### 25.3 RAG Taskleri Milestone'lari

| Hafta | Task ID | Task Adi | Durum | Cikti |
|-------|---------|----------|-------|-------|
| 4 | R-01 | Embedding Pipeline Kurulumu | [x] ✅ | `embedding_service.py` |
| 5 | R-02 | Filtreleme Sistemi | [x] ✅ | `filter_service.py` |
| 5 | R-03 | Benzer Soru Getirme Modulu | [x] ✅ | `retriever.py` |
| 5 | R-04 | RAG Ciktisinin Formatlenmasi | [x] ✅ | `output_formatter.py` |
| 6 | R-05 | Prompt Tasarimi | [x] ✅ | `prompt_builder.py` |
| 6 | R-06 | Soru Uretim Pipeline'i | [x] ✅ | `question_generator.py` |
| 7 | R-07 | Cesitlilik Mekanizmasi | [x] ✅ | `diversity_service.py` |
| 7 | R-08 | LangChain Entegrasyonu | [x] ✅ | `langchain_rag.py` |
| 8 | R-09 | Manim Gorsel Uretimi | [x] ✅ | `visual_generator.py`, `ebob_visualization.py` |
| 8 | - | API Endpoints | [x] ✅ | `api/main.py` |
| 8 | - | Test & Deployment | [ ] ⏳ | Eksik (Bolum 28.7'ye bkz.) |

### 25.5 Task Bagimliliklari Grafiksel

```
MODEL                                    RAG
═══════════════════════════════════════════════════════════════════════════

[M-01] ─► [M-02] ─► [M-03] ─► [M-04] ─► [M-05] ─► [M-06] ─────────────────┐
                                                    │                     │
                                                    │  dataset_ocr_li.csv │
                                                    ▼                     │
                                               [R-01] Embedding           │
                                                    │                     │
[M-07] ─► [M-08] ─► [M-09] ─► [M-10] ─► [M-11] ────┼──────────────────────┤
                                           │       │                      │
                                           │       ▼                      │
                               configs.json│  [R-02] Filtreleme           │
                                           │       │                      │
                                           └──────►│                      │
                                                   ▼                      │
                                              [R-03] Retrieval            │
                                                   │                      │
                                                   ▼                      │
                                              [R-04] Formatlama           │
                                                   │                      │
                                    [R-05] Prompt ─┘                      │
                                           │                              │
                                           ▼                              │
                                      [R-06] Pipeline ◄───────────────────┘
                                           │
                                           ▼
                                      [R-07] Cesitlilik
                                           │
                                           ▼
                                      [API & Deploy]
```

### 25.4 Kritik Bagimliliklar

```
[M-06] ─────────────────────────────────────────────────► [RAG Retrieval]
       dataset_ocr_li.csv
       (Hafta 2 sonunda hazir)

[M-11] ─────────────────────────────────────────────────► [RAG Baslangic]
       configs.json
       (Hafta 4 sonunda hazir)

[RAG API] ──────────────────────────────────────────────► [AR Entegrasyon]
          /api/v1/generate
          (Hafta 7-8)
```

---

## 26. Riskler ve Azaltma Stratejileri

| Risk | Olasilik | Etki | Azaltma |
|------|----------|------|---------|
| LLM API kesintisi | Orta | Yuksek | Fallback provider |
| Dusuk kalite uretim | Orta | Yuksek | Kalite kontrol + retry |
| Model config gecikmesi | Dusuk | Orta | Mock config ile devam |
| Maliyet asimi | Dusuk | Orta | Rate limiting |
| Turkce embedding kalitesi | Dusuk | Orta | Model karsilastirma |

---

## 27. Basari Kriterleri

### 27.1 MVP Kriterleri

- [x] ✅ configs.json'dan kombinasyon okuma (CombinationSelector)
- [x] ✅ En az 3 benzer soru retrieval (QuestionRetriever)
- [x] ✅ LLM ile soru uretimi (QuestionGenerator + LLMClient)
- [x] ✅ 4 secenekli, tek dogru cevapli soru (validation mevcut)
- [x] ✅ JSON formatta cikti (GeneratedQuestion modeli)

### 27.2 Uretim Kriterleri

- [ ] ⏳ %95+ format uygunluk (test gerekli)
- [ ] ⏳ %90+ icerik uygunluk (test gerekli)
- [ ] ⏳ <10 saniye uretim suresi (benchmark gerekli)
- [ ] ⏳ <%5 tekrar orani (test gerekli)
- [ ] ⏳ <%1 matematiksel hata (test gerekli)

---

## 28. Ekler

### 28.1 Gelecek Vizyonu

| Hedef | Aciklama | Oncelik |
|-------|----------|---------|
| **YZ-AR Entegrasyonu** | Yapay zeka modelinin AR uygulamasina entegre edilerek dinamik zorluk ayarlamasi | P1 |
| **Play Store Yayini** | Uygulamanin genis kullanici kitlesine sunulmasi ve kullanici geri bildirimlerinin toplanmasi | P1 |
| **Konu Cesitliligi** | Diger matematik konularina (Geometri, Cebir) genisleme | P2 |
| **Ogretmen Paneli** | Egitimciler icin analitik ve ilerleme takip sistemi | P2 |

### 28.2 YZ-AR Entegrasyon Plani

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      YZ-AR ENTEGRASYON VIZYONU                              │
└─────────────────────────────────────────────────────────────────────────────┘

Faz 1: Temel Entegrasyon
├── RAG API'den soru cekme
├── Unity'de JSON parse
└── ScriptableObject'e donusum

Faz 2: Dinamik Zorluk
├── Ogrenci performans takibi
├── LGS skoruna gore soru secimi
└── Adaptif zorluk ayarlama

Faz 3: Kisisellestirilmis Ogrenme
├── Ogrenci profili olusturma
├── Zayif konulari tespit
└── Hedefli soru onerisi
```

### 28.3 Proje Ekibi

| Rol | Isim | Sorumluluk |
|-----|------|------------|
| Scrum Master, VR/AR Gelistirme | Baris Kaya | AR modulu, Unity gelistirme |
| Yapay Zeka Modelleme | Mehmet Said Huseyinoglu | Model egitimi, soru uretimi |
| Frontend | Mehmet Fatih Akbas | Kullanici arayuzu |
| Backend | Utku Alyuz | API gelistirme |
| Backend | Ozgun Deniz Sevilmis | API gelistirme |

### 28.4 Referanslar

- [rag.md - Model-RAG Sozlesmesi](../04-DOCS/rag.md)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Unity AR Foundation](https://docs.unity3d.com/Packages/com.unity.xr.arfoundation@5.0/manual/index.html)

### 28.5 Sozluk

| Terim | Tanim |
|-------|-------|
| RAG | Retrieval-Augmented Generation |
| Kombinasyon | (alt_konu, zorluk, gorsel_tipi) uclusu |
| LGS Skoru | Kombinasyonun LGS profiline benzerligi (0-1) |
| Retrieval | Vector store'dan benzer soru arama |
| Threshold | Minimum kabul edilebilir skor |
| AR | Artirilmis Gerceklik (Augmented Reality) |
| XR Origin | Unity AR oturum baslangic noktasi |
| Level Kit | Kod yazmadan senaryo tasarimi sistemi |
| ScriptableObject | Unity veri container sinifi |
| EBOB | En Buyuk Ortak Bolen |
| EKOK | En Kucuk Ortak Kat |

### 28.6 Iletisim

- **Proje Yonetimi:** Trello
- **Kod Deposu:** GitHub (ymgk-qa-ai)
- **Dokumantasyon:** 04-DOCS/

---

### 28.7 Implementasyon Durumu ve Eksik Dosyalar

> **Son Guncelleme:** 2025-12-24  
> **Tamamlanma Orani:** ~40% (Core RAG: 100%, Test/Deploy: 0%)

#### 28.7.1 Tamamlanan Dosyalar

```
06-RAG-WITH-LANGCHAIN/
├── src/
│   ├── __init__.py                      ✅
│   ├── config.py                        ✅ Pydantic Settings
│   ├── langchain_rag.py                 ✅ LangChain RAG entegrasyonu
│   │
│   ├── models/
│   │   ├── __init__.py                  ✅
│   │   ├── combination.py               ✅ Combination modeli
│   │   ├── question.py                  ✅ GeneratedQuestion modeli
│   │   ├── config_schema.py             ✅ configs.json semasi
│   │   └── api_models.py                ✅ Request/Response modelleri
│   │
│   ├── services/
│   │   ├── __init__.py                  ✅
│   │   ├── embedding_service.py         ✅ R-01: FAISS + Turkce embedding
│   │   ├── filter_service.py            ✅ R-02: Filtreleme sistemi
│   │   ├── retriever.py                 ✅ R-03: Benzer soru getirme
│   │   ├── output_formatter.py          ✅ R-04: Formatlama
│   │   ├── prompt_builder.py            ✅ R-05: Prompt tasarimi
│   │   ├── question_generator.py        ✅ R-06: Soru uretim pipeline
│   │   ├── diversity_service.py         ✅ R-07: Cesitlilik mekanizmasi
│   │   ├── combination_selector.py      ✅ Kombinasyon secimi
│   │   └── llm_client.py                ✅ OpenAI/Anthropic client
│   │
│   ├── manim_visuals/
│   │   ├── __init__.py                  ✅
│   │   ├── ebob_visualization.py        ✅ R-09: EBOB/EKOK animasyonlari
│   │   └── visual_generator.py          ✅ Gorsel uretim servisi
│   │
│   └── api/
│       ├── __init__.py                  ✅
│       └── main.py                      ✅ FastAPI endpoints
│
├── scripts/
│   ├── init_vectorstore.py              ✅ Index olusturma
│   ├── test_generation.py               ✅ Manuel test scripti
│   └── run_api.py                       ✅ API baslatma
│
├── env.example                          ✅ Ortam degiskenleri sablonu
├── environment.yml                      ✅ Conda ortami
├── requirements.txt                     ✅ Python bagimliliklari
└── README.md                            ✅ Proje dokumantasyonu
```

#### 28.7.2 Eksik Dosyalar - P0 (Kritik)

| Dosya | Konum | Aciklama | Sorumlu |
|-------|-------|----------|---------|
| `configs.json` | `lgs-model/outputs/` | Model katmani ciktisi - RAG bagimliligi | Model Ekibi |

> **Not:** `configs.json` olmadan sistem mock veri ile calisabilir ancak gercek LGS skorlari kullanılamaz.

#### 28.7.3 Eksik Dosyalar - P1 (Test)

| Dosya | Konum | Aciklama |
|-------|-------|----------|
| `conftest.py` | `tests/` | Pytest fixture'lari |
| `test_combination_selector.py` | `tests/unit/` | Kombinasyon secimi unit testleri |
| `test_quality_checker.py` | `tests/unit/` | Kalite kontrol unit testleri |
| `test_retriever.py` | `tests/unit/` | Retriever unit testleri |
| `test_prompt_builder.py` | `tests/unit/` | Prompt builder unit testleri |
| `test_pipeline.py` | `tests/integration/` | Tam pipeline integration testi |
| `test_api.py` | `tests/integration/` | API integration testleri |
| `sample_configs.json` | `tests/fixtures/` | Test icin mock config |
| `sample_questions.csv` | `tests/fixtures/` | Test icin mock sorular |

#### 28.7.4 Eksik Dosyalar - P2 (Deployment)

| Dosya | Konum | Aciklama |
|-------|-------|----------|
| `Dockerfile` | `docker/` | Container imaji |
| `docker-compose.yml` | `docker/` | Multi-container orchestration |
| `requirements-dev.txt` | `./` | Gelistirme bagimliliklari (pytest, ruff, mypy) |
| `.gitignore` | `06-RAG-WITH-LANGCHAIN/` | Git ignore kurallari |
| `quality_checker.py` | `src/services/` | Ayri kalite kontrol servisi (simdilik question_generator icinde) |

#### 28.7.5 Eksik Dosyalar - P3 (CI/CD)

| Dosya | Konum | Aciklama |
|-------|-------|----------|
| `ci.yml` | `.github/workflows/` | GitHub Actions CI pipeline |
| `pyproject.toml` | `./` | Modern Python proje konfigurasyonu |

#### 28.7.6 Olusturulmasi Gereken Test Dosyalari Detayi

```python
# tests/conftest.py - Ornek
import pytest
from src.services import EmbeddingPipeline, FilterService

@pytest.fixture
def sample_configs():
    return {
        "schema_version": "1.0",
        "threshold": 0.75,
        "combinations": [
            {"alt_konu": "ebob_ekok", "zorluk": 4, "gorsel_tipi": "sematik", "lgs_skor": 0.85}
        ]
    }

@pytest.fixture
def mock_llm_response():
    return {
        "hikaye": "Bir fabrikada...",
        "soru": "En az kac kutu gerekir?",
        "secenekler": {"A": "12", "B": "18", "C": "24", "D": "36"},
        "dogru_cevap": "C",
        "cozum": ["Adim 1: ...", "Adim 2: ..."]
    }
```

```python
# tests/unit/test_combination_selector.py - Ornek
def test_threshold_filtering():
    """Threshold alti kombinasyonlar filtrelenmeli"""
    pass
    
def test_weighted_sampling_distribution():
    """Yuksek skorlu kombinasyonlar daha sik secilmeli"""
    pass
```

#### 28.7.7 Sonraki Adimlar

1. **Hemen:** Model ekibinden `configs.json` talep et
2. **Kisa Vadeli:** Test dosyalarini olustur
3. **Orta Vadeli:** Docker deployment hazirla
4. **Uzun Vadeli:** CI/CD pipeline kur

---

**Dokuman Sonu**

*Son Guncelleme: 2025-12-24*
*Versiyon: 2.1*

