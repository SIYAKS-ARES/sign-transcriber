# 🚀 226 Kelime (TÜM AUTSL) - Çalıştırma Rehberi

**Tarih:** 7 Ekim 2025  
**Sistem:** 226 Kelime İşaret Dili Tanıma  
**Strateji:** Direkt 226 Kelime (Agresif)

---

## 📊 SİSTEM ÖZETİ

### Veri Seti (AUTSL - Tam)

```
┌──────────────────────────────────────────────────────────┐
│  AUTSL DATASET (TÜM VERİ)                                │
├──────────────────────────────────────────────────────────┤
│  Toplam Sınıf:        226 kelime (ClassId: 0-225)       │
│  Train Videos:        28,142 (31 signer)                 │
│  Validation Videos:   4,418 (6 signer)                   │
│  Test Videos:         3,742 (6 signer)                   │
│  TOPLAM:              36,302 video                       │
│                                                           │
│  10 Kelime ile Kıyasla: 22.6x DAHA BÜYÜK! 🚀           │
└──────────────────────────────────────────────────────────┘
```

### Hazırlanan Sistem

```
✅ Config güncellendi (NUM_CLASSES=226)
✅ CLASS_NAMES otomatik yükleniyor (226 kelime)
✅ Tüm script'ler uyumlu (değişiklik gerekmedi!)
✅ Model hiperparametreleri optimize edildi
✅ Disk alanı yeterli (193 GB boş)
✅ 10-kelime modeli yedeklendi
```

---

## ⏱️ BEKLENEN SÜRELER

| Adım | Süre | Açıklama |
|------|------|----------|
| **1. Video Seçimi** | 2 dakika | CSV dosyaları oluşturulur |
| **2. Keypoint Extraction** | **30-50 SAAT** ⏰ | En uzun adım! |
| **3. Normalization** | 2-3 saat | Veri ön işleme |
| **4. Setup Validation** | 1 dakika | Doğrulama |
| **5. Model Training** | **50-80 SAAT** ⏰ | Eğitim |
| **6. Evaluation** | 15-30 dakika | Test ve metrikler |
| **7. (Opsiyonel) Attention Viz** | 30-60 dakika | Görselleştirme |
| **TOPLAM** | **~85-135 saat** | **3.5-5.5 gün** |

⚠️ **DİKKAT:** Keypoint extraction ve training çok uzun! Bilgisayarı başka işler için kullanabilirsiniz ama kapatmayın.

---

## 💾 DISK ALANI GEREKSİNİMLERİ

| Dosya Tipi | Boyut | Açıklama |
|------------|-------|----------|
| Keypoints (.npy) | ~1.8 GB | 36,302 video × 50 KB |
| Processed Data | ~11 GB | Normalized + padded |
| Model Checkpoints | ~350 MB | best + last model |
| Results | ~50 MB | Evaluation + plots |
| **TOPLAM** | **~13-15 GB** | - |

**Mevcut Boş Alan:** 193 GB ✅ **Yeterli!**

---

## 🎯 BEKLENEN PERFORMANS

### Hedef Metrikler

```
Test Accuracy:     68-75%  (Hedef: >70%)
F1-Score (Macro):  65-73%
Top-5 Accuracy:    85-90%

10 Kelime:         91.57% ✅
226 Kelime:        ~72%   (tahmin - normal düşüş)
```

**Neden Düşük?**
- 226 sınıf 10'a göre 22.6x daha zor
- %70-75 accuracy çok iyi sayılır!
- State-of-the-art modeller ~%80-85 civarında

---

## 📝 ÇALIŞTIRMA ADIMLARI

### ÖNCESİ: Ortam Hazırlığı

```bash
# Terminal'i aç
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang

# Conda ortamını aktive et
conda activate transformers

# Python ve paketleri kontrol et
python --version  # 3.10+
python -c "import torch; print(torch.__version__)"  # 2.0+
python -c "import mediapipe; print(mediapipe.__version__)"  # 0.10+
```

---

### ADIM 1: Video Seçimi (2 dakika)

**Komut:**
```bash
python scripts/01_select_videos.py
```

**Ne Yapar:**
- `Data/` dizininden 226 sınıfa ait tüm videoları seçer
- Train/Val/Test setlerine ayırır
- CSV dosyaları oluşturur

**Beklenen Çıktı:**
```
📹 VİDEO SEÇİMİ - TRAIN/VAL/TEST SETLER
================================================================================

🎯 Hedef Sınıflar:
   ClassId 0: abla
   ClassId 1: acele
   ...
   ClassId 225: zor

================================================================================
✅ TRAIN SET İŞLENİYOR
================================================================================
   ✅ Toplam 28142 video etiketi yüklendi
   ✅ Filtrelenmiş video sayısı: 28142
   ✅ Bulunan: 28142 video

================================================================================
✅ VAL SET İŞLENİYOR
================================================================================
   ✅ Toplam 4418 video etiketi yüklendi
   ✅ Filtrelenmiş video sayısı: 4418
   ✅ Bulunan: 4418 video

================================================================================
✅ TEST SET İŞLENİYOR
================================================================================
   ✅ Toplam 3742 video etiketi yüklendi
   ✅ Filtrelenmiş video sayısı: 3742
   ✅ Bulunan: 3742 video

💾 SONUÇLAR KAYDEDİLİYOR
   ✅ Train: data/selected_videos_train.csv (28142 video)
   ✅ Val:   data/selected_videos_val.csv (4418 video)
   ✅ Test:  data/selected_videos_test.csv (3742 video)

✅ TOPLAM: 36302 video seçildi!
```

**Oluşan Dosyalar:**
```
data/
├── selected_videos_train.csv  (28,142 satır)
├── selected_videos_val.csv    (4,418 satır)
└── selected_videos_test.csv   (3,742 satır)
```

**Sorun Giderme:**
- ❌ `FileNotFoundError`: `Data/` dizini yolunu kontrol et
- ❌ `KeyError: 'class_id'`: CSV formatı bozuk olabilir

---

### ADIM 2: Keypoint Extraction ⏰ (30-50 SAAT!)

**Komut:**
```bash
python scripts/02_extract_keypoints.py
```

**Ne Yapar:**
- Her video için MediaPipe ile keypoint'leri çıkarır
- 36,302 video × ~50 KB = ~1.8 GB veri üretir
- Her frame: Pose (99D) + Face (33D) + Hands (126D) = 258D

**⏰ SÜRE:** 30-50 SAAT! (Video başına ~3-5 saniye)

**İlerleme Takibi:**
```
🎬 MEDİAPİPE KEYPOINT ÇIKARIMI
================================================================================

📂 CSV dosyaları yükleniyor...
   ✅ Train: 28142 video
   ✅ Val:   4418 video
   ✅ Test:  3742 video

📊 Toplam: 36302 video

🎯 Keypoint çıkarımı başlıyor...
   📁 Çıktı dizini: data/keypoints/

Processing videos: 100%|██████████| 36302/36302 [30:00:00<00:00, 3.00s/video]

✅ TAMAMLANDI!
   ✅ Başarılı: 36302/36302 video
   ✅ Çıktı: data/keypoints/ (~1.8 GB)
```

**İpuçları:**
- ✅ Progress bar ile ilerlemeyi takip edebilirsin
- ✅ Bilgisayarı başka işler için kullanabilirsin (arka planda çalışır)
- ✅ Kesintide kaldığı yerden devam eder (skip existing files)
- ⚠️ Bilgisayarı **KAPATMA!** (30-50 saat çalışacak)

**Oluşan Dosyalar:**
```
data/keypoints/
├── signer0_sample0_color.npy
├── signer0_sample1_color.npy
├── ...
└── signer42_sample225_color.npy
(Toplam 36,302 dosya, ~1.8 GB)
```

**Sorun Giderme:**
- ❌ `ModuleNotFoundError: mediapipe`: `conda install -c conda-forge mediapipe`
- ⚠️ Bazı videolarda keypoint çıkaramıyor: Normal, skip edilir
- 🐌 Çok yavaş (>10s/video): CPU yavaş olabilir, normaldir

---

### ADIM 3: Normalization (2-3 saat)

**Komut:**
```bash
python scripts/03_normalize_data.py
```

**Ne Yapar:**
- Keypoint'leri yükler
- Z-score normalizasyonu uygular (StandardScaler)
- Sequence padding/truncation (max_length=200)
- Train/Val/Test setleri oluşturur

**Beklenen Çıktı:**
```
📊 VERİ NORMALİZASYONU
================================================================================

📦 TRAIN KEYPOINT'LER YÜKLENİYOR
   ✅ 28142 video bulundu
Loading train: 100%|██████████| 28142/28142 [15:00<00:00, 31.27file/s]
   ✅ Yükleme tamamlandı: 28142 dosya

📦 VAL KEYPOINT'LER YÜKLENİYOR
   ✅ 4418 video bulundu
Loading val: 100%|██████████| 4418/4418 [02:20<00:00, 31.46file/s]
   ✅ Yükleme tamamlandı: 4418 dosya

📦 TEST KEYPOINT'LER YÜKLENİYOR
   ✅ 3742 video bulundu
Loading test: 100%|██████████| 3742/3742 [02:00<00:00, 31.18file/s]
   ✅ Yükleme tamamlandı: 3742 dosya

🔧 SCALER FIT EDİLİYOR (Train verisi)
   ✅ Scaler fit edildi ve kaydedildi: data/scaler.pkl

📊 SEQUENCE PADDING/TRUNCATION (max_length=200)
   ✅ Train padding tamamlandı
   ✅ Val padding tamamlandı
   ✅ Test padding tamamlandı

💾 KAYDETME
   ✅ data/processed/X_train.npy (28142, 200, 258) - 9.8 GB
   ✅ data/processed/y_train.npy (28142,)
   ✅ data/processed/X_val.npy (4418, 200, 258) - 1.5 GB
   ✅ data/processed/y_val.npy (4418,)
   ✅ data/processed/X_test.npy (3742, 200, 258) - 1.3 GB
   ✅ data/processed/y_test.npy (3742,)

✅ VERİ HAZIRLAMA TAMAMLANDI!
```

**Oluşan Dosyalar:**
```
data/
├── scaler.pkl             (~1 KB)
└── processed/
    ├── X_train.npy        (~9.8 GB)
    ├── y_train.npy        (~110 KB)
    ├── X_val.npy          (~1.5 GB)
    ├── y_val.npy          (~17 KB)
    ├── X_test.npy         (~1.3 GB)
    └── y_test.npy         (~15 KB)
```

**Sorun Giderme:**
- ❌ `MemoryError`: RAM yetersiz → Batch processing kullan
- ⚠️ Bazı dosyalar eksik: Normal, keypoint extraction sırasında atlananlar

---

### ADIM 4: Setup Validation (1 dakika)

**Komut:**
```bash
python validate_setup.py
```

**Ne Yapar:**
- Tüm dosyaların varlığını kontrol eder
- Config parametrelerini doğrular
- Model oluşturulabilirliğini test eder

**Beklenen Çıktı:**
```
================================================================================
🔍 SİSTEM DOĞRULAMA - 226 Kelime
================================================================================

1/7 Python versiyonu...              ✅ OK (3.10.x)
2/7 PyTorch kurulumu...              ✅ OK (2.x.x)
3/7 Veri dosyaları...                ✅ OK (6/6 dosya mevcut)
4/7 Config parametreleri...          ✅ OK (NUM_CLASSES=226)
5/7 Class mapping...                 ✅ OK (226 sınıf)
6/7 Model oluşturma...               ✅ OK (~17M params)
7/7 Device...                        ✅ OK (mps)

================================================================================
✅ TÜM KONTROLLER BAŞARILI!
================================================================================
SİSTEM EĞİTİME HAZIR! 🚀
```

**Sorun Giderme:**
- ❌ Herhangi bir check FAILED: İlgili adımı tekrar et
- ❌ `ModuleNotFoundError`: Paketi yükle (`conda install ...`)

---

### ADIM 5: Model Training ⏰ (50-80 SAAT!)

**Komut:**
```bash
python train.py
```

**Ne Yapar:**
- Transformer modelini eğitir
- Best model'i kaydeder (val_accuracy en yüksek)
- Training log'ları tutar

**⏰ SÜRE:** 50-80 SAAT! (Epoch başına ~30-50 dakika × 100 epoch)

**İlerleme Takibi:**
```
================================================================================
🚀 MODEL EĞİTİMİ BAŞLIYOR
================================================================================

📊 Model Bilgileri:
   Architecture:  Transformer Encoder
   Params:        17,423,618 (~17M)
   Device:        mps (Apple Silicon GPU)
   
📊 Veri Bilgileri:
   Train:         28142 samples
   Val:           4418 samples
   Batch size:    16
   
📊 Eğitim Ayarları:
   Max epochs:    100
   Learning rate: 0.0001
   Optimizer:     AdamW
   Scheduler:     CosineAnnealingLR (warmup: 15 epochs)

================================================================================
Epoch 1/100
================================================================================
Train: 100%|██████████| 1759/1759 [45:23<00:00, 1.64s/batch]
Val:   100%|██████████| 277/277 [02:15<00:00, 2.05it/s]

Epoch 1/100 - Train Loss: 4.8523 - Train Acc: 12.34% - Val Loss: 4.2156 - Val Acc: 18.92%
⏱️  Epoch time: 47:38

...

================================================================================
Epoch 33/100
================================================================================
Train: 100%|██████████| 1759/1759 [44:12<00:00, 1.51s/batch]
Val:   100%|██████████| 277/277 [02:10<00:00, 2.12it/s]

Epoch 33/100 - Train Loss: 0.8234 - Train Acc: 78.45% - Val Loss: 1.1234 - Val Acc: 72.18% 🌟
⏱️  Epoch time: 46:22
💾 New best model saved! (val_acc: 72.18%)

...

⏹️  Early stopping triggered! (patience: 20)
✅ Best val accuracy: 72.18% (epoch 33)

💾 Final checkpoint kaydedildi:
   ✅ checkpoints/best_model.pth (val_acc: 72.18%)
   ✅ checkpoints/last_model.pth (epoch: 53)
   ✅ logs/training_history.json

================================================================================
✅ EĞİTİM TAMAMLANDI!
================================================================================
Total time: ~48 hours
Best val accuracy: 72.18%
```

**İpuçları:**
- ✅ Tensorboard ile izleyebilirsin: `tensorboard --logdir logs/`
- ✅ Training log: `logs/training_history.json`
- ✅ Checkpoint'ler otomatik kaydedilir
- ⚠️ Early stopping devrede (20 epoch patience)
- ⚠️ Bilgisayarı **KAPATMA!** (50-80 saat çalışacak)

**Oluşan Dosyalar:**
```
checkpoints/
├── best_model.pth         (~350 MB)
└── last_model.pth         (~350 MB)

logs/
└── training_history.json  (~50 KB)
```

**Sorun Giderme:**
- 🐌 Çok yavaş: Normal, 226 sınıf zor
- ⚠️ Val accuracy düşük (<60%): 30-40 epoch'a kadar bekle
- ❌ `CUDA out of memory`: BATCH_SIZE'ı küçült (16 → 8)

---

### ADIM 6: Evaluation (15-30 dakika)

**Komut:**
```bash
python evaluate.py
```

**Ne Yapar:**
- Test seti üzerinde modeli değerlendirir
- Accuracy, Precision, Recall, F1-Score hesaplar
- Confusion matrix ve per-class metrikleri oluşturur

**Beklenen Çıktı:**
```
================================================================================
📊 MODEL DEĞERLENDİRME - 226 Kelime
================================================================================

Loading test data...      ✅ (3742 samples)
Loading best model...     ✅ (val_acc: 72.18%)

Testing: 100%|██████████| 235/235 [15:23<00:00, 3.93s/batch]

================================================================================
✅ TEST SONUÇLARI
================================================================================

📊 Genel Metrikler:
   Test Accuracy:        71.84%
   Precision (Macro):    70.52%
   Recall (Macro):       69.87%
   F1-Score (Macro):     70.19%
   Top-5 Accuracy:       89.23%

📊 En Başarılı 10 Sınıf:
   1. anne      (F1: 92.3%)
   2. baba      (F1: 91.7%)
   3. evet      (F1: 89.5%)
   ...

📊 En Zor 10 Sınıf:
   1. akilsiz   (F1: 45.2%)
   2. yildiz    (F1: 48.7%)
   3. bal       (F1: 51.3%)
   ...

💾 Sonuçlar kaydedildi:
   ✅ results/evaluation_report.json
   ✅ results/confusion_matrix_raw.png
   ✅ results/confusion_matrix_normalized.png
   ✅ results/per_class_metrics.csv
   ✅ results/test_predictions.csv

================================================================================
✅ DEĞERLENDİRME TAMAMLANDI!
================================================================================
```

**Oluşan Dosyalar:**
```
results/
├── evaluation_report.json          (~20 KB)
├── confusion_matrix_raw.png        (~2 MB - 226x226!)
├── confusion_matrix_normalized.png (~2 MB)
├── per_class_metrics.csv           (~30 KB - 226 satır)
├── per_class_f1_score.png
├── per_class_precision.png
├── per_class_recall.png
├── prediction_confidence.png
└── test_predictions.csv            (~500 KB - 3742 satır)
```

**Başarı Değerlendirmesi:**

| Sonuç | Test Acc | Yorum | Aksiyon |
|-------|----------|-------|---------|
| **Mükemmel** 🎉 | >75% | Hedefin üstünde! | Deployment'a hazır! |
| **İyi** ✅ | 70-75% | Beklenen aralıkta | Deployment OK, opsiyonel iyileştirme |
| **Kabul Edilebilir** 🟡 | 65-70% | Minimum üstünde | İyileştirme önerilir |
| **Zayıf** ❌ | <65% | Beklenenin altında | Model revizyonu gerekli |

---

### ADIM 7: (Opsiyonel) Attention Visualization (30-60 dakika)

**Komut:**
```bash
# 10 örnek için (önerilen)
python visualize_attention.py --num_samples 10

# Veya interaktif mod
python visualize_attention.py --interactive
```

**Ne Yapar:**
- Transformer attention weight'lerini görselleştirir
- Hangi frame'lere odaklandığını gösterir

**Süre:** 30-60 dakika (226 sınıf için yavaş)

**Oluşan Dosyalar:**
```
results/
├── attention_heatmap_sample_0.png
├── attention_heatmap_sample_1.png
├── ...
└── attention_statistics.json
```

---

## 📊 SONUÇLARI YORUMLAMA

### 1. Evaluation Report (JSON)

```json
{
  "overall": {
    "test_accuracy": 71.84,
    "f1_macro": 70.19,
    "top5_accuracy": 89.23
  },
  "per_class": {
    "0": {"precision": 0.92, "recall": 0.89, "f1": 0.91},
    ...
  }
}
```

**Kontrol Listesi:**
- ✅ Test Accuracy >70%? → **Başarılı!**
- ✅ Top-5 Accuracy >85%? → **İyi!**
- ⚠️ Bazı sınıflar F1 <50%? → Normal, zor sınıflar

### 2. Confusion Matrix

**226x226 matrix çok büyük!** Şu noktalara bak:
- Diagonal (köşegen) parlak mı? → Doğru tahminler
- Hangi sınıflar karışıyor? → Benzer işaretler

### 3. Per-Class Metrics

**En zor 10 sınıfı analiz et:**
- Veri az mı? (train'de <100 video)
- Benzer sınıflarla karışıyor mu?
- İyileştirme için data augmentation dene

---

## ⚠️ SORUN GİDERME

### Keypoint Extraction Sorunları

**Problem:** Bazı videolarda keypoint çıkaramıyor  
**Çözüm:** Normal, MediaPipe bazı videolarda başarısız olabilir. Skip edilir.

**Problem:** Çok yavaş (>10s/video)  
**Çözüm:** CPU yavaş olabilir. 30-50 saat normaldir.

### Training Sorunları

**Problem:** Val accuracy düşük (<60%)  
**Çözüm:** 30-40 epoch'a kadar bekle. 226 sınıf zordur!

**Problem:** Overfitting (train acc >> val acc)  
**Çözüm:** DROPOUT artır (0.2 → 0.3), LABEL_SMOOTHING artır (0.15 → 0.2)

**Problem:** Memory error  
**Çözüm:** BATCH_SIZE küçült (16 → 8 → 4)

### Evaluation Sorunları

**Problem:** Test accuracy çok düşük (<60%)  
**Çözüm:**
1. Training log'ları kontrol et (overfitting var mı?)
2. Model hiperparametrelerini gözden geçir
3. Data augmentation ekle
4. Daha uzun eğit (100 → 150 epoch)

---

## 📁 DOSYA YAPISI

```
transformer-signlang/
├── config.py                          # ✅ Güncellendi (NUM_CLASSES=226)
├── utils/
│   └── load_class_names.py           # ✅ Yeni eklendi
├── scripts/
│   ├── 01_select_videos.py           # ✅ Uyumlu
│   ├── 02_extract_keypoints.py       # ✅ Uyumlu
│   └── 03_normalize_data.py          # ✅ Uyumlu
├── train.py                          # ✅ Uyumlu
├── evaluate.py                       # ✅ Uyumlu
├── visualize_attention.py            # ✅ Uyumlu
├── validate_setup.py                 # ✅ Uyumlu
├── data/
│   ├── selected_videos_train.csv     # ADIM 1
│   ├── selected_videos_val.csv       # ADIM 1
│   ├── selected_videos_test.csv      # ADIM 1
│   ├── keypoints/                    # ADIM 2 (~1.8 GB)
│   ├── scaler.pkl                    # ADIM 3
│   └── processed/                    # ADIM 3 (~11 GB)
├── checkpoints/
│   ├── best_model.pth                # ADIM 5 (~350 MB)
│   └── last_model.pth                # ADIM 5 (~350 MB)
├── logs/
│   └── training_history.json         # ADIM 5
├── results/                          # ADIM 6 (~5 MB)
│   ├── evaluation_report.json
│   ├── confusion_matrix_*.png
│   ├── per_class_metrics.csv
│   └── test_predictions.csv
├── backups/
│   └── 10-kelime-final/              # 10-kelime yedek (128 MB)
├── ilerleme-226-kelime.md            # İlerleme takibi
├── 226-KELIME-IS-PLANI.md            # İş planı
└── 226-KELIME-CALISTIRMA-REHBERI.md  # Bu dosya!
```

---

## 🎯 BAŞARI DEĞERLENDİRME

### Mükemmel Sonuç 🎉

```
✅ Test Accuracy: >75%
✅ F1-Score (Macro): >73%
✅ Top-5 Accuracy: >92%
✅ En az %90 sınıfın F1 > 65%

→ DEPLOYMENT'A HAZIR!
→ Kutlama zamanı! 🎉
```

### İyi Sonuç ✅

```
✅ Test Accuracy: 70-75%
✅ F1-Score (Macro): 68-73%
✅ Top-5 Accuracy: 87-92%
✅ En az %85 sınıfın F1 > 60%

→ DEPLOYMENT OK
→ Opsiyonel iyileştirme yapılabilir
```

### Kabul Edilebilir Sonuç 🟡

```
🟡 Test Accuracy: 65-70%
🟡 F1-Score (Macro): 62-68%
🟡 Top-5 Accuracy: 82-87%

→ İYİLEŞTİRME ÖNERİLİR
→ Aksiyonlar:
   - Data augmentation ekle
   - Daha uzun eğit (150 epoch)
   - Ensemble model dene
```

### Zayıf Sonuç ❌

```
❌ Test Accuracy: <65%
❌ F1-Score (Macro): <62%

→ MODEL REVİZYONU GEREKLİ
→ Aksiyonlar:
   - Model büyütme (D_MODEL=512)
   - Transfer learning (pretrained)
   - Hyperparameter tuning
```

---

## 🚀 DEPLOYMENT SONRASİ

### Model Kaydedildi ✅

```
checkpoints/best_model.pth (~350 MB)
```

**Kullanım:**
```python
from config import TransformerConfig
from models.transformer_classifier import TransformerSignLanguageClassifier
import torch

# Config
config = TransformerConfig()

# Model yükle
model = TransformerSignLanguageClassifier(...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
# ... (infer_videos.py kullan)
```

**Sonraki Adımlar:**
1. ✅ Model export (ONNX, TorchScript)
2. ✅ Inference pipeline optimize et
3. ✅ Real-time test (webcam)
4. ✅ API deployment (FastAPI, Flask)
5. ✅ Mobile deployment (TFLite, CoreML)

---

## 📞 DESTEK

**Sorun mu yaşıyorsun?**

1. İlerleme dosyasını kontrol et: `ilerleme-226-kelime.md`
2. İş planını oku: `226-KELIME-IS-PLANI.md`
3. Log dosyalarını incele: `logs/training_history.json`

**Başarılar! 🚀**

---

**Son Güncelleme:** 7 Ekim 2025  
**Versiyon:** 1.0 (226 Kelime)

