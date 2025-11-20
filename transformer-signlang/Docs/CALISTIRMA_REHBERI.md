# 🚀 TRANSFORMER SIGN LANGUAGE - ÇALIŞTIRMA REHBERİ

## ✅ Ön Koşullar (TAMAMLANDI)
- [x] Conda environment "transformers" kuruldu
- [x] Python 3.10+ yüklü
- [x] Tüm dependencies kuruldu (`requirements.txt`)
- [x] Veri setleri mevcut (56k train, 8.8k val, 7.4k test)

---

## 📋 ADIM ADIM ÇALIŞTIRMA

### 🔧 0. Environment Aktivasyonu

```bash
# Terminal'de çalıştır:
conda activate transformers
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang
```

**Kontrol:**
```bash
python --version  # Python 3.10+ olmalı
which python      # transformers env'ındaki python olmalı
```

---

### 📊 1. VERİ HAZIRLAMA AŞAMASI

#### Adım 1.1: Video Seçimi
**Script:** `scripts/01_select_videos.py`

```bash
python scripts/01_select_videos.py
```

**Ne yapar:**
- Train/Val/Test setlerinden 3 kelimeye (acele, acikmak, agac) ait videoları seçer
- ClassId 1, 2, 5'e karşılık gelen videoları filtreler
- CSV dosyaları oluşturur

**Beklenen Süre:** ~1-2 dakika

**Çıktılar:**
```
data/selected_videos_train.csv
data/selected_videos_val.csv
data/selected_videos_test.csv
```

**Doğrulama:**
```bash
# Kaç video seçildi?
wc -l data/selected_videos_*.csv

# İçeriğe bakın:
head -5 data/selected_videos_train.csv
```

**Beklenen Sonuç:**
- Her CSV'de video_path, class_id, path, split kolonları
- Train: 373 video (ClassId 1,2,5)
- Val: 59 video (ClassId 1,2,5)
- Test: 50 video (ClassId 1,2,5)

**⚠️ ÖNEMLI NOT:**
Script otomatik olarak doğru label dosyalarını kullanır:
- Validation: `ground_truth 2.csv` (4,418 satır)
- Test: `ground_truth.csv` (3,742 satır)

---

#### Adım 1.2: Keypoint Extraction
**Script:** `scripts/02_extract_keypoints.py`

```bash
python scripts/02_extract_keypoints.py
```

**Ne yapar:**
- Her videodan MediaPipe ile keypoint'leri çıkarır
- Pose (99D) + Face (33D) + Left Hand (63D) + Right Hand (63D) = 258D
- Her video için `.npy` dosyası oluşturur

**⚠️ ÖNEMLİ:**
- **Bu adım EN UZUN süren adımdır!**
- Video sayısına göre **30-90 dakika** sürebilir
- Progress bar ile ilerlemeyi takip edebilirsiniz

**Beklenen Süre:** ~30-90 dakika (video sayısına göre)

**Çıktılar:**
```
data/keypoints/
├── signer0_sample16.npy
├── signer0_sample25.npy
├── signer0_sample29.npy
└── ... (482 dosya - tüm train/val/test videoları)
```

**Doğrulama:**
```bash
# Kaç keypoint dosyası oluşturuldu?
ls data/keypoints/*.npy | wc -l
# Beklenen: 482

# Bir dosyanın boyutunu kontrol et:
python -c "import numpy as np; data = np.load('data/keypoints/signer0_sample16.npy'); print(f'Shape: {data.shape}')"
# Beklenen: Shape: (frame_count, 258)
```

**İpucu:**
- Script progress bar gösterir
- Her 10 videoda bir otomatik kayıt yapar
- Kesintide kaldığı yerden devam eder

---

#### Adım 1.3: Normalization & Padding
**Script:** `scripts/03_normalize_data.py`

```bash
python scripts/03_normalize_data.py
```

**Ne yapar:**
- Keypoint'leri Z-score normalization ile normalize eder
- Scaler'ı **sadece train data**'da fit eder (data leakage önlenir)
- Tüm sequence'leri aynı uzunluğa getirir (padding/truncate)
- MAX_SEQ_LENGTH: 95th percentile'a göre belirlenir

**Beklenen Süre:** ~2-5 dakika

**Çıktılar:**
```
data/processed/
├── X_train.npy        # (N_train, seq_len, 258)
├── y_train.npy        # (N_train,)
├── train_ids.npy      # (N_train,)
├── X_val.npy          # (N_val, seq_len, 258)
├── y_val.npy          # (N_val,)
├── val_ids.npy        # (N_val,)
├── X_test.npy         # (N_test, seq_len, 258)
├── y_test.npy         # (N_test,)
├── test_ids.npy       # (N_test,)
└── metadata.json

data/scaler.pkl         # StandardScaler (train'de fit)
```

**Doğrulama:**
```bash
# Shape'leri kontrol et:
python -c "
import numpy as np
print('Train:', np.load('data/processed/X_train.npy').shape)
print('Val:  ', np.load('data/processed/X_val.npy').shape)
print('Test: ', np.load('data/processed/X_test.npy').shape)
"

# Metadata'yı incele:
cat data/processed/metadata.json
```

**Beklenen Sonuç:**
```
Train: (N_train, seq_len, 258)
Val:   (N_val, seq_len, 258)
Test:  (N_test, seq_len, 258)

seq_len: ~150-200 frame (95th percentile)
```

---

### 🎓 2. MODEL EĞİTİMİ

#### Adım 2: Training
**Script:** `train.py`

```bash
python train.py
```

**Ne yapar:**
- Transformer modelini eğitir
- Validation accuracy'ye göre best model'i kaydeder
- Early stopping ile gereksiz eğitimi önler
- Training history'yi JSON olarak kaydeder

**Model Özellikleri:**
- **Architecture:** 6-layer Transformer Encoder
- **d_model:** 256
- **Attention heads:** 8
- **Feedforward dim:** 1024
- **Optimizer:** AdamW (lr=1e-4)
- **Scheduler:** Cosine Annealing with Warmup (10 epochs)
- **Loss:** Label Smoothing Cross-Entropy (ε=0.1)
- **Early Stopping:** 10 epoch patience

**Beklenen Süre:**
- **GPU (CUDA):** ~30-60 dakika
- **CPU (M1/M2/M3):** ~90-150 dakika
- **CPU (Intel):** ~120-240 dakika

**⚠️ ÖNEMLİ:**
- Progress bar gösterir (epoch/batch tracking)
- Her epoch sonunda val accuracy/loss yazdırır
- Best model otomatik kaydedilir
- CUDA varsa otomatik GPU kullanır

**Çıktılar:**
```
checkpoints/
├── best_model.pth     # En iyi val accuracy modeli
└── last_model.pth     # Son checkpoint

logs/
└── training_history.json  # Loss/accuracy history
```

**Monitoring:**
```bash
# Training sırasında başka bir terminal'de:
tail -f logs/training.log  # (eğer log dosyası oluşturuluyorsa)

# Training sonrası history'yi incele:
cat logs/training_history.json | python -m json.tool | head -30
```

**Doğrulama:**
```bash
# Checkpoint'leri kontrol et:
ls -lh checkpoints/

# Best model'in epoch/accuracy bilgisi:
python -c "
import torch
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Val Accuracy: {ckpt[\"val_acc\"]:.4f}')
print(f'Val F1: {ckpt[\"val_f1\"]:.4f}')
"
```

**Beklenen Performans (İlk 3 Kelime):**
- **Accuracy:** %70-85
- **F1-Score:** %68-83
- **Loss (final):** 0.3-0.6

#### 🔄 Checkpoint Resume (Kaldığı Yerden Devam)

**NEW!** Eğitim yarıda kesildiyse kaldığı yerden devam edebilirsiniz:

**Kullanım Senaryoları:**

**1. Normal Eğitim (Sıfırdan):**
```bash
python train.py
```

**2. Last Checkpoint'ten Devam:**
```bash
# Eğitim kesildiyse (Ctrl+C, elektrik kesintisi, vb.)
python train.py --resume checkpoints/last_model.pth
```

**3. Best Model'den Devam (Fine-tuning):**
```bash
# En iyi model'den devam et
python train.py --resume-from-best
```

**4. Spesifik Checkpoint'ten Devam:**
```bash
python train.py --resume checkpoints/epoch_50.pth
```

**Resume Özelliği Detayları:**

| Özellik | Açıklama |
|---------|----------|
| ✅ Model Weights | Tam olarak kaldığı yerden |
| ✅ Optimizer State | Momentum ve variance korunur |
| ✅ LR Scheduler | Learning rate doğru pozisyondan |
| ✅ Best Accuracy | En iyi skor takibi devam eder |
| ✅ Training History | Grafikler kopuksuz devam eder |
| ✅ Early Stopping | Patience counter korunur |

**Console Output Örneği:**
```
📂 Loading checkpoint from checkpoints/last_model.pth...
   ✅ Model weights loaded
   ✅ Optimizer state loaded
   ✅ Scheduler state loaded
   📊 Resuming from epoch 26
   📈 Best val accuracy: 0.8542
   📈 Best val F1: 0.8401
   📜 Training history restored (25 epochs)
   ⏳ Early stopping patience counter: 3/15

✅ Successfully loaded checkpoint!
   Training will resume from epoch 26

🔄 RESUMING TRAINING from Epoch 26
```

**Faydaları:**
- 🔴 **Elektrik Kesintisi:** Eğitim kaybı yok
- 🎯 **GPU Timeout:** Uzun eğitimleri bölebilirsin
- ⚡ **Hiperparametre Değişikliği:** İstediğin noktadan farklı LR ile devam
- 💾 **Disk Tasarrufu:** Her epoch'u kaydetmeye gerek yok

**Checkpoint Dosyası İçeriği:**
```python
checkpoint = {
    'epoch': 25,                         # Hangi epoch'ta
    'model_state_dict': ...,            # Model ağırlıkları
    'optimizer_state_dict': ...,        # AdamW momentum/variance
    'scheduler_state_dict': ...,        # LR scheduler position
    'val_acc': 0.8542,                  # En iyi val accuracy
    'val_f1': 0.8401,                   # En iyi val F1
    'history': {...},                   # Training curves
    'patience_counter': 3,              # Early stopping counter
    'config': {...}                     # Tüm hiperparametreler
}
```

**Önemli Notlar:**
- ⚠️ Checkpoint ve yeni kod aynı model architecture'a sahip olmalı
- ⚠️ Resume edilirken config.NUM_EPOCHS yeterince büyük olmalı
- ✅ Hata durumunda otomatik sıfırdan başlar (güvenli)

---

### 📊 3. MODEL DEĞERLENDİRME

#### Adım 3: Evaluation
**Script:** `evaluate.py`

```bash
python evaluate.py
```

**Ne yapar:**
- Test setinde model performansını ölçer
- Comprehensive metrics hesaplar
- Confusion matrix oluşturur
- Per-class performance analizi
- Visualization'lar oluşturur

**Beklenen Süre:** ~2-5 dakika

**Çıktılar:**
```
results/
├── evaluation_report.json              # Tüm metrics
├── confusion_matrix_raw.csv            # Raw confusion matrix
├── confusion_matrix_normalized.csv     # Normalized CM
├── confusion_matrix_raw.png            # Raw CM heatmap
├── confusion_matrix_normalized.png     # Normalized CM heatmap
├── per_class_metrics.csv               # Per-class precision/recall/F1
├── per_class_metrics.png               # Bar chart
└── prediction_confidence.png           # Confidence distribution
```

**Metrics:**
- **Overall:** Accuracy, Precision (macro/weighted), Recall, F1-Score
- **Per-Class:** Her sınıf için ayrı metrics
- **Confusion Matrix:** Raw counts ve normalized
- **Confidence Analysis:** Doğru/yanlış tahminlerin güven dağılımı

**Doğrulama:**
```bash
# Results'ları listele:
ls results/

# Evaluation report'u incele:
cat results/evaluation_report.json | python -m json.tool

# Per-class metrics:
cat results/per_class_metrics.csv
```

**Beklenen Çıktı:**
```json
{
    "overall": {
        "accuracy": 0.75-0.85,
        "precision_macro": 0.73-0.83,
        "recall_macro": 0.72-0.82,
        "f1_macro": 0.72-0.82
    },
    "per_class": {
        "acele": {"precision": 0.xx, "recall": 0.xx, "f1_score": 0.xx},
        "acikmak": {"precision": 0.xx, "recall": 0.xx, "f1_score": 0.xx},
        "agac": {"precision": 0.xx, "recall": 0.xx, "f1_score": 0.xx}
    }
}
```

---

### 🎨 4. ATTENTION VISUALIZATION

#### Adım 4: Attention Görselleştirme
**Script:** `visualize_attention.py`

```bash
# Default: 5 sample
python visualize_attention.py

# Daha fazla sample:
python visualize_attention.py --num_samples 10
```

**Ne yapar:**
- Test setinden random örnekler seçer
- Her layer'ın attention weights'lerini çıkarır
- Multi-head attention'ları görselleştirir
- Attention rollout (cumulative) hesaplar
- Layer/Head bazında istatistikler

**Beklenen Süre:** ~3-10 dakika (sample sayısına göre)

**Çıktılar:**
```
results/attention/
├── sample_0_layer_0_multihead.png     # Multi-head attention (2x4 grid)
├── sample_0_layer_0_avg.png           # Average attention
├── sample_0_layer_1_multihead.png
├── sample_0_layer_1_avg.png
├── ...
├── sample_0_attention_rollout.png     # Cumulative attention
├── sample_1_...                       # İkinci sample
├── ...
├── layer_wise_attention_stats.png     # Layer statistics
└── head_wise_attention_stats.png      # Head statistics
```

**Toplam Dosya:** ~47 dosya (5 samples × 6 layers × 2 + 5 rollout + 2 stats)

**Doğrulama:**
```bash
# Kaç görselleştirme oluşturuldu?
ls results/attention/*.png | wc -l

# Preview (macOS):
open results/attention/sample_0_attention_rollout.png
open results/attention/layer_wise_attention_stats.png
```

**Insight'lar:**
- **Temporal patterns:** Hangi frame'lere odaklanılıyor?
- **Head specialization:** Her head farklı pattern mı?
- **Layer hierarchy:** Alt layer local, üst layer global mı?

---

## ✅ TAMAMLANAN CHECKLIST

Pipeline'ı tamamladıktan sonra:

```bash
# 1. Tüm data dosyaları oluşturuldu mu?
ls data/selected_videos_*.csv
ls data/keypoints/*.npy | wc -l
ls data/processed/*.npy

# 2. Model eğitildi mi?
ls checkpoints/

# 3. Evaluation tamamlandı mı?
ls results/*.png
ls results/*.csv
ls results/*.json

# 4. Attention visualization tamamlandı mı?
ls results/attention/*.png | wc -l
```

---

## 📊 BEKLENEN SONUÇLAR ÖZET

### Performans Metrikleri (İlk 3 Kelime)
```
Accuracy:          75-85%
Precision (macro): 73-83%
Recall (macro):    72-82%
F1-Score (macro):  72-82%
```

### Dosya Sayıları
```
Keypoints:         ~500-1000 .npy dosyası
Processed data:    9 .npy + 1 .pkl + 1 .json
Checkpoints:       2 .pth dosyası
Evaluation:        8 dosya (4 CSV + 4 PNG)
Attention:         ~47 PNG dosyası
```

### Toplam Süre
```
Veri Hazırlama:    ~35-100 dakika
Model Eğitimi:     ~30-240 dakika (GPU/CPU)
Evaluation:        ~2-5 dakika
Visualization:     ~3-10 dakika
─────────────────────────────────
TOPLAM:            ~70-355 dakika (1-6 saat)
```

---

## 🔧 SORUN GİDERME

### Hata: "No module named 'torch'"
```bash
# Environment'ı kontrol edin:
conda activate transformers
pip list | grep torch
```

### Hata: "FileNotFoundError: Data/Train Data/train"
```bash
# config.py'daki veri yollarını kontrol edin
# BASE_DATA_DIR doğru mu?
```

### Hata: "CUDA out of memory"
```bash
# config.py'da BATCH_SIZE'ı küçültün:
BATCH_SIZE = 16  # veya 8
```

### Training çok yavaş
```bash
# GPU kullanıldığını kontrol edin:
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# CPU kullanıyorsanız, model boyutunu küçültün:
# config.py'da NUM_ENCODER_LAYERS = 4 yapın
```

---

## 🎯 SONRAKI ADIMLAR

Pipeline başarıyla tamamlandıktan sonra:

1. **Results İnceleme:**
   - `results/evaluation_report.json` → Overall performance
   - `results/confusion_matrix_normalized.png` → Hangi sınıflar karıştırılıyor?
   - `results/per_class_metrics.png` → Hangi sınıf daha zor?
   - `results/attention/` → Model neye bakıyor?

2. **İyileştirmeler:**
   - Hiperparametre tuning (learning rate, batch size)
   - Daha fazla kelime ekleme (config.py → TARGET_CLASS_IDS)
   - Data augmentation (config.py → USE_AUGMENTATION = True)
   - Model büyütme/küçültme

3. **Deneyler:**
   - Farklı pooling stratejileri (GAP vs CLS vs Last)
   - Farklı model boyutları (tiny vs small vs base vs large)
   - Farklı optimizer'lar (Adam vs AdamW)

---

## 📞 YARDIM

- **README.md:** Teknik detaylar ve architecture açıklaması
- **RUN_PIPELINE.md:** Detaylı troubleshooting
- **ilerleme.md:** Her adımın ne yaptığının notları

---

**🎉 HAZIRSıNıZ! Pipeline'ı çalıştırmaya başlayabilirsiniz!**

