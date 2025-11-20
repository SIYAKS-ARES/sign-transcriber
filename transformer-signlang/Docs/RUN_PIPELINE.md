# 🚀 Pipeline Çalıştırma Rehberi

Bu doküman, Transformer Sign Language projesinin tüm pipeline'ını baştan sona çalıştırmak için gerekli adımları içerir.

## 📋 Gereksinimler

### 1. Conda Environment Kurulumu

```bash
# Miniconda environment'ını aktif et
conda activate transformers

# Gerekli paketleri yükle
cd transformer-signlang
pip install -r requirements.txt
```

### 2. Veri Kontrolü

Aşağıdaki dizinlerin mevcut olduğundan emin olun:

```
Data/
├── Train Data/train/          # Eğitim videoları
├── Validation Data/val/       # Validation videoları
├── Test Data & Valid, Labels/test/  # Test videoları
└── Class ID/SignList_ClassId_TR_EN.csv  # Sınıf etiketleri
```

---

## 🔄 Pipeline Adımları

### Adım 1: Video Seçimi

**Amaç:** İlk 3 kelime (acele, acikmak, agac) için videoları seç.

```bash
python scripts/01_select_videos.py
```

**Beklenen Çıktı:**

```
data/selected_videos_train.csv  (373 videos)
data/selected_videos_val.csv    (59 videos)
data/selected_videos_test.csv   (50 videos)
```

**Kontrol:**

```bash
wc -l data/selected_videos_*.csv
# Beklenen:
#   51 data/selected_videos_test.csv
#  374 data/selected_videos_train.csv
#   60 data/selected_videos_val.csv
```

**📌 Not:** Script otomatik olarak doğru label dosyalarını kullanır:

- Validation → `ground_truth 2.csv`
- Test → `ground_truth.csv`

---

### Adım 2: Keypoint Extraction

**Amaç:** MediaPipe ile videolardan 258 boyutlu keypoint'leri çıkar.

```bash
python scripts/02_extract_keypoints.py
```

**Beklenen Çıktı:**

```
data/keypoints/
├── signer0_sample16.npy
├── signer0_sample25.npy
└── ... (482 dosya toplam)
```

**Not:** Script otomatik olarak train/val/test CSV'lerini birleştirir ve tüm 482 videoyu işler.

**İlerleme Takibi:**

- Script progress bar gösterir
- Her 10 videoda bir kaydetme yapar
- Hata durumunda kaydedilen yerden devam eder

**Kontrol:**

```bash
ls data/keypoints/*.npy | wc -l
# Beklenen: 482 (373 train + 59 val + 50 test)
```

---

### Adım 3: Data Normalization & Padding

**Amaç:** Keypoint'leri normalize et, padding uygula, scaler kaydet.

```bash
python scripts/03_normalize_data.py
```

**Beklenen Çıktı:**

```
data/processed/
├── X_train.npy
├── y_train.npy
├── train_ids.npy
├── X_val.npy
├── y_val.npy
├── val_ids.npy
├── X_test.npy
├── y_test.npy
├── test_ids.npy
└── metadata.json

data/scaler.pkl
```

**Kontrol:**

```bash
python -c "import numpy as np; print('Train:', np.load('data/processed/X_train.npy').shape)"
python -c "import numpy as np; print('Val:', np.load('data/processed/X_val.npy').shape)"
python -c "import numpy as np; print('Test:', np.load('data/processed/X_test.npy').shape)"
```

---

### Adım 4: Model Training

**Amaç:** Transformer modelini eğit.

```bash
python train.py
```

**Beklenen Çıktı:**

```
checkpoints/
├── best_model.pth
└── last_model.pth

logs/
└── training_history.json
```

**Training Süresi:**

- CPU: ~2-4 saat (100 epoch için)
- GPU: ~30-60 dakika (100 epoch için)

**Early Stopping:**

- Patience: 15 epoch
- Validation accuracy gelişmezse erken durur

**Kontrol:**

```bash
ls -lh checkpoints/
cat logs/training_history.json | python -m json.tool | head -20
```

#### 🔄 Checkpoint Resume (Kaldığı Yerden Devam Etme)

**NEW!** Eğitim yarıda kesildiyse kaldığı yerden devam edebilirsiniz:

**Senaryo 1: Last Checkpoint'ten Devam**

```bash
# Eğitimi başlat
python train.py

# Ctrl+C ile durdur (örn: epoch 25'te)

# Kaldığı yerden devam et
python train.py --resume checkpoints/last_model.pth
```

**Senaryo 2: Best Model'den Devam (Fine-tuning)**

```bash
# İlk eğitim tamamlandı

# En iyi model'den devam et
python train.py --resume-from-best
```

**Senaryo 3: Spesifik Checkpoint'ten Devam**

```bash
python train.py --resume checkpoints/epoch_50.pth
```

**Resume Özelliği:**

- ✅ Model ağırlıkları yüklenir
- ✅ Optimizer state (momentum, variance) restore edilir
- ✅ Learning rate scheduler position korunur
- ✅ Best accuracy tracking devam eder
- ✅ Early stopping patience counter korunur
- ✅ Training history grafiklerde kopukluk olmaz

**Console Output:**

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

- 🔴 **Kritik:** Elektrik kesintisi veya sistem çökmelerinde kayıp yok
- 🎯 **Esnek:** GPU timeout'larında bölümlere ayırabilme
- ⚡ **Verimli:** Hiperparametre değişiklikleriyle devam edebilme

---

### Adım 5: Model Evaluation

**Amaç:** Test setinde model performansını değerlendir.

```bash
python evaluate.py
```

**Beklenen Çıktı:**

```
results/
├── evaluation_report.json
├── confusion_matrix_raw.csv
├── confusion_matrix_normalized.csv
├── confusion_matrix_raw.png
├── confusion_matrix_normalized.png
├── per_class_metrics.csv
├── per_class_metrics.png
└── prediction_confidence.png
```

**Kontrol:**

```bash
cat results/evaluation_report.json | python -m json.tool
```

---

### Adım 6: Attention Visualization

**Amaç:** Transformer attention weights'leri görselleştir.

```bash
# Default: 5 sample
python visualize_attention.py

# Custom: 10 sample
python visualize_attention.py --num_samples 10
```

**Beklenen Çıktı:**

```
results/attention/
├── sample_{i}_layer_{l}_multihead.png
├── sample_{i}_layer_{l}_avg.png
├── sample_{i}_attention_rollout.png
├── layer_wise_attention_stats.png
└── head_wise_attention_stats.png
```

**Kontrol:**

```bash
ls results/attention/*.png | wc -l
```

---

## 🔍 Troubleshooting

### Problem 1: Val Setinde 0 Video Bulunuyor

**Belirtiler:**

```
Val: 0 videos (0.0%)
```

**Neden:** Eski script versiyonu yanlış label dosyası kullanıyordu.

**Çözüm:**
Script güncellenmiş durumda. Tekrar çalıştırın:

```bash
python scripts/01_select_videos.py
```

Beklenen: Val setinde 59 video bulunmalı.

---

### Problem 2: MediaPipe Import Hatası

```bash
ImportError: No module named 'mediapipe'
```

**Çözüm:**

```bash
pip install mediapipe opencv-python
```

---

### Problem 3: CUDA/GPU Bulunamadı

```bash
# CPU kullanımı için걱정 yok, otomatik CPU'ya geçer
# GPU kullanmak isterseniz:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Problem 4: Memory Error

```bash
# Batch size'ı küçült
# config.py'da BATCH_SIZE = 16 yap
```

---

### Problem 5: Keypoint Extraction Çok Yavaş

```bash
# NUM_WORKERS'ı artır (CPU core sayısına göre)
# config.py'da NUM_WORKERS = 4 (veya daha fazla)
```

---

## 📊 Beklenen Performans

### İlk 3 Kelime (acele, acikmak, agac):

**Baseline Beklentiler:**

- **Accuracy:** %70-85
- **F1-Score (macro):** %68-83
- **Training time:** 30-120 dakika (GPU/CPU)

**Not:** Bu ilk denemedir. Hiperparametre optimizasyonu ile iyileştirilebilir.

---

## 🎯 Sonraki Adımlar

1. **Hiperparametre Tuning:**

   - Learning rate grid search
   - Batch size optimization
   - Model architecture tweaks (d_model, num_layers, num_heads)
2. **Daha Fazla Kelime:**

   - config.py'da TARGET_CLASS_IDS genişlet
   - Pipeline'ı tekrar çalıştır
3. **Data Augmentation:**

   - Temporal augmentation
   - Spatial perturbations
   - Noise injection
4. **Model Ensembling:**

   - Farklı seed'lerle multiple model eğit
   - Voting/averaging ile ensemble

---

## 📁 Proje Yapısı Kontrolü

```bash
# Tüm yapıyı görüntüle
tree -L 2 transformer-signlang/

# Beklenen:
# transformer-signlang/
# ├── config.py
# ├── train.py
# ├── evaluate.py
# ├── visualize_attention.py
# ├── requirements.txt
# ├── README.md
# ├── RUN_PIPELINE.md
# ├── ilerleme.md
# ├── data/
# │   ├── selected_videos_train.csv
# │   ├── selected_videos_val.csv
# │   ├── selected_videos_test.csv
# │   ├── keypoints/
# │   ├── processed/
# │   └── scaler.pkl
# ├── scripts/
# │   ├── 01_select_videos.py
# │   ├── 02_extract_keypoints.py
# │   └── 03_normalize_data.py
# ├── models/
# │   ├── __init__.py
# │   └── transformer_model.py
# ├── checkpoints/
# │   ├── best_model.pth
# │   └── last_model.pth
# ├── results/
# │   ├── evaluation_report.json
# │   ├── confusion_matrix_*.png
# │   ├── per_class_metrics.*
# │   ├── prediction_confidence.png
# │   └── attention/
# └── logs/
#     └── training_history.json
```

---

## ✅ Quick Start Checklist

- [X] Conda environment aktif (`conda activate transformers`)
- [X] Dependencies yüklü (`pip install -r requirements.txt`)
- [X] Data dizinleri mevcut (`Data/Train Data/`, etc.)
- [X] Script 1: Video seçimi (`python scripts/01_select_videos.py`)
- [X] Script 2: Keypoint extraction (`python scripts/02_extract_keypoints.py`)
- [X] Script 3: Normalization (`python scripts/03_normalize_data.py`)
- [X] Training (`python train.py`)
- [X] Evaluation (`python evaluate.py`)
- [X] Visualization (`python visualize_attention.py`)
- [X] Results kontrol (`ls results/`)

---

## 💡 Tips

1. **GPU kullanımı:** CUDA available ise otomatik GPU kullanır
2. **Checkpoint'lerden devam:** train.py checkpoint'ten resume edebilir (kod eklenebilir)
3. **TensorBoard:** Opsiyonel olarak TensorBoard ile training takibi yapılabilir
4. **Weights & Biases:** W&B entegrasyonu ile experiment tracking (opsiyonel)

---

## 📞 Yardım

Herhangi bir sorun yaşarsanız:

1. `ilerleme.md` dosyasını kontrol edin
2. `README.md` dokümantasyonuna bakın
3. Linter errors: `python -m pylint <script.py>`
4. Import errors: `conda list` ile paket kontrolü

---

**🎉 Başarılar! Transformer Sign Language Recognition projeniz hazır!**
