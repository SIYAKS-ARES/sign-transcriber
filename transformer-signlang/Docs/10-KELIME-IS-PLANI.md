# 🎯 10 Kelime İşaret Dili Tanıma - İş Planı

## 📋 Proje Özeti

**Hedef:** 3 kelimeden 10 kelimeye genişletme  
**Süre Tahmini:** 5-6 saat (keypoint: 2-3 saat, training: 2-3 saat)  
**Yaklaşım:** Her adım onay sonrası ilerle  
**Beklenen Accuracy:** %80-85

---

## 🎯 Seçilen 10 Kelime

Mevcut 3 kelimeyi koruyarak dengeli bir set:

| # | ClassId | TR | EN | Kategori | Neden Seçildi |
|---|---------|----|----|----------|---------------|
| 1 | 1 | acele | hurry | Hareket | ✅ Mevcut - Zor sınıf |
| 2 | 2 | acikmak | hungry | Durum | ✅ Mevcut - Mükemmel |
| 3 | 5 | agac | tree | Nesne | ✅ Mevcut - Mükemmel |
| 4 | 14 | anne | mother | Aile | Temel kelime, net işaret |
| 5 | 20 | baba | father | Aile | Anne ile karşılaştırma |
| 6 | 30 | ben | I | Zamir | Kendine işaret, basit |
| 7 | 65 | evet | yes | Onay | Baş hareketi, distinctive |
| 8 | 86 | hayir | no | Red | Evet ile karşıt |
| 9 | 100 | iyi | good | Sıfat | Pozitif ifade |
| 10 | 196 | tesekkur | thanks | Nezaket | Kompleks hareket |

**Seçim Kriterleri:**
- ✅ Farklı kategoriler (aile, zamir, onay/red, sıfat, nezaket)
- ✅ Görsel olarak birbirinden farklı
- ✅ Basit (ben, evet) ve kompleks (tesekkur) karışımı
- ✅ Mevcut 3 kelime korunuyor (karşılaştırma için)

---

## 📊 Beklenen Veri Miktarı

3 kelime → 10 kelime genişlemesi:

| Metrik | 3 Kelime (Mevcut) | 10 Kelime (Hedef) | Artış |
|--------|-------------------|-------------------|-------|
| **Train Videos** | 373 | ~1,240 | 3.3x |
| **Val Videos** | 59 | ~196 | 3.3x |
| **Test Videos** | 50 | ~166 | 3.3x |
| **Toplam** | 482 | **~1,602** | **3.3x** |
| **Keypoint Dosyası** | 482 × 50KB = 24 MB | ~1,602 × 50KB = **~80 MB** | 3.3x |
| **Processed Data** | ~150 MB | **~500 MB** | 3.3x |

**Disk İhtiyacı:** ~1 GB boş alan

---

## 🔄 ADIM ADIM İŞ PLANI

### ✅ ADIM 0: HAZIRLIK VE DOĞRULAMA
**Süre:** 5 dakika  
**Durum:** ✋ ONAY BEKLİYOR

**Yapılacaklar:**
1. Seçilen 10 kelimeyi onayla
2. Disk alanını kontrol et (~1 GB gerekli)
3. Mevcut 3-kelime sonuçlarını yedekle

**Komutlar:**
```bash
# Disk kontrolü
df -h /Users/siyaksares/Developer/GitHub/klassifier-sign-language

# Mevcut sonuçları yedekle
cd transformer-signlang
mkdir -p backups/3-kelime
cp -r results backups/3-kelime/
cp -r checkpoints backups/3-kelime/
cp 3-kelime.md backups/3-kelime/
```

**Çıktı:**
- [ ] Disk alanı yeterli (>1 GB)
- [ ] Yedek alındı
- [ ] 10 kelime onaylandı

---

### ✅ ADIM 1: CONFIG GÜNCELLEME
**Süre:** 2 dakika  
**Durum:** ⏸️ Adım 0 sonrası

**Yapılacaklar:**
1. `config.py` dosyasını güncelle
2. `TARGET_CLASS_IDS` değiştir: `[1, 2, 5]` → `[1, 2, 5, 14, 20, 30, 65, 86, 100, 196]`
3. `NUM_CLASSES` değiştir: `3` → `10`

**Güncellenecek Satırlar:**
```python
# config.py
TARGET_CLASS_IDS = [1, 2, 5, 14, 20, 30, 65, 86, 100, 196]  # 10 kelime
NUM_CLASSES = 10
```

**Doğrulama:**
```bash
python -c "from config import TransformerConfig; c=TransformerConfig(); print(f'Classes: {c.NUM_CLASSES}, IDs: {c.TARGET_CLASS_IDS}')"
```

**Beklenen Çıktı:**
```
Classes: 10, IDs: [1, 2, 5, 14, 20, 30, 65, 86, 100, 196]
```

**Çıktı:**
- [ ] Config güncellendi
- [ ] Doğrulama başarılı

---

### ✅ ADIM 2: VIDEO SEÇİMİ
**Süre:** 30 saniye  
**Durum:** ⏸️ Adım 1 sonrası

**Yapılacaklar:**
1. `scripts/01_select_videos.py` çalıştır
2. 10 kelimeye ait videoları seç (train/val/test)
3. CSV dosyaları oluştur

**Komut:**
```bash
cd transformer-signlang
python scripts/01_select_videos.py
```

**Beklenen Çıktı:**
```
✅ Train: ~1,240 videos
✅ Val:   ~196 videos
✅ Test:  ~166 videos
─────────────────────────────
Total: ~1,602 videos

Sınıf Dağılımı:
  ClassId 1 (acele):     ~124 train, ~19 val, ~16 test
  ClassId 2 (acikmak):   ~123 train, ~20 val, ~17 test
  ClassId 5 (agac):      ~125 train, ~20 val, ~17 test
  ClassId 14 (anne):     ~124 train, ~20 val, ~17 test
  ClassId 20 (baba):     ~124 train, ~20 val, ~16 test
  ClassId 30 (ben):      ~124 train, ~19 val, ~17 test
  ClassId 65 (evet):     ~124 train, ~20 val, ~16 test
  ClassId 86 (hayir):    ~124 train, ~19 val, ~17 test
  ClassId 100 (iyi):     ~124 train, ~20 val, ~17 test
  ClassId 196 (tesekkur): ~124 train, ~19 val, ~16 test
```

**Oluşan Dosyalar:**
```
data/selected_videos_train.csv   (~1,240 satır)
data/selected_videos_val.csv     (~196 satır)
data/selected_videos_test.csv    (~166 satır)
```

**Doğrulama:**
```bash
wc -l data/selected_videos_*.csv
```

**Çıktı:**
- [ ] CSV dosyaları oluşturuldu
- [ ] Video sayıları doğru
- [ ] Sınıf dağılımı dengeli

---

### ✅ ADIM 3: KEYPOINT EXTRACTION (EN UZUN ADIM!)
**Süre:** 2-3 SAAT ⏰  
**Durum:** ⏸️ Adım 2 sonrası

**Yapılacaklar:**
1. `scripts/02_extract_keypoints.py` çalıştır
2. ~1,602 videodan MediaPipe keypoint'leri çıkar
3. Her video için `.npy` dosyası oluştur

**⚠️ ÖNEMLİ UYARILAR:**
- **Bu adım 2-3 saat sürecek!**
- Bilgisayar uyku moduna geçmemeli
- Progress bar ile ilerleme takip edilebilir
- Kesinti olursa kaldığı yerden devam eder

**Komut:**
```bash
cd transformer-signlang
python scripts/02_extract_keypoints.py
```

**Progress Takibi:**
```
Processing videos: 100%|████████████| 1602/1602 [2:15:30<00:00, 5.07s/video]

Frame statistics:
  Min frames:    40
  Max frames:    120
  Mean frames:   65.3
  Median frames: 62

✅ Başarıyla işlenen: 1,602 video
❌ Hatalı videolar:   0
```

**Oluşan Dosyalar:**
```
data/keypoints/
├── signer0_sample16.npy     (existing - 3 kelime)
├── signer0_sample25.npy     (existing - 3 kelime)
├── ...
├── signerX_sampleY.npy      (new - 7 yeni kelime)
└── ...
Toplam: ~1,602 .npy dosyası (~80 MB)
```

**Doğrulama:**
```bash
# Kaç dosya oluşturuldu?
ls data/keypoints/*.npy | wc -l
# Beklenen: ~1,602

# Bir dosyanın şeklini kontrol et
python -c "import numpy as np; d=np.load('data/keypoints/signer0_sample16.npy'); print(d.shape)"
# Beklenen: (frame_count, 258)
```

**Çıktı:**
- [ ] Tüm videolar işlendi
- [ ] Keypoint dosyaları oluşturuldu (~1,602 adet)
- [ ] Dosya boyutları mantıklı

**💡 İpucu:** Bu adım sırasında başka işler yapılabilir, bilgisayar arka planda çalışacak.

---

### ✅ ADIM 4: NORMALIZATION VE PADDING
**Süre:** 5-10 dakika  
**Durum:** ⏸️ Adım 3 sonrası

**Yapılacaklar:**
1. `scripts/03_normalize_data.py` çalıştır
2. Z-score normalization (scaler sadece train'de fit)
3. Sequence padding/truncating (max_length hesapla)
4. Train/val/test setlerini hazırla

**Komut:**
```bash
cd transformer-signlang
python scripts/03_normalize_data.py
```

**Beklenen Çıktı:**
```
📊 Keypoint dosyaları yükleniyor...
   ✅ Train: 1,240 videos loaded
   ✅ Val:   196 videos loaded
   ✅ Test:  166 videos loaded

📈 Sekans uzunlukları analizi:
   Min:    40 frames
   Max:    120 frames
   Mean:   65.3 frames
   Median: 62 frames
   95th percentile: 95 frames

🔧 Scaler fit ediliyor (SADECE TRAIN)...
   ✅ StandardScaler fit edildi (1,240 videoda)

📏 MAX_SEQ_LENGTH belirlendi: 95 frames

🔄 Normalization ve padding...
   Train: 100%|████████████| 1240/1240
   Val:   100%|████████████| 196/196
   Test:  100%|████████████| 166/166

💾 Dosyalar kaydediliyor...
   ✅ data/processed/X_train.npy    (1240, 95, 258)
   ✅ data/processed/y_train.npy    (1240,)
   ✅ data/processed/X_val.npy      (196, 95, 258)
   ✅ data/processed/y_val.npy      (196,)
   ✅ data/processed/X_test.npy     (166, 95, 258)
   ✅ data/processed/y_test.npy     (166,)
   ✅ data/scaler.pkl

📊 Label distribution:
   Label 0 (ClassId 1):   124 train, 19 val, 16 test
   Label 1 (ClassId 2):   123 train, 20 val, 17 test
   Label 2 (ClassId 5):   125 train, 20 val, 17 test
   Label 3 (ClassId 14):  124 train, 20 val, 17 test
   Label 4 (ClassId 20):  124 train, 20 val, 16 test
   Label 5 (ClassId 30):  124 train, 19 val, 17 test
   Label 6 (ClassId 65):  124 train, 20 val, 16 test
   Label 7 (ClassId 86):  124 train, 19 val, 17 test
   Label 8 (ClassId 100): 124 train, 20 val, 17 test
   Label 9 (ClassId 196): 124 train, 19 val, 16 test

✅ Tamamlandı!
```

**Doğrulama:**
```bash
# Shape kontrol
python -c "
import numpy as np
print('Train:', np.load('data/processed/X_train.npy').shape)
print('Val:  ', np.load('data/processed/X_val.npy').shape)
print('Test: ', np.load('data/processed/X_test.npy').shape)
"

# Label kontrol (0-9 arası olmalı)
python -c "
import numpy as np
y = np.load('data/processed/y_train.npy')
print(f'Labels: {sorted(set(y))}')
print(f'Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]')
"
```

**Çıktı:**
- [ ] Processed dosyaları oluşturuldu
- [ ] Shape'ler doğru
- [ ] Labels 0-indexed (0-9)
- [ ] Sınıf dağılımı dengeli

---

### ✅ ADIM 5: VALIDATION CHECK
**Süre:** 1 dakika  
**Durum:** ⏸️ Adım 4 sonrası

**Yapılacaklar:**
1. `validate_setup.py` çalıştır
2. Tüm setup'ın doğru olduğunu kontrol et
3. Training öncesi final check

**Komut:**
```bash
cd transformer-signlang
python validate_setup.py
```

**Beklenen Çıktı:**
```
================================================================================
🔍 TRANSFORMER SIGN LANGUAGE - SETUP VALIDATION
================================================================================

✅ Python Version:       PASSED (3.12.11)
✅ Dependencies:         PASSED (all required packages installed)
✅ Project Structure:    PASSED (all files present)
✅ Configuration:        PASSED (10 classes, 10 target IDs)
✅ Device Compatibility: PASSED (MPS available)
✅ Data Availability:    PASSED (1240 train, 196 val, 166 test)
✅ Class Mapping:        PASSED (labels 0-9, ClassIds [1,2,5,14,20,30,65,86,100,196])

================================================================================
📊 SUMMARY: 7/7 checks PASSED
================================================================================

✅ System is ready for training!
```

**Çıktı:**
- [ ] Tüm validasyonlar PASSED
- [ ] Eğitime hazır

---

### ✅ ADIM 6: MODEL TRAINING
**Süre:** 2-3 SAAT ⏰  
**Durum:** ⏸️ Adım 5 sonrası

**Yapılacaklar:**
1. `train.py` çalıştır
2. Transformer modelini 10 sınıf için eğit
3. Best model'i kaydet

**⚠️ ÖNEMLİ UYARILAR:**
- **Bu adım 2-3 saat sürecek!**
- GPU kullanımı önerilir (MPS/CUDA)
- Checkpoint resume aktif (kesinti olursa devam eder)
- Early stopping (patience: 15 epoch)

**Komut:**
```bash
cd transformer-signlang
python train.py
```

**Model Hiperparametreleri:**
```python
NUM_CLASSES = 10        # 3'ten 10'a çıktı
BATCH_SIZE = 32         # Aynı (bellek yeterse)
LEARNING_RATE = 1e-4    # Aynı
NUM_ENCODER_LAYERS = 6  # Aynı
D_MODEL = 256           # Aynı
MAX_EPOCHS = 100        # Aynı
EARLY_STOPPING = 15     # Aynı
```

**Beklenen Progress:**
```
================================================================================
🚀 TRANSFORMER TRAINING - 10 CLASSES
================================================================================

🖥️  Device: MPS (Apple Silicon GPU)

📊 Data Shapes:
   Train: (1240, 95, 258)
   Val:   (196, 95, 258)

🏗️  Model: Transformer (6 layers, 8 heads, 256 d_model)
   Total params: 8.2M
   Trainable params: 8.2M

================================================================================
Epoch 1/100
================================================================================
Training:   100%|████████████| 39/39 [02:15<00:00, 3.47s/batch]
Validation: 100%|████████████| 7/7 [00:12<00:00, 1.78s/batch]

📊 Epoch 1 Results:
   Train Loss: 2.145 | Train Acc: 15.2%
   Val Loss:   1.987 | Val Acc:   22.4% | Val F1: 18.7%
   LR: 1.09e-05

...

================================================================================
Epoch 25/100
================================================================================
Training:   100%|████████████| 39/39 [02:12<00:00, 3.40s/batch]
Validation: 100%|████████████| 7/7 [00:11<00:00, 1.65s/batch]

📊 Epoch 25 Results:
   Train Loss: 0.245 | Train Acc: 92.3%
   Val Loss:   0.312 | Val Acc:   88.8% | Val F1: 87.2%
   LR: 9.95e-05

   ✅ Best model saved! (Val Acc: 88.8%)

...

⏹️  Early stopping at epoch 40
   Best Val Accuracy: 89.3% (Epoch 35)

✅ Training completed!
```

**Oluşan Dosyalar:**
```
checkpoints/
├── best_model.pth       (~32 MB)
└── last_model.pth       (~32 MB)

logs/
└── training_history.json
```

**Doğrulama:**
```bash
# Best model info
python -c "
import torch
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Val Acc: {ckpt[\"val_acc\"]:.4f}')
print(f'Val F1: {ckpt[\"val_f1\"]:.4f}')
"
```

**Beklenen Performans:**
- **Val Accuracy:** %85-90
- **Val F1:** %83-88
- **Training Time:** 2-3 saat

**Çıktı:**
- [ ] Eğitim tamamlandı
- [ ] Best model kaydedildi
- [ ] Val accuracy %85+ (hedef)

**💡 İpucu:** Eğitim sırasında `logs/training_history.json` dosyasını takip edebilirsiniz.

---

### ✅ ADIM 7: EVALUATION
**Süre:** 5 dakika  
**Durum:** ⏸️ Adım 6 sonrası

**Yapılacaklar:**
1. `evaluate.py` çalıştır
2. Test seti performansını ölç
3. Confusion matrix ve metrics oluştur

**Komut:**
```bash
cd transformer-signlang
python evaluate.py
```

**Beklenen Çıktı:**
```
================================================================================
📊 TEST RESULTS
================================================================================

Overall Performance:
   Accuracy:           85.5%
   Precision (macro):  84.2%
   Recall (macro):     83.8%
   F1-Score (macro):   83.6%

Per-Class Performance:
   acele (1):     Precision: 78%, Recall: 75%, F1: 76%
   acikmak (2):   Precision: 100%, Recall: 100%, F1: 100%
   agac (5):      Precision: 94%, Recall: 100%, F1: 97%
   anne (14):     Precision: 88%, Recall: 82%, F1: 85%
   baba (20):     Precision: 85%, Recall: 88%, F1: 86%
   ben (30):      Precision: 92%, Recall: 94%, F1: 93%
   evet (65):     Precision: 88%, Recall: 81%, F1: 84%
   hayir (86):    Precision: 82%, Recall: 85%, F1: 83%
   iyi (100):     Precision: 79%, Recall: 76%, F1: 77%
   tesekkur (196): Precision: 73%, Recall: 71%, F1: 72%

✅ Results saved to results/
```

**Oluşan Dosyalar:**
```
results/
├── evaluation_report.json
├── confusion_matrix_normalized.png   (10×10 matrix)
├── confusion_matrix_raw.png
├── per_class_metrics.png
├── prediction_confidence.png
├── test_predictions.csv              (166 rows)
└── test_predictions.json
```

**Doğrulama:**
```bash
# Results kontrol
ls -lh results/
cat results/evaluation_report.json | python -m json.tool | head -30
```

**Çıktı:**
- [ ] Evaluation tamamlandı
- [ ] Test accuracy %80+ (hedef)
- [ ] Results dosyaları oluşturuldu

---

### ✅ ADIM 8: ATTENTION VISUALIZATION
**Süre:** 5-10 dakika  
**Durum:** ⏸️ Adım 7 sonrası

**Yapılacaklar:**
1. `visualize_attention.py` çalıştır
2. Attention haritalarını oluştur
3. Model'in neye odaklandığını gör

**Komut:**
```bash
cd transformer-signlang
python visualize_attention.py --num_samples 5
```

**Beklenen Çıktı:**
```
================================================================================
🎨 ATTENTION VISUALIZATION - 10 CLASSES
================================================================================

Processing sample 1/5...
   True: anne (ClassId 14)
   Pred: anne (94.2% confidence)
   ✅ Saved: results/attention/sample_0_*.png

Processing sample 2/5...
   True: tesekkur (ClassId 196)
   Pred: tesekkur (78.5% confidence)
   ✅ Saved: results/attention/sample_1_*.png

...

✅ Total visualizations created: 71 PNG files
```

**Oluşan Dosyalar:**
```
results/attention/
├── sample_0_layer_*_multihead.png   (6 layers × 5 samples = 30)
├── sample_0_layer_*_avg.png         (6 layers × 5 samples = 30)
├── sample_*_attention_rollout.png   (5 samples)
├── layer_wise_attention_stats.png   (1)
└── head_wise_attention_stats.png    (1)
Toplam: ~71 dosya
```

**Çıktı:**
- [ ] Attention visualizations oluşturuldu
- [ ] 71 görsel dosya

---

### ✅ ADIM 9: SONUÇ RAPORU OLUŞTURMA
**Süre:** 5 dakika  
**Durum:** ⏸️ Adım 8 sonrası

**Yapılacaklar:**
1. Kapsamlı değerlendirme raporu oluştur
2. 3-kelime ile 10-kelime karşılaştırması
3. İyileştirme önerileri

**El ile oluşturulacak:** `10-kelime-rapor.md`

**İçerik:**
- Overall performance metrics
- Per-class breakdown
- 3-kelime vs 10-kelime comparison
- Confusion matrix analysis
- Attention insights
- Hangi kelimeler zor?
- İyileştirme önerileri
- Sonraki adımlar (25-50-226 kelime)

**Çıktı:**
- [ ] Rapor oluşturuldu
- [ ] Sonuçlar analiz edildi

---

## 📊 BAŞARI KRİTERLERİ

| Metrik | Hedef | Minimum Kabul |
|--------|-------|---------------|
| **Test Accuracy** | %85-90 | %80+ |
| **Val Accuracy** | %85-90 | %80+ |
| **F1-Score (macro)** | %83-88 | %78+ |
| **Training Time** | <3 saat | <4 saat |
| **Tüm sınıflar F1** | >%70 | >%65 |

---

## 🚨 RISK YÖNETİMİ

### Risk 1: Keypoint Extraction Çok Uzun Sürüyor
**Belirti:** 3 saatten uzun sürüyor  
**Çözüm:** 
- Model complexity azalt (config'de `model_complexity=0`)
- Batch processing ekle
- Kesinti olursa kaldığı yerden devam eder (zaten hazır)

### Risk 2: Training Overfitting
**Belirti:** Train acc %95+, Val acc %75-  
**Çözüm:**
- Dropout artır (0.1 → 0.2)
- Label smoothing artır (0.1 → 0.15)
- Data augmentation ekle

### Risk 3: Bellek Yetersiz
**Belirti:** CUDA/MPS out of memory  
**Çözüm:**
- Batch size küçült (32 → 16 → 8)
- Model küçült (d_model: 256 → 128)

### Risk 4: Bazı Sınıflar Çok Zor
**Belirti:** 2-3 sınıf F1 <%50  
**Çözüm:**
- Focal loss kullan (zor sınıflara odaklan)
- Class weights ekle
- O sınıfları temporal augment et

---

## 📝 CHECKPOINT VE YEDEKLEME

### Önemli Checkpoint'ler
1. **Adım 3 sonrası:** Keypoint'ler hazır → yedekle!
2. **Adım 4 sonrası:** Processed data hazır → yedekle!
3. **Adım 6 sonrası:** Best model eğitildi → yedekle!

### Yedekleme Komutu
```bash
# Kritik dosyaları yedekle
mkdir -p backups/10-kelime-$(date +%Y%m%d)
cp -r data/keypoints backups/10-kelime-$(date +%Y%m%d)/
cp -r data/processed backups/10-kelime-$(date +%Y%m%d)/
cp -r checkpoints backups/10-kelime-$(date +%Y%m%d)/
cp -r results backups/10-kelime-$(date +%Y%m%d)/
```

---

## ✅ FINAL CHECKLIST

Pipeline tamamlandığında:

```bash
# 1. Data hazır mı?
[ ] ls data/selected_videos_*.csv  # 3 CSV
[ ] ls data/keypoints/*.npy | wc -l  # ~1,602 dosya
[ ] ls data/processed/*.npy  # 6 .npy dosyası

# 2. Model eğitildi mi?
[ ] ls checkpoints/best_model.pth
[ ] ls checkpoints/last_model.pth

# 3. Evaluation tamamlandı mı?
[ ] ls results/*.json  # evaluation_report.json
[ ] ls results/*.png  # 4 görsel
[ ] ls results/*.csv  # 3 CSV

# 4. Attention viz tamamlandı mı?
[ ] ls results/attention/*.png | wc -l  # ~71 dosya

# 5. Rapor hazır mı?
[ ] cat 10-kelime-rapor.md
```

---

## 📞 YARDIM VE DESTEK

Her adımda sorun yaşarsan:

1. **Hata mesajını oku** (genelde ne yapman gerektiğini söyler)
2. **validate_setup.py çalıştır** (setup doğru mu?)
3. **İlgili script'in başındaki docstring'e bak** (kullanım talimatları)
4. **İlerleme dosyasına bak:** `ilerleme.md` (benzer sorunlar yaşandı mı?)

---

**🎯 HAZIRSAN ADIM 0 İLE BAŞLAYALIM!**

**Onay bekleniyor...** ✋

