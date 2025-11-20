# 🚀 10 Kelime İşaret Dili Tanıma - Çalıştırma Rehberi

**Tarih:** 7 Ekim 2025  
**Durum:** Sistem 10 kelime için hazır! 🎉

---

## 📋 ÖZET

**Hedef:** 3 kelime → 10 kelime genişleme  
**Kelimeler:** acele, acikmak, agac, anne, baba, ben, evet, hayir, iyi, tesekkur  
**Video Sayısı:** ~1,607 (1,243 train, 198 val, 166 test)  
**Tahmini Süre:** 5-6 saat (keypoint: 2-3h, training: 2-3h)

---

## ✅ HAZIRLIK DURUMU

| Adım | Durum | Açıklama |
|------|-------|----------|
| Config güncelleme | ✅ Tamamlandı | NUM_CLASSES=10, TARGET_CLASS_IDS güncellendi |
| Script kontrolü | ✅ Tamamlandı | Tüm scriptler 10 kelime için hazır |
| Veri hazırlama | ✅ Tamamlandı | CSV'ler oluşturuldu (1,607 video) |
| Yedekleme | ✅ Tamamlandı | 3-kelime sonuçları backups/3-kelime/ |

---

## 🎯 ÇALIŞTIRMA ADIMLARI

### 📂 ADIM 1: Environment Aktivasyonu

```bash
# Terminal'de:
conda activate transformers
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang
```

**Kontrol:**
```bash
python --version  # Python 3.10+ olmalı
which python      # transformers env'ındaki python olmalı
```

---

### 🎬 ADIM 2: Keypoint Extraction (2-3 SAAT ⏰)

```bash
python scripts/02_extract_keypoints.py
```

**Ne Yapılacak:**
- 1,607 videodan MediaPipe keypoint'leri çıkarılacak
- Her video için `.npy` dosyası oluşturulacak
- Çıktı: `data/keypoints/*.npy` (~1,607 dosya, ~80 MB)

**Beklenen Süre:** 2-3 saat

**Progress Takibi:**
- Progress bar gösterilecek
- Her video ~5 saniye sürer
- Kesinti olursa kaldığı yerden devam eder

**Doğrulama:**
```bash
# Kaç dosya oluşturuldu?
ls data/keypoints/*.npy | wc -l
# Beklenen: ~1,607

# Bir dosyanın şeklini kontrol et
python -c "import numpy as np; d=np.load('data/keypoints/signer0_sample16.npy'); print(d.shape)"
# Beklenen: (frame_count, 258)
```

---

### 📊 ADIM 3: Normalization ve Padding (5-10 dakika)

```bash
python scripts/03_normalize_data.py
```

**Ne Yapılacak:**
- Z-score normalization (scaler sadece train'de fit)
- Sequence padding/truncating (max_length hesaplanır)
- Train/val/test setlerini hazırlama
- Çıktı: `data/processed/*.npy` (9 dosya, ~500 MB)

**Beklenen Süre:** 5-10 dakika

**Beklenen Çıktı:**
```
data/processed/
├── X_train.npy        (1243, max_length, 258)
├── y_train.npy        (1243,)
├── train_ids.npy      
├── X_val.npy          (198, max_length, 258)
├── y_val.npy          (198,)
├── val_ids.npy        
├── X_test.npy         (166, max_length, 258)
├── y_test.npy         (166,)
└── test_ids.npy       

data/scaler.pkl
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
print(f'Unique labels: {sorted(set(y))}')
print(f'Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]')
"
```

---

### 🔍 ADIM 4: Setup Validation (Opsiyonel)

```bash
python validate_setup.py
```

**Ne Yapılacak:**
- Tüm setup'ın doğru olduğunu kontrol eder
- Device compatibility (MPS/CUDA/CPU)
- Data availability
- Class mapping doğrulaması

**Beklenen:** 7/7 checks PASSED

---

### 🎓 ADIM 5: Model Training (2-3 SAAT ⏰)

```bash
python train.py
```

**Ne Yapılacak:**
- Transformer modelini 10 sınıf için eğitir
- Best model'i kaydeder (`checkpoints/best_model.pth`)
- Early stopping ile otomatik dur (patience: 15 epoch)

**Beklenen Süre:**
- **MPS (M3 Mac):** 2-3 saat
- **CUDA GPU:** 1-2 saat
- **CPU:** 4-6 saat

**Model Hiperparametreleri:**
```
NUM_CLASSES = 10
BATCH_SIZE = 32 (bellek yetmezse 16'ya düşür)
LEARNING_RATE = 1e-4
NUM_ENCODER_LAYERS = 6
D_MODEL = 256
MAX_EPOCHS = 100
EARLY_STOPPING = 15
```

**Progress Takibi:**
- Her epoch'ta train/val loss ve accuracy gösterilir
- Best model otomatik kaydedilir
- `logs/training_history.json` dosyasını takip edebilirsin

**Beklenen Performans:**
- **Val Accuracy:** %85-90
- **Val F1-Score:** %83-88
- **Training Epochs:** 25-40 (early stopping)

**Kesinti Durumu:**
```bash
# Kaldığı yerden devam et
python train.py --resume checkpoints/last_model.pth
```

---

### 📈 ADIM 6: Evaluation

```bash
python evaluate.py
```

**Ne Yapılacak:**
- Test seti performansını ölçer
- Confusion matrix, per-class metrics
- Visualization'lar oluşturur

**Beklenen Süre:** 2-5 dakika

**Çıktılar:**
```
results/
├── evaluation_report.json              (metrics)
├── confusion_matrix_raw.csv            
├── confusion_matrix_normalized.csv     
├── confusion_matrix_raw.png            (10×10 heatmap)
├── confusion_matrix_normalized.png     (10×10 heatmap)
├── per_class_metrics.csv               (10 sınıf)
├── per_class_metrics.png               
├── prediction_confidence.png           
├── test_predictions.csv                (166 satır)
└── test_predictions.json
```

**Beklenen Test Accuracy:** %80-85

---

### 🎨 ADIM 7: Attention Visualization (Opsiyonel)

```bash
python visualize_attention.py --num_samples 5
```

**Ne Yapılacak:**
- 5 random test sample için attention haritaları
- Her layer'ın neye odaklandığını gösterir

**Beklenen Süre:** 5-10 dakika

**Çıktılar:**
```
results/attention/
├── sample_*_layer_*_multihead.png   (~30 dosya)
├── sample_*_layer_*_avg.png         (~30 dosya)
├── sample_*_attention_rollout.png   (5 dosya)
├── layer_wise_attention_stats.png   
└── head_wise_attention_stats.png    
```

---

## 📊 BAŞARI KRİTERLERİ

| Metrik | Hedef | Minimum |
|--------|-------|---------|
| **Test Accuracy** | %85-90 | %80+ |
| **Val Accuracy** | %85-90 | %80+ |
| **F1-Score (macro)** | %83-88 | %78+ |
| **Training Time** | <3 saat | <4 saat |

---

## 🚨 SORUN GİDERME

### Hata: "CUDA/MPS out of memory"
```bash
# config.py'da batch size küçült
# BATCH_SIZE = 16  # veya 8
```

### Hata: "FileNotFoundError: keypoints"
```bash
# ADIM 2'yi çalıştırmayı unuttun
python scripts/02_extract_keypoints.py
```

### Training çok yavaş
```bash
# Device kontrolü
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"

# MPS yoksa, CPU ile devam eder (yavaş ama çalışır)
```

### Bazı sınıflar F1 < %50
```bash
# Bu normal olabilir (10 sınıf 3'ten çok daha zor!)
# Öneriler:
# - Dropout artır: DROPOUT = 0.2
# - Data augmentation ekle
# - Daha fazla epoch eğit
```

---

## 📁 DOSYA YAPISI (Çalıştırma Sonrası)

```
transformer-signlang/
├── data/
│   ├── selected_videos_*.csv          (3 dosya) ✅
│   ├── keypoints/*.npy                (~1,607 dosya, 80 MB) ← ADIM 2
│   ├── processed/*.npy                (9 dosya, 500 MB) ← ADIM 3
│   └── scaler.pkl                     ← ADIM 3
├── checkpoints/
│   ├── best_model.pth                 (~32 MB) ← ADIM 5
│   └── last_model.pth                 (~32 MB) ← ADIM 5
├── logs/
│   └── training_history.json          ← ADIM 5
└── results/
    ├── *.json, *.csv, *.png           (8 dosya) ← ADIM 6
    └── attention/*.png                (~71 dosya) ← ADIM 7
```

**Toplam Disk Kullanımı:** ~1-1.5 GB

---

## ⏱️ ZAMAN ÇİZELGESİ

| Adım | Süre | Açıklama |
|------|------|----------|
| 1. Environment | 1 dk | Conda activate |
| 2. Keypoint extraction | 2-3 saat | ⏰ En uzun adım |
| 3. Normalization | 5-10 dk | Hızlı |
| 4. Validation | 1 dk | Opsiyonel |
| 5. Training | 2-3 saat | ⏰ İkinci en uzun |
| 6. Evaluation | 2-5 dk | Hızlı |
| 7. Visualization | 5-10 dk | Opsiyonel |
| **TOPLAM** | **5-7 saat** | **Kahve molası verilebilir!** ☕ |

---

## 💡 İPUÇLARI

1. **Keypoint extraction sırasında:** Bilgisayarı başka işler için kullanabilirsin, arka planda çalışır
2. **Training sırasında:** `logs/training_history.json` dosyasını başka terminal'de izleyebilirsin
3. **Kesinti:** Her iki uzun adım da (keypoint, training) kesintide kaldığı yerden devam eder
4. **Disk alanı:** ~1 GB gerekli, ~195 GB boş alan var ✅
5. **Yedek:** 3-kelime sonuçları `backups/3-kelime/` altında güvende

---

## 📊 SONUÇLARI YORUMLAMA REHBERİ

### 1️⃣ **evaluation_report.json** - Genel Performans

**Nasıl Açılır:**
```bash
cat results/evaluation_report.json | python -m json.tool
# veya
open results/evaluation_report.json  # Text editor ile
```

**Ne Bakmalı:**

**a) Overall Metrics (En Önemli):**
```json
"overall": {
    "accuracy": 0.8313,           // ← TEST ACCURACY (hedef: >0.80)
    "precision_macro": 0.8215,    // ← Ortalama precision
    "recall_macro": 0.8198,       // ← Ortalama recall
    "f1_macro": 0.8201,           // ← F1-SCORE (hedef: >0.78)
    "precision_weighted": 0.8298,
    "recall_weighted": 0.8313,
    "f1_weighted": 0.8299
}
```

**Yorumlama:**
- ✅ **Accuracy > 80%:** Çok iyi! 10 sınıf için başarılı
- ⚠️ **Accuracy 70-80%:** Kabul edilebilir, bazı sınıflar zor olabilir
- ❌ **Accuracy < 70%:** Sorun var, model yeniden eğitilmeli

**b) Per-Class Metrics:**
```json
"per_class": {
    "acele": {
        "precision": 0.9333,
        "recall": 0.8750,
        "f1_score": 0.9032,
        "support": 16          // ← Kaç örnek test edildi
    },
    ...
}
```

**Yorumlama:**
- **Precision yüksek, Recall düşük:** Model bu sınıfı tahmin etmekten çekiniyor (conservative)
- **Recall yüksek, Precision düşük:** Model bu sınıfı fazla tahmin ediyor (false positives)
- **F1-Score < 0.50:** Bu sınıf çok zor, daha fazla veri veya farklı yaklaşım gerekli

---

### 2️⃣ **confusion_matrix_normalized.png** - Hangi Sınıflar Karışıyor?

**Nasıl Açılır:**
```bash
open results/confusion_matrix_normalized.png
```

**Nasıl Okunur:**

```
         PREDICTED →
TRUE ↓   acele  acikmak  agac  anne  baba  ...
acele     0.88    0.06   0.00  0.06  0.00  ...  ← Bu satırı oku!
acikmak   0.05    0.90   0.05  0.00  0.00  ...
agac      0.00    0.00   0.94  0.06  0.00  ...
...
```

**Yorumlama Örnekleri:**

**Örnek 1: İyi Durum**
```
acele: [0.88, 0.06, 0.00, ...]
       ↑
       Diyagonal değer yüksek (0.88 = %88 doğru)
```
✅ Model "acele" işaretini %88 doğrulukla tanıyor

**Örnek 2: Karışan Sınıflar**
```
anne: [0.00, 0.00, 0.10, 0.70, 0.20, ...]
                            ↑     ↑
                          anne   baba
```
⚠️ "anne" işaretinin %20'si "baba" olarak tahmin ediliyor → Bu iki işaret benzer olabilir!

**Örnek 3: Dağılmış Tahminler**
```
hayir: [0.12, 0.15, 0.08, 0.20, 0.18, 0.10, 0.17]
```
❌ Tahminler dağılmış → Bu işaret çok zor, model kararsız

**Ne Yapmalı:**
- **Diagonal (köşegen) değerler yüksekse:** ✅ İyi performans
- **Belirli sınıf çiftleri karışıyorsa:** → Video örneklerini incele, benzer mi?
- **Bir sınıf çok dağıtık tahmin:** → Daha fazla eğitim verisi gerekebilir

---

### 3️⃣ **per_class_metrics.png** - Hangi Sınıf Daha Zor?

**Nasıl Açılır:**
```bash
open results/per_class_metrics.png
```

**Bar Chart Yorumlama:**

```
Precision ■  Recall ■  F1-Score ■

acele      |████████████| 0.93
acikmak    |████████████| 0.90
agac       |█████████   | 0.75  ← ⚠️ Düşük!
anne       |███████████ | 0.88
...
```

**Ne Bakmalı:**
- **Üç bar da yüksek (>0.85):** ✅ Sınıf başarılı
- **Üç bar da düşük (<0.70):** ❌ En zor sınıf, öncelik bu
- **Precision yüksek, Recall düşük:** Model çekingen
- **Recall yüksek, Precision düşük:** Model agresif

**Eylem Planı:**
1. En düşük F1-Score'lu 2-3 sınıfı belirle
2. Bu sınıfların videolarını izle
3. Neden zor olduklarını anla (hızlı hareket, benzer işaret, vb.)

---

### 4️⃣ **prediction_confidence.png** - Model Ne Kadar Emin?

**Nasıl Açılır:**
```bash
open results/prediction_confidence.png
```

**İki Grafik Var:**

**a) Histogram (Sol):**
```
Doğru tahminler  ■ (yeşil)
Yanlış tahminler ■ (kırmızı)

Frequency
    |     ■■■■  
    |    ■■■■■     ■
    |   ■■■■■■    ■■
    |  ■■■■■■■   ■■■  ■
    |─────────────────────
     0.0  0.5  0.7  1.0
         Confidence
```

**Yorumlama:**
- ✅ **Doğru tahminler sağda (>0.8):** Model emin ve doğru
- ⚠️ **Doğru tahminler ortada (0.5-0.7):** Model kararsız ama şanslı
- ❌ **Yanlış tahminler sağda (>0.8):** Model emin ama yanlış (en kötü!)

**b) Box Plot (Sağ):**
```
Sınıf bazında confidence dağılımı

acele   |━━━━━|  ← Yüksek, tutarlı
acikmak |━━━━━|
agac    |━━━|    ← Düşük, dağınık
```

**Ne Yapmalı:**
- Düşük confidence'lı sınıflar → Daha fazla eğitim verisi
- Yüksek confidence ama yanlış → Özellik mühendisliği gerekebilir

---

### 5️⃣ **test_predictions.csv** - Detaylı Tahmin Listesi

**Nasıl Açılır:**
```bash
# İlk 10 tahmini gör
head -10 results/test_predictions.csv

# Excel/LibreOffice ile aç
open results/test_predictions.csv
```

**Sütunlar:**
```csv
video_id,num_frames,true_class_id,true_class_name,pred_class_id,pred_class_name,confidence,is_correct
signer0_sample16,45,1,acele,1,acele,0.9234,True
signer1_sample32,67,2,acikmak,5,agac,0.7123,False  ← ⚠️ Yanlış tahmin!
...
```

**Yanlış Tahminleri Bul:**
```bash
# Sadece yanlış tahminleri filtrele
cat results/test_predictions.csv | grep ",False" > yanlis_tahminler.csv

# En düşük confidence'lı 10 tahmin
cat results/test_predictions.csv | sort -t',' -k7 -n | head -10
```

**Ne Yapmalı:**
1. Yanlış tahmin edilen videoları bul
2. Videoları izle (`Data/Test Data/.../test/{video_id}.mp4`)
3. Neden yanlış tahmin edildiğini anla

---

## 📊 BAŞARI DEĞERLENDİRME ÖZETİ

### ✅ **MÜKEMMEL SONUÇ** (Hedefi Aştı)
```
✅ Test Accuracy > 85%
✅ F1-Score (macro) > 83%
✅ Tüm sınıflar F1 > 75%
✅ Confusion matrix diagonal dominant
✅ Confidence ortalaması > 80%
```
**Yorum:** Model production'a hazır! 🎉

---

### 👍 **İYİ SONUÇ** (Hedefi Tuttu)
```
✅ Test Accuracy 80-85%
✅ F1-Score (macro) 78-83%
✅ Çoğu sınıf F1 > 70%
⚠️ 1-2 sınıf zor olabilir (F1 < 70%)
✅ Confidence ortalaması > 70%
```
**Yorum:** Başarılı! Zor sınıflar için iyileştirme yapılabilir.

---

### ⚠️ **KABUL EDİLEBİLİR** (Geliştirilebilir)
```
⚠️ Test Accuracy 70-80%
⚠️ F1-Score (macro) 70-78%
⚠️ 3-4 sınıf zor (F1 < 65%)
⚠️ Bazı sınıf çiftleri karışıyor
⚠️ Confidence dağınık
```
**Yorum:** Çalışıyor ama iyileştirme gerekli.

**İyileştirme Önerileri:**
1. Zor sınıflar için daha fazla veri ekle
2. Data augmentation kullan
3. Dropout artır (0.1 → 0.2)
4. Daha uzun eğit (early stopping patience artır)

---

### ❌ **ZAYIF SONUÇ** (Yeniden Eğitilmeli)
```
❌ Test Accuracy < 70%
❌ F1-Score (macro) < 70%
❌ Birçok sınıf F1 < 60%
❌ Confusion matrix dağınık
❌ Confidence düşük
```
**Yorum:** Ciddi sorun var!

**Olası Nedenler:**
- Veri kalitesi düşük (keypoint extraction hatalı)
- Model çok küçük veya çok büyük
- Overfitting (train acc yüksek, test düşük)
- Underfitting (her iki acc de düşük)

---

## 📝 SONRAKI ADIMLAR

Pipeline tamamlandıktan sonra:

1. **Sonuçları İncele (ÖNEMLİ!):**
   - `results/evaluation_report.json` → **İLK BAK BURAYA!** (Overall metrics)
   - `results/confusion_matrix_normalized.png` → Hangi sınıflar karışıyor?
   - `results/per_class_metrics.png` → Hangi sınıf daha zor?
   - `results/prediction_confidence.png` → Model ne kadar emin?
   - `results/test_predictions.csv` → Yanlış tahminleri incele

2. **3-Kelime ile Karşılaştır:**
   - 3-kelime: %90 accuracy
   - 10-kelime: %80-85 accuracy (beklenen)
   - Normal düşüş (10 sınıf 3'ten çok daha zor!)

3. **Rapor Oluştur:**
   - İlerleme dosyasını tamamla
   - Sonuçları `10-kelime-rapor.md`'ye yaz
   - Yukarıdaki yorumlama rehberini kullan
   - Zor sınıfları ve iyileştirme önerilerini belirt

4. **Gelecek Planı:**
   - 10-kelime başarılıysa → 25-50 kelimeye geç
   - Sorunlar varsa → İyileştir, tekrar eğit

---

## ✅ HAZIRLIK KONTROLÜ

Başlamadan önce kontrol et:

```bash
- [ ] Conda environment aktif: `conda activate transformers`
- [ ] Dizindeyim: `cd transformer-signlang`
- [ ] Config güncel: `python -c "from config import TransformerConfig; print(TransformerConfig.NUM_CLASSES)"`  → 10 olmalı
- [ ] CSV'ler hazır: `ls data/selected_videos_*.csv`  → 3 dosya olmalı
- [ ] Disk yeterli: `df -h .`  → >1 GB boş olmalı
- [ ] Yedek alındı: `ls backups/3-kelime/`  → results/, checkpoints/, 3-kelime.md olmalı
```

---

**🎉 HER ŞEY HAZIR! Çalıştırmaya başlayabilirsin!**

**İlk komut:**
```bash
conda activate transformers
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang
python scripts/02_extract_keypoints.py
```

**Kolay gelsin! ☕🚀**

