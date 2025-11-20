# 📊 TRANSFORMER SIGN LANGUAGE MODEL - KAPSAMLI ANALİZ RAPORU

**Tarih:** 9 Ekim 2025
**Model:** Transformer-based Sign Language Classifier
**Dataset:** AUTSL (226 Türk İşaret Dili Kelimesi)
**Training Duration:** 2 saat 54 dakika (100 epoch)

---

## 📈 1. PERFORMANS ÖZETİ

### Genel Metrikler

| Split                | Samples | Accuracy          | F1 Score (Macro) | F1 Score (Weighted) |
| -------------------- | ------- | ----------------- | ---------------- | ------------------- |
| **Train**      | 28,142  | **100.00%** | -                | -                   |
| **Validation** | 4,418   | **87.87%**  | **87.56%** | -                   |
| **Test**       | 3,742   | **76.96%**  | **76.19%** | **76.44%**    |

### Kritik Gözlemler

✅ **Güçlü Yönler:**

- Model train setini %100 öğrenmiş
- 15 sınıfta mükemmel performans (100% F1)
- Genel test accuracy %76.96 (226 sınıf için makul)

⚠️ **Zayıf Yönler:**

- **Ciddi Overfitting**: Train-Val gap %12.13, Val-Test gap %10.91
- **Sınıf Dengesizliği**: 3 sınıf %0 F1 (hiç tahmin edilememiş!)
- **Generalization Problemi**: Test performansı beklenenden düşük

---

## 🔴 2. OVERFİTTİNG ANALİZİ

### Training Curve İstatistikleri

```
Total Epochs: 100
Best Train Acc: 100.00% (Epoch 78)
Best Val Acc:   87.87% (Epoch 98)
Best Val F1:    87.56% (Epoch 98)

Train-Val Gap (final): 12.29%
Val-Test Gap:          10.91%
```

### Son 10 Epoch Performansı

| Epoch        | Train Acc         | Val Acc          | Gap                      |
| ------------ | ----------------- | ---------------- | ------------------------ |
| 91           | 100.00%           | 87.51%           | 12.49%                   |
| 92           | 100.00%           | 87.71%           | 12.29%                   |
| 93           | 100.00%           | 87.82%           | 12.18%                   |
| 94           | 100.00%           | 87.85%           | 12.15%                   |
| 95           | 100.00%           | 87.78%           | 12.22%                   |
| 96           | 100.00%           | 87.75%           | 12.25%                   |
| 97           | 100.00%           | 87.87%           | 12.13%                   |
| **98** | **100.00%** | **87.87%** | **12.13%** ← BEST |
| 99           | 100.00%           | 87.71%           | 12.29%                   |
| 100          | 100.00%           | 87.71%           | 12.29%                   |

**Teşhis:**

- Model Epoch 78'de train setini %100 ezberlemis
- Val accuracy %87-88 civarında plato yapmış
- **Tipik overfitting pattern**: Regularization yetersiz

---

## 🏆 3. SINIF BAZLI PERFORMANS

### 🥇 Mükemmel Performans (100% F1) - 15 Sınıf

```
cuzdan, dusman, eczane, gol, hastane, hayirli_olsun, 
kolonya, komur, kopru, oda, pencere, salca, serbest, 
uzak, yemek_pisirmek, yorgun
```

### ⭐ Çok İyi (90-99% F1) - 41 Sınıf

```
ozur_dilemek (97.14%), benzin (97.14%), kopek (97.14%),
bayram (96.97%), bekar (96.97%), calismak (96.77%),
devirmek (96.97%), hep (96.97%), kemer (96.97%), kiz (96.97%),
pantolon (96.97%), tatil (96.97%), yasak (96.97%), yatak (96.97%),
acikmak (91.89%), anne (91.89%), ... (26 sınıf daha)
```

### ⚠️ Orta Performans (50-75% F1) - 89 Sınıf

En düşük performanslı örnekler:

- **masa**: 51.61% (karışık tahminler)
- **evli**: 52.00% (savas ile karışıyor)
- **bal**: 53.06%
- **para**: 53.57%

### ❌ BAŞARISIZ SINIFLAR (0-30% F1) - 12 Sınıf

#### 🔴 Hiç Tahmin Edilemeyenler (0% F1):

1. **nasil** (16 örnek) - %0 recall!
2. **okul** (17 örnek) - %0 recall!
3. **seker** (11 örnek) - %0 recall!

#### 🟠 Çok Düşük Performans (<30% F1):

4. **ilac**: 21.05% F1 (Precision 100%, Recall 11.76%)
5. **dakika**: 30.00% F1 (Precision 50%, Recall 21.43%)
6. **iyi**: 30.00% F1 (Precision 100%, Recall 17.65%)
7. **kotu**: 30.00% F1 (Precision 100%, Recall 17.65%)
8. **olur**: 30.00% F1 (Precision 100%, Recall 17.65%)
9. **yapmak**: 30.00% F1 (Precision 100%, Recall 17.65%)
10. **ataturk**: 24.56% F1 (Precision 16.28%, Recall 50%)
11. **oruc**: 27.27% F1 (Precision 50%, Recall 18.75%)
12. **tamam**: 31.58% F1 (Precision 100%, Recall 18.75%)

---

## 🔍 4. BAŞARISIZ SINIFLARIN KÖK NEDEN ANALİZİ

### A) Yüksek Precision, Düşük Recall Pattern

**Sınıflar:** `ilac, iyi, kotu, olur, yapmak, tamam`

**Precision: 100%, Recall: ~17-21%**

**Kök Neden:**

- Model bu sınıfları **tanıdığında çok emin** (precision yüksek)
- Ama çoğu örnekte **tahmin etmeyi tercih etmiyor** (recall düşük)
- **Muhtemel sebep**: Başka sınıflarla karıştırılıyor

**Çözüm Önerileri:**

1. Class weights uygula (rare class'lara daha fazla weight)
2. Focal Loss kullan (hard examples'a odaklan)
3. Data augmentation ile bu sınıfları artır

---

### B) Düşük Precision, Orta Recall Pattern

**Sınıflar:** `ataturk, bardak, aile, bal, para`

**Precision: 16-45%, Recall: 47-88%**

**Kök Neden:**

- Model bu sınıfları **sıkça tahmin ediyor** ama **yanlış** (recall yüksek, precision düşük)
- **False positive** çok fazla
- Başka sınıfları bu sınıf olarak etiketliyor

**Muhtemel Karışan Sınıflar:**

- `ataturk` → muhtemelen benzer el işaretleri olan sınıflarla (örn: `tarih`, `devlet`)
- `bardak` → `cay`, `su` gibi sınıflarla
- `para` → `kredi_karti`, `banka` gibi

**Çözüm Önerileri:**

1. Confusion matrix'i detaylı incele
2. Discriminative features öğrenmek için contrastive learning
3. Hard negative mining

---

### C) Tamamen Başarısız Pattern

**Sınıflar:** `nasil, okul, seker` (0% F1)

**Precision: 0%, Recall: 0%**

**Kök Neden:**

- Model bu sınıfları **hiç tahmin etmiyor**
- **Olası sebepler:**
  1. **Data quality**: Train/val/test'te bu sınıfların örnekleri bozuk olabilir
  2. **Class imbalance**: Başka sınıflar dominant
  3. **Similar gestures**: Başka bir sınıfla tamamen karıştırılıyor

**Acil İnceleme Gerekli:**

```bash
# Bu sınıfların train/val/test dağılımını kontrol et
python -c "
from config import TransformerConfig
import numpy as np
config = TransformerConfig()

for split in ['train', 'val', 'test']:
    y = np.load(f'data/processed/y_{split}.npy')
    for problem_class in ['nasil', 'okul', 'seker']:
        idx = config.CLASS_NAMES.index(problem_class)
        count = (y == idx).sum()
        print(f'{split} - {problem_class}: {count} samples')
"
```

---

## 📊 5. RASTGELE TEST ÖRNEKLERİ ANALİZİ

**20 rastgele test örneği:**

- **Accuracy**: 95.0% (19/20)
- **Ortalama Confidence**: 78.4%
- **Doğru tahminlerde confidence**: 78.1%
- **Yanlış tahminlerde confidence**: 84.5%

**İlginç Gözlem:**

- Yanlış tahmin **daha yüksek confidence** ile yapılmış! (84.5% vs 78.1%)
- Bu **overconfidence** problemi gösteriyor
- Model yanlış tahminlerinde bile çok emin

**Çözüm:**

- Label smoothing artırılabilir (ε=0.1 → 0.2)
- Temperature scaling (calibration)

---

## 💡 6. İYİLEŞTİRME ÖNERİLERİ

### 🔴 ÖNCE

LİK 1: Overfitting'i Azalt

**Mevcut:**

```python
DROPOUT = 0.2
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 20
```

**Önerilen:**

```python
DROPOUT = 0.3  # 0.2 → 0.3
LABEL_SMOOTHING = 0.2  # 0.1 → 0.2
EARLY_STOPPING_PATIENCE = 15  # 20 → 15
WEIGHT_DECAY = 5e-5  # 1e-5 → 5e-5 (daha güçlü L2)
```

**Ek Regularization:**

- **Stochastic Depth**: Transformer layer'larda dropout
- **Mixup/Cutmix**: Video augmentation
- **Gradient Noise**: Training stability

---

### 🟠 ÖNCELİK 2: Sınıf Dengesizliği

**Problem:** 3 sınıf %0 F1, bazı sınıflar dominant

**Çözüm 1: Class Weights**

```python
# train.py'de
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(
    label_smoothing=config.LABEL_SMOOTHING,
    weight=class_weights  # ← EKLE
)
```

**Çözüm 2: Focal Loss**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
  
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

---

### 🟡 ÖNCELİK 3: Başarısız Sınıfları İncele

**Yapılacaklar:**

1. **Data Quality Check:**

```bash
# nasil, okul, seker sınıflarının video kalitesini kontrol et
python scripts/inspect_failed_classes.py
```

2. **Confusion Matrix İnceleme:**

```bash
# Bu sınıflar hangi sınıflarla karıştırılıyor?
python scripts/analyze_confusion.py --classes nasil okul seker
```

3. **Feature Visualization:**

```bash
# Attention pattern'leri başarılı vs başarısız sınıflarda farklı mı?
python visualize_attention.py --compare-classes
```

---

### 🟢 ÖNCELİK 4: Model Architecture İyileştirmeleri

**Mevcut:**

```python
D_MODEL = 256
NUM_ENCODER_LAYERS = 6
NHEAD = 8
```

**Deneyebilirsin:**

1. **Daha büyük model**: 226 sınıf için D_MODEL=512 dene
2. **Layer Normalization pozisyonu**: Pre-LN vs Post-LN
3. **Relative Positional Encoding**: Mutlak yerine göreceli
4. **Multi-scale features**: Farklı temporal resolution'lar

---

## 📋 7. SONUÇ VE TAVSİYELER

### ✅ Başarılı Olan

1. **Model architecture seçimi**: Transformer video temporal modeling için uygun
2. **Training setup**: 100 epoch, warmup, cosine scheduler çalışıyor
3. **Bazı sınıflar mükemmel**: 15 sınıf %100 F1

### ❌ Sorunlu Olan

1. **Ciddi overfitting**: %12 train-val gap
2. **Generalization zayıf**: %11 val-test gap
3. **Sınıf dengesizliği**: 3 sınıf %0, bazıları %100

### 🎯 Eylem Planı

#### Kısa Vadede (1-2 gün):

1. ✅ Başarısız 3 sınıfın (nasil, okul, seker) datasını incele
2. ✅ Class weights ekle
3. ✅ Dropout ve label smoothing artır
4. ✅ Yeniden eğit ve karşılaştır

#### Orta Vadede (1 hafta):

1. Focal Loss dene
2. Data augmentation ekle (temporal jitter, spatial transforms)
3. Confusion matrix detaylı analiz
4. Hard negative mining

#### Uzun Vadede (2+ hafta):

1. Ensemble models (Transformer + LSTM + CNN)
2. Multi-task learning (gesture + facial expression)
3. Self-supervised pretraining
4. Active learning (zor örnekleri manual label)

---

## 📊 8. KULLANILAN DOSYALAR

### Training

- `train.py` - Ana eğitim scripti
- `config.py` - Hiperparametreler
- `models/transformer_model.py` - Model architecture

### Evaluation

- `evaluate.py` - Test set evaluation
- `visualize_attention.py` - Attention analysis

### Results

- `results/training_curves.png` - Loss/accuracy curves
- `results/confusion_matrix_normalized.png` - Sınıf karışıklıkları
- `results/per_class_metrics.csv` - Sınıf bazlı metrikler
- `results/evaluation_report.json` - Detaylı rapor

### Checkpoints

- `checkpoints/best_model.pth` - En iyi model (Epoch 98, Val Acc 87.87%)
- `checkpoints/last_model.pth` - Son model (Epoch 100)
- `logs/training_history.json` - Eğitim geçmişi

---

## 🚀 9. BİR SONRAKİ DENEYİ ÇALIŞTIRMA

```bash
# 1. Config'i güncelle
nano config.py
# DROPOUT = 0.3
# LABEL_SMOOTHING = 0.2
# EARLY_STOPPING_PATIENCE = 15

# 2. Class weights ekle
nano train.py
# Yukarıdaki class_weights kodunu ekle

# 3. Yeni eğitimi başlat
conda activate transformers
python train.py

# 4. Sonuçları karşılaştır
python scripts/compare_experiments.py
```

---

**Hazırlayan:** AI Assistant
**Son Güncelleme:** 9 Ekim 2025
**Versiyon:** 1.0



🔧 **Bozuk Testleri Düzeltelim!**

[3 tools called]

✅ **Düzeltilmiş Test Scriptleri:**

## 2. Training History Analizi (Düzeltilmiş)

[1 tool called]

## 5. Test Seti Inference (Düzeltilmiş)

[1 tool called]

**Şimdi Düzeltilmiş Testleri Çalıştıralım:**

[5 tools called]

🎉 **TESTLER BAŞARIYLA TAMAMLANDI!**

---

# 📊 GENEL ANALİZ - TRANSFORMER SIGN LANGUAGE MODEL

## 🎯 1. Model Performans Özeti

| Metric                  | Train   | Validation | Test                     |
| ----------------------- | ------- | ---------- | ------------------------ |
| **Accuracy**      | 100.00% | 87.87%     | **76.96%**         |
| **F1 Score**      | -       | 87.56%     | **76.19%** (macro) |
| **Train-Val Gap** | 12.13%  | -          | -                        |
| **Val-Test Gap**  | -       | -          | **10.91%** ⚠️    |

### ⚠️ Ana Problemler:

1. **Ciddi Overfitting**: Train 100%, Val 87.87%, Test 76.96%
2. **Generalization Sorunu**: Test performansı val'den %10.91 düşük
3. **Sınıf Dengesizliği**: Bazı sınıflar %100, bazıları %0 F1

---

## 📈 2. Training Curve Analizi

✅ **Kaydedildi**: `results/training_curves.png`

```
Total Epochs: 100
Best Train Acc: 100% (Epoch 78)
Best Val Acc: 87.87% (Epoch 98)
Best Val F1: 87.56% (Epoch 98)
Train-Val Gap (final): 12.29% ⚠️
```

**Gözlemler:**

- Train accuracy Epoch 78'de %100'e ulaşmış ve kalmış
- Val accuracy ~%87-88 civarında plato yapmış
- **Tipik overfitting pattern**: Model train set'i ezberliyor

---

## 🎯 3. Sınıf Bazlı Performans

### 🏆 **En İyi Performans Gösteren Sınıflar** (100% F1):

```
cuzdan, dusman, eczane, gol, hastane, hayirli_olsun, 
kolonya, komur, kopru, oda, pencere, serbest, uzak, 
yemek_pisirmek, yorgun
```

**Toplam: 15 sınıf**

### ⭐ **Çok İyi Performans** (90-99% F1):

```
ozur_dilemek (97.14%), benzin (97.14%), kopek (97.14%),
bayram (96.97%), bekar (96.97%), devirmek (96.97%)
... (26 sınıf daha)
```

### ⚠️ **Orta Performans** (50-70% F1):

```
bardak (34.15%), ataturk (24.56%), ilac (21.05%)
... (yaklaşık 60 sınıf)
```

### ❌ **Başarısız Sınıflar** (0-30% F1):

```
- nasil:  0.00% ❌ (Hiç tahmin edilememiş!)
- okul:   0.00% ❌ (Hiç tahmin edilememiş!)
- seker:  0.00% ❌ (Hiç tahmin edilememiş!)
- ilac:   21.05%
- dakika: 30.00%
- iyi:    30.00%
- kotu:   30.00%
- olur:   30.00%
- yapmak: 30.00%
```

---

## 🔍 4. Rastgele Test Örnekleri Sonucu

```
📊 20 örnekte Accuracy: 95.0% (19/20 doğru)
📈 Ortalama Confidence: 78.4%
```

**İlginç Gözlem:**

- Rastgele örneklerde %95 accuracy
- Genel test accuracy %76.96
- **Muhtemel neden**: Bazı sınıflar çok iyi, bazıları çok kötü → average pooling düşürüyor

---

## 🧠 5. Overfitting Analizi

### Son 10 Epoch:

```
Epoch 90: Train=100%, Val=87.71%, Gap=12.29%
Epoch 91: Train=100%, Val=87.51%, Gap=12.49%
...
Epoch 98: Train=100%, Val=87.85%, Gap=12.15% ← BEST
Epoch 99: Train=100%, Val=87.71%, Gap=12.29%
```

**Model davranışı:**

- ✅ Train set'i **tamamen ezberlemiş** (100%)
- ⚠️ Val set'te **genelleme sorunları** var (%87)
- ❌ Test set'te **daha da kötü** (%76)

---

## 💡 6. Problemli Sınıflar - Confusion Analizi

[2 tools called]

Mükemmel! Şimdi detaylı analiz raporu oluşturalım:

[3 tools called]

🎯 **HARIKA BULGU! Başarısız Sınıfların Kök Nedeni Bulundu!**

[1 tool called]

---

# 🎉 TÜMÜ TAMAMLANDI! İŞTE KAPSAMLI ANALİZ

## 📁 Oluşturulan Dosyalar

### ✅ 1. Düzeltilmiş Test Scriptleri

- `scripts/plot_training_history.py` ✅
- `scripts/test_random_samples.py` ✅
- `scripts/inspect_failed_classes.py` ✅

### ✅ 2. Analiz Raporları

- `FINAL_ANALYSIS.md` - **14 sayfa detaylı analiz** ✅
- `ANALYSIS_SUMMARY.md` - **Hızlı özet** ✅

### ✅ 3. Sonuç Görselleri

- `results/training_curves.png` ✅
- `results/confusion_matrix_normalized.png` ✅
- `results/per_class_metrics.csv` ✅
- `results/attention/` (67 görsel) ✅

---

## 🎯 ANAHTAR BULGULAR

### 1. **PERFORMANS**

```
Train Accuracy:      100.00%  ← Ezberlenmiş!
Validation Accuracy:  87.87%  (↓ 12.13%)
Test Accuracy:        76.96%  (↓ 10.91%)
─────────────────────────────────────────
TOPLAM DÜŞÜŞ:         23.04%  ⚠️ OVERFITTING!
```

### 2. **BAŞARILI SINIFLAR** 🏆

- **15 sınıf 100% F1**: `cuzdan, dusman, eczane, gol, hastane, hayirli_olsun, kolonya, komur, kopru, oda, pencere, salca, serbest, uzak, yemek_pisirmek, yorgun`
- **41 sınıf 90-99% F1**: Toplam 56/226 sınıf mükemmel!

### 3. **BAŞARISIZ SINIFLAR** ❌

#### 🔴 Hiç Tahmin Edilemeyenler (0% F1):

- `nasil` (16 test sample)
- `okul` (17 test sample)
- `seker` (11 test sample)

#### 🚨 **KRİTİK BULGU:**

| Sınıf         | Avg Zero Frames     | Percentage           |
| --------------- | ------------------- | -------------------- |
| **nasil** | **31.2 / 81** | **38.5%** ⚠️ |
| **okul**  | **26.4 / 81** | **32.6%** ⚠️ |
| **seker** | **22.2 / 81** | **27.4%** ⚠️ |

**Normal:** ~15-20 zero frames (~20%)
**Bu sınıflar:** ~25-31 zero frames (~30-40%)

**TEŞHİS:** Videolar çok kısa veya keypoint extraction başarısız!

### 4. **DÜŞÜK PERFORMANS** (<30% F1):

```
ilac (21%), dakika (30%), iyi (30%), kotu (30%), 
olur (30%), yapmak (30%), ataturk (24%), oruc (27%)
```

---

## 💡 ÇÖZÜM ÖNERİLERİ

### 🔴 **ACİL (BUGÜN)**

#### 1. **Config Güncellemesi**

```python
# config.py
DROPOUT = 0.3              # 0.2 → 0.3
LABEL_SMOOTHING = 0.2      # 0.1 → 0.2
WEIGHT_DECAY = 5e-5        # 1e-5 → 5e-5
EARLY_STOPPING_PATIENCE = 15  # 20 → 15
```

#### 2. **Class Weights Ekle**

```python
# train.py'de criterion'dan önce:
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(
    label_smoothing=config.LABEL_SMOOTHING,
    weight=class_weights  # ← EKLE
)
```

#### 3. **Başarısız Sınıfları İncele**

```bash
# Orijinal videoları kontrol et
ls -lh Data/Train\ Data/train/nasil/
ls -lh Data/Train\ Data/train/okul/
ls -lh Data/Train\ Data/train/seker/

# Video sürelerini ölç
python scripts/check_video_durations.py --classes nasil okul seker
```

---

### 🟠 **KISA VADE (1-2 GÜN)**

#### 4. **Data Augmentation**

```python
# Temporal augmentation
def temporal_jitter(x, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift, axis=0)

# Spatial noise
def add_noise(x, noise_level=0.01):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise
```

#### 5. **Focal Loss**

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
  
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1-pt)**self.gamma * ce_loss).mean()
```

---

## 📊 BEKLENEN İYİLEŞTİRMELER

| Metric                    | Şimdi | Hedef            | İyileştirme |
| ------------------------- | ------ | ---------------- | ------------- |
| **Test Acc**        | 76.96% | **82-85%** | +5-8%         |
| **0% F1 Sınıf**   | 3      | **0-1**    | -2-3 sınıf  |
| **<30% F1 Sınıf** | 12     | **<5**     | -7 sınıf    |
| **Train-Val Gap**   | 12.13% | **<8%**    | -4%           |
| **Val-Test Gap**    | 10.91% | **<5%**    | -6%           |

---

## 🚀 BİR SONRAKİ ADIMLARI ÇALIŞTIRMA

```bash
cd transformer-signlang
conda activate transformers

# 1. Training history görselleştir
python scripts/plot_training_history.py

# 2. Rastgele test örnekleri
python scripts/test_random_samples.py

# 3. Başarısız sınıfları incele
python scripts/inspect_failed_classes.py

# 4. Detaylı raporları oku
cat ANALYSIS_SUMMARY.md
cat FINAL_ANALYSIS.md

# 5. Config'i güncelle ve yeni eğitim başlat
nano config.py
python train.py
```

---

## 📋 TÜM SONUÇLAR

### Training

- ✅ **Duration:** 2:53:58
- ✅ **Epochs:** 100
- ✅ **Best Val Acc:** 87.87% (Epoch 98)
- ✅ **Files:** `checkpoints/best_model.pth`, `logs/training_history.json`

### Evaluation

- ✅ **Test Acc:** 76.96%
- ✅ **Test F1 (Macro):** 76.19%
- ✅ **Precision (Macro):** 82.31%
- ✅ **Recall (Macro):** 76.84%

### Visualizations

- ✅ **Training Curves:** `results/training_curves.png`
- ✅ **Confusion Matrix:** `results/confusion_matrix_normalized.png`
- ✅ **Per-Class Metrics:** `results/per_class_metrics.csv`
- ✅ **Attention Maps:** `results/attention/` (67 files)

---

## 🎯 ÖZETİN ÖZETİ

**✅ NE BAŞARILI?**

- 56/226 sınıf mükemmel/çok iyi performans
- Model architecture uygun
- Training pipeline çalışıyor

**❌ NE SORUNLU?**

- %23 overfitting (train→test drop)
- 3 sınıf %0 F1 (data quality!)
- 12 sınıf <%30 F1

**💡 NASIL DÜZELTİRİZ?**

- Dropout/regularization artır
- Class weights/focal loss ekle
- Başarısız sınıfların datasını düzelt

**🎯 HEDEF:**

- Test Acc: 82-85% (şimdi 76.96%)
- 0% sınıf: 0-1 (şimdi 3)
- Overfitting: <8% gap (şimdi 12.13%)

---

**Tüm analiz raporları hazır! Şimdi iyileştirmelere başlayabilirsiniz.** 🚀

Hangi adımla başlamak istersiniz?

1. Config güncelleyip yeni eğitim mi?
2. Başarısız sınıfların videolarını mı incelemek?
3. Class weights ekleyip denemek mi?
