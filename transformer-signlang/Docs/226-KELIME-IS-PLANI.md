# 🚀 226 Kelime (TÜM VERİ) İşaret Dili Tanıma - İş Planı

**Tarih:** 7 Ekim 2025  
**Durum:** 📋 PLANLAMA AŞAMASI  
**Önceki Başarı:** 10 kelime → %91.57 accuracy 🎉

---

## 📊 HEDEF: TÜM AUTSL VERİ SETİ

### Veri Seti Özeti

| Özellik | Değer |
|---------|-------|
| **Toplam Sınıf** | 226 kelime (Class ID: 0-225) |
| **Train Videos** | 28,142 video (31 signer) |
| **Validation Videos** | 4,418 video (6 signer) |
| **Test Videos** | 3,742 video (6 signer) |
| **TOPLAM** | **36,302 video** |
| **Modality** | RGB + Depth (biz sadece RGB kullanacağız) |
| **Resolution** | 512x512 |
| **Signer Bağımsız** | ✅ Evet (farklı signer'lar farklı setlerde) |

### 10 Kelime ile Karşılaştırma

| Metrik | 10 Kelime | 226 Kelime | Artış |
|--------|-----------|------------|-------|
| Sınıf sayısı | 10 | 226 | **22.6x** |
| Toplam video | 1,607 | 36,302 | **22.6x** |
| Train video | 1,243 | 28,142 | **22.6x** |
| Val video | 198 | 4,418 | **22.3x** |
| Test video | 166 | 3,742 | **22.5x** |

**Ölçek:** Veri seti **22.6 kat daha büyük!** 🚀

---

## ⏱️ BEKLENEN SÜRELER VE KAYNAKLAR

### Tahmini İşlem Süreleri (M3 Mac)

| Adım | 10 Kelime | 226 Kelime | Artış |
|------|-----------|------------|-------|
| **Keypoint Extraction** | 2-3 saat | **30-50 saat** | ~20x |
| **Normalization** | 5-10 dk | **2-3 saat** | ~20x |
| **Training (100 epoch)** | 2-3 saat | **50-80 saat** | ~25x |
| **Evaluation** | 2-5 dk | **15-30 dk** | ~10x |
| **TOPLAM** | ~5-7 saat | **~85-135 saat** (~3.5-5.5 gün) | ~20x |

⚠️ **DİKKAT:** Training çok uzun! Stratejik planlama gerekli.

### Disk Alanı Gereksinimleri

| Dosya Tipi | 10 Kelime | 226 Kelime | Açıklama |
|------------|-----------|------------|----------|
| **Keypoints (.npy)** | ~80 MB | **~1.8 GB** | 36,302 × 50 KB |
| **Processed Data** | ~500 MB | **~11 GB** | Normalized + padded |
| **Model Checkpoints** | ~64 MB | **~350 MB** | Best + Last (daha büyük) |
| **Results** | ~10 MB | **~50 MB** | Evaluation + viz |
| **TOPLAM** | ~1 GB | **~13-15 GB** | - |

**Mevcut Disk:** 193 GB boş ✅ Yeterli!

---

## 🎯 BEKLENEN PERFORMANS

### Optimistik Senaryo (İyi Giderse)
```
Test Accuracy:     75-82%
F1-Score (Macro):  72-80%
Val Accuracy:      78-85%
```
**Yorum:** 226 sınıf çok zor! %75+ çok iyi sayılır.

### Gerçekçi Senaryo (Muhtemelen)
```
Test Accuracy:     68-75%
F1-Score (Macro):  65-73%
Val Accuracy:      70-78%
```
**Yorum:** State-of-the-art modeller ~%75-85 arası.

### Kötü Senaryo (Sorunlar Olursa)
```
Test Accuracy:     <65%
F1-Score (Macro):  <60%
```
**Yorum:** Ciddi sorunlar var, model revizyon gerekli.

**Benchmark:**
- AUTSL Paper (2020): ~76-83% accuracy (farklı modeller)
- Transformer-based: ~80-85% (son yıl çalışmaları)

**Bizim Hedef:** %75-80 accuracy (production için yeterli)

---

## 📋 DETAYLI İŞ PLANI

### 🛑 **ADIM 0: ÖN ANALİZ VE KARAR (1 gün)**

#### 0.1. Risk Değerlendirmesi

**Riskler:**
1. ⚠️ **Keypoint extraction: 30-50 saat** - Çok uzun!
2. ⚠️ **Training: 50-80 saat** - Çok uzun!
3. ⚠️ **Memory: 28K video** - RAM sorunları olabilir
4. ⚠️ **Performance: 226 sınıf** - Düşük accuracy riski
5. ⚠️ **Overfitting:** Bazı sınıflarda az veri olabilir

**Çözümler:**
1. ✅ Keypoint extraction: Batch processing, resume capability
2. ✅ Training: Learning rate tuning, early stopping
3. ✅ Memory: Batch loading, data generators
4. ✅ Performance: Transfer learning, data augmentation
5. ✅ Overfitting: Stronger regularization (dropout=0.2)

#### 0.2. Alternatif Strateji: Aşamalı Genişleme

**Seçenek A: Direkt 226 Kelime** (Riskli ama Hızlı)
-장점: Tek seferde bitir
- Dezavantaj: Başarısızlık riski yüksek

**Seçenek B: 50 Kelime → 100 Kelime → 226 Kelime** (Güvenli)
- 장점: Her adımda öğren, iyileştir
- Dezavantaj: 3x daha uzun sürer

**Seçenek C: 25-50 Kelime → 226 Kelime** (ÖNERİLEN) ⭐
-장점: Orta risk, makul süre
- 1. Adım: 25-50 kelime (~2-3 gün)
- 2. Adım: Başarılıysa 226 kelime (~5 gün)
- TOPLAM: ~7-8 gün

**KARAR NOKTASI:** Hangi strateji seçilecek?

---

### ✅ **ADIM 1: CONFIG GÜNCELLEME (15 dakika)**

#### 1.1. config.py Değişiklikleri

```python
# ÖNCE (10 kelime):
NUM_CLASSES = 10
TARGET_CLASS_IDS = [1, 2, 5, 14, 20, 30, 65, 86, 100, 196]
CLASS_NAMES = ['acele', 'acikmak', 'agac', 'anne', 'baba', 
               'ben', 'evet', 'hayir', 'iyi', 'tesekkur']

# SONRA (226 kelime):
NUM_CLASSES = 226
TARGET_CLASS_IDS = list(range(0, 226))  # [0, 1, 2, ..., 225]
CLASS_NAMES = [...226 kelime...]  # SignList_ClassId_TR_EN.csv'den yükle
```

#### 1.2. Hiperparametre Ayarları (226 Sınıf İçin)

**Model Büyütme (Önerilen):**
```python
# Daha büyük model gerekli (226 sınıf için)
D_MODEL = 512              # 256 → 512 (2x)
NUM_ENCODER_LAYERS = 8     # 6 → 8 (daha derin)
NHEAD = 16                 # 8 → 16 (daha fazla attention)
DIM_FEEDFORWARD = 2048     # 1024 → 2048
```

**Training Ayarları:**
```python
BATCH_SIZE = 16            # 32 → 16 (memory için)
LEARNING_RATE = 5e-5       # 1e-4 → 5e-5 (daha küçük)
DROPOUT = 0.2              # 0.1 → 0.2 (daha güçlü regularization)
LABEL_SMOOTHING = 0.15     # 0.1 → 0.15
EARLY_STOPPING_PATIENCE = 20  # 10 → 20 (daha sabırlı)
WARMUP_EPOCHS = 15         # 10 → 15
```

**Alternatif: Mevcut Model Koru (Daha Hızlı Eğitim)**
```python
# Aynı architecture ama 226 sınıf
D_MODEL = 256
NUM_ENCODER_LAYERS = 6
NHEAD = 8
# Sadece NUM_CLASSES değiştir
```

**KARAR:** Hangi model boyutu? (Büyük vs Mevcut)

---

### ✅ **ADIM 2: VERİ HAZIRLAMA (30-50 saat!)**

#### 2.1. Class Names Yükleme (Python script)

```python
# scripts/load_class_names.py (yeni)
import pandas as pd

def load_all_classes():
    """SignList_ClassId_TR_EN.csv'den tüm sınıfları yükle"""
    df = pd.read_csv('../Data/Class ID/SignList_ClassId_TR_EN.csv')
    # ClassId sırasına göre sırala (0-225)
    df = df.sort_values('ClassId')
    class_names = df['TR'].tolist()  # Türkçe isimler
    return class_names

# config.py'da kullan
CLASS_NAMES = load_all_classes()
```

#### 2.2. Video Seçimi (01_select_videos.py)

**Değişiklik:** TARGET_CLASS_IDS = list(range(226))

**Beklenen Çıktı:**
```
Train:      ~28,142 video
Validation: ~4,418 video
Test:       ~3,742 video
TOPLAM:     ~36,302 video
```

**Süre:** ~2 dakika

#### 2.3. Keypoint Extraction (02_extract_keypoints.py)

⚠️ **EN UZUN ADIM: 30-50 SAAT!**

**Stratejiler:**

**A) Paralel İşleme (Önerilen)**
```python
# scripts/02_extract_keypoints_parallel.py (yeni)
# Multiprocessing kullan - 4-8 core
# 4 core: ~40 saat → ~12 saat
# 8 core: ~40 saat → ~8 saat
```

**B) Batch Processing (Güvenli)**
```python
# Her 5000 videoda bir checkpoint kaydet
# Kesinti olursa kaldığı yerden devam et
```

**C) Resume Capability**
```python
# Hangi videolar işlendi kontrol et
# Sadece eksikleri işle
```

**Beklenen Çıktı:**
```
data/keypoints/
  └── 36,302 × .npy dosyası (~1.8 GB)
```

**KARAR:** Paralel kullan mı? (Hızlı ama riskli)

#### 2.4. Normalization (03_normalize_data.py)

**Süre:** 2-3 saat

**Beklenen Çıktı:**
```
data/processed/
  ├── X_train.npy  (28142, max_len, 258) ~10 GB
  ├── y_train.npy  (28142,)
  ├── X_val.npy    (4418, max_len, 258)  ~1.5 GB
  ├── y_val.npy    (4418,)
  ├── X_test.npy   (3742, max_len, 258)  ~1.3 GB
  └── y_test.npy   (3742,)
data/scaler.pkl
```

**Memory Sorun:** 28K video belleğe sığmayabilir!

**Çözüm:**
```python
# Batch normalization - 5000'er 5000'er işle
# Her batch için scaler.partial_fit() kullan
```

---

### ✅ **ADIM 3: SETUP VALIDATION (5 dakika)**

```bash
python validate_setup.py
```

**Kontroller:**
- ✅ 226 sınıf doğru mu?
- ✅ Class mapping tutarlı mı?
- ✅ Veri dosyaları var mı?
- ✅ Model oluşturuluyor mu?

---

### ✅ **ADIM 4: MODEL TRAINING (50-80 SAAT!)**

⚠️ **EN RİSKLİ ADIM!**

#### 4.1. Training Stratejileri

**Strateji 1: Direkt Eğitim (Basit)**
```bash
python train.py
# 50-80 saat bekle...
```

**Strateji 2: Transfer Learning (ÖNERİLEN)** ⭐
```python
# 10-kelime modelinden başla
# Son layer'ı değiştir (10 → 226 class)
# Fine-tune et

# train.py'da:
if resume_from_10_class:
    # Load 10-class weights
    checkpoint = torch.load('checkpoints/10-kelime-best.pth')
    # Sadece son layer hariç yükle
    model.load_partial_weights(checkpoint)
    # 226-class için yeni classifier
    model.classifier = create_new_classifier(226)
```

**Beklenen Süre (Transfer Learning):** ~30-40 saat (normal: 50-80 saat)

**Strateji 3: Progressive Training**
```python
# İlk 50 epoch: Freeze encoder, sadece classifier eğit
# Sonraki 50 epoch: Tüm model eğit
```

#### 4.2. Training İzleme

```bash
# Başka terminal'de
watch -n 30 'tail -20 logs/training_history.json'

# Tensorboard (opsiyonel)
tensorboard --logdir logs/
```

#### 4.3. Early Stopping

**Kritik:** 226 sınıf için early stopping aggressive olabilir!

```python
EARLY_STOPPING_PATIENCE = 20  # 10 → 20
# Çünkü model yavaş öğrenecek (226 sınıf)
```

---

### ✅ **ADIM 5: EVALUATION (15-30 dakika)**

```bash
python evaluate.py
```

**Beklenen Çıktılar:**
```
results/
├── evaluation_report.json
├── confusion_matrix_226x226.png  (çok büyük!)
├── per_class_metrics.csv  (226 satır)
├── test_predictions.csv   (3,742 satır)
└── ...
```

**226x226 Confusion Matrix:** Çok büyük, yorumlaması zor!

**Alternatif Analiz:**
```python
# Top-10 accuracy yerine top-5 accuracy
# En zor 20 sınıf analizi
# Benzer sınıf grupları (el, yüz, vücut hareketleri)
```

---

### ✅ **ADIM 6: ATTENTION VISUALIZATION (Opsiyonel, 30-60 dk)**

```bash
python visualize_attention.py --num_samples 10
```

**Süre:** 30-60 dakika (226 sınıf için yavaş)

---

### ✅ **ADIM 7: RAPOR OLUŞTURMA (1 saat)**

```
226-kelime-model-rapor.md
```

**İçerik:**
- Genel performans (accuracy, F1)
- Top-10 en başarılı sınıflar
- Bottom-10 en zor sınıflar
- Sınıf grupları analizi (benzer işaretler)
- 10-kelime ile karşılaştırma
- Production hazırlık değerlendirmesi

---

## 🎯 BAŞARI KRİTERLERİ

### Minimum Kabul Edilebilir (Production İçin)

| Metrik | Minimum | İdeal | Mükemmel |
|--------|---------|-------|----------|
| **Test Accuracy** | %65 | %72 | %78+ |
| **F1-Score (Macro)** | %62 | %70 | %75+ |
| **Val Accuracy** | %68 | %75 | %80+ |
| **Top-5 Accuracy** | %85 | %90 | %95+ |

**Özel Kriterler:**
- ✅ En az %80 sınıfın F1 > %60
- ✅ Hiçbir sınıf F1 < %30
- ✅ Val-Test gap < %5

---

## ⚠️ RİSK YÖNETİMİ VE SORUN GİDERME

### Risk 1: Keypoint Extraction Çok Uzun (30-50 saat)

**Önlem:**
- ✅ Paralel işleme (multiprocessing)
- ✅ Resume capability
- ✅ Batch checkpoint (her 5000 video)

**Sorun Çıkarsa:**
- Plan B: Daha az video kullan (her sınıftan ilk 100)
- Plan C: Cloud GPU kullan (Google Colab Pro)

### Risk 2: Training Çok Uzun (50-80 saat)

**Önlem:**
- ✅ Transfer learning (10-kelime modelinden)
- ✅ Smaller model (mevcut D_MODEL=256 kalsın)
- ✅ Early stopping

**Sorun Çıkarsa:**
- Plan B: Daha küçük model (D_MODEL=128)
- Plan C: Fewer epochs (50 epoch max)

### Risk 3: Memory Yetersiz (28K video)

**Önlem:**
- ✅ Batch loading
- ✅ Data generator kullan
- ✅ Smaller batch size (16)

**Sorun Çıkarsa:**
- Plan B: Virtual memory kullan
- Plan C: Disk'ten streaming read

### Risk 4: Düşük Performans (<65%)

**Önlem:**
- ✅ Transfer learning
- ✅ Data augmentation
- ✅ Stronger regularization

**Sorun Çıkarsa:**
- Plan B: Ensemble model (3-5 model)
- Plan C: Sınıf grupları (benzer işaretler birleştir)

### Risk 5: Overfitting

**Önlem:**
- ✅ DROPOUT = 0.2
- ✅ LABEL_SMOOTHING = 0.15
- ✅ Data augmentation
- ✅ Early stopping

---

## 💰 MALIYET ANALİZİ

### Zaman Maliyeti (M3 Mac)

| Adım | Süre | İnsan Müdahalesi |
|------|------|------------------|
| Planlama | 4 saat | %100 |
| Config | 0.5 saat | %100 |
| Video seçimi | 0.1 saat | %10 |
| Keypoint extract | 30-50 saat | %5 (monitoring) |
| Normalization | 2-3 saat | %5 |
| Training | 50-80 saat | %5 |
| Evaluation | 0.5 saat | %50 |
| Rapor | 2 saat | %100 |
| **TOPLAM** | **~90-140 saat** | **~8-10 saat aktif** |

**Takvim:** ~4-6 gün (bilgisayar çalışıyor, sen başka iş yapıyorsun)

### Disk Maliyeti

- Gerekli: ~15 GB
- Mevcut: 193 GB
- ✅ Yeterli!

---

## 🚀 ÖNERİLEN STRATEJİ

### Seçenek 1: Direkt 226 Kelime (Agresif)

**장점:**
- Tek seferde bitir
- En hızlı yol

**Dezavantaj:**
- Yüksek risk (başarısızlık olabilir)
- Uzun bekleme (4-6 gün)

**Kime Önerilir:** Sabırlı ve risk alabilenler

---

### Seçenek 2: 50 Kelime → 226 Kelime (ÖNERİLEN) ⭐

**Adımlar:**
1. **50 Kelime Pilot (2-3 gün)**
   - En sık kullanılan 50 kelime
   - Hızlı eğitim (~10 saat)
   - Sorunları erken tespit et
   
2. **Başarılıysa → 226 Kelime (4-6 gün)**
   - Transfer learning ile başla
   - Güvenle devam et

**장점:**
- Risk azalır
- Erken feedback
- Öğrenerek ilerle

**Dezavantaj:**
- Biraz daha uzun (toplam 6-9 gün)

**Kime Önerilir:** Çoğu kişi (güvenli yaklaşım)

---

### Seçenek 3: 25 Kelime → 50 Kelime → 226 Kelime (Çok Güvenli)

**Adımlar:**
1. 25 kelime (1-2 gün)
2. 50 kelime (2-3 gün)  
3. 226 kelime (4-6 gün)
4. TOPLAM: 7-11 gün

**장점:**
- En güvenli
- Her adımda iyileştirme

**Dezavantaj:**
- En uzun süre

**Kime Önerilir:** İlk kez büyük ölçekli ML yapanlar

---

## ✅ KARAR MATRISI

| Kriter | Direkt 226 | 50→226 (Önerilen) | 25→50→226 |
|--------|------------|-------------------|-----------|
| **Süre** | 4-6 gün | 6-9 gün | 7-11 gün |
| **Risk** | 🔴 Yüksek | 🟡 Orta | 🟢 Düşük |
| **Başarı Şansı** | %60 | %80 | %90 |
| **Öğrenme** | Az | Orta | Çok |
| **Esneklik** | Yok | Var | Çok Var |

---

## 📋 SON KONTROL LİSTESİ

Başlamadan önce:

- [ ] Strateji seçildi (Direkt / 50→226 / 25→50→226)
- [ ] Disk alanı yeterli (193 GB > 15 GB) ✅
- [ ] Zaman planlaması yapıldı (4-11 gün)
- [ ] Yedekleme: 10-kelime modeli yedeklendi
- [ ] Config stratejisi: Büyük model vs Mevcut model
- [ ] Transfer learning kullanılacak mı?
- [ ] Paralel keypoint extraction kullanılacak mı?
- [ ] Beklenen performans belirlendi (%65-78)

---

## 🎯 SONUÇ VE TAVSİYE

**10 Kelime Başarısı:** %91.57 accuracy 🎉

**226 Kelime Hedefi:** %70-78 accuracy (gerçekçi)

**TAVSİYE EDİLEN YÖNTEM:**

### 📌 **2-Aşamalı Yaklaşım (50→226)**

**Neden:**
1. ✅ Risk/fayda dengesi en iyi
2. ✅ Erken sorun tespiti
3. ✅ 50 kelime ile production başlayabilirsin
4. ✅ 226'ya geçiş daha güvenli

**Timeline:**
```
Gün 1-2:   50 kelime hazırlık + eğitim
Gün 3:     50 kelime değerlendirme + karar
Gün 4-8:   226 kelime (başarılıysa)
Gün 9:     Final rapor + deployment planı
```

**Hazırsan başlayalım! 🚀**

---

**Sonraki Adım:** Hangi stratejiyi seçiyorsun?
1. Direkt 226 kelime (agresif)
2. 50 → 226 kelime (önerilen) ⭐
3. 25 → 50 → 226 kelime (güvenli)

