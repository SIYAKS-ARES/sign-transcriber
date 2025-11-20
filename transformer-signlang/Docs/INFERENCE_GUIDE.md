# Test Video Inference Rehberi

## 🎯 Amaç

`inference_test_videos.py` scripti, eğitilmiş Transformer modelini test videoları üzerinde çalıştırarak:
- Gerçek zamanlı tahmin görselleştirmesi yapar
- Tahmin sonuçlarını kaydeder
- Detaylı performans analizi sağlar

## 📋 Gereksinimler

### 1. Eğitilmiş Model
```
checkpoints/best_model.pth  # Eğitimden gelen model
```

### 2. Normalizasyon Scaler
```
data/scaler.pkl  # Eğitim sırasında oluşturulan scaler
```

### 3. Test Video Listesi
```
data/selected_videos_test.csv  # Test videoları ve etiketleri
```

### 4. Test Videoları
```
../Data/Test Data & Valid, Labels/test/*.mp4
```

## 🚀 Kullanım

### Temel Kullanım

```bash
# Transformer klasörüne git
cd transformer-signlang

# Scripti çalıştır
python inference_test_videos.py
```

### İnteraktif Seçenekler

Script çalıştırıldığında size sorar:
```
▶️  Videoları göstermek ister misiniz? (y/n) [y]:
```

- **`y` veya Enter**: Videoları oynatır, tahminleri gösterir
- **`n`**: Videoları göstermeden sadece tahmin yapar ve kaydeder

### Video Oynatma Kontrolleri

Video oynatılırken:
- **`q`**: Çıkış (tüm işlemi durdur)
- **`n`**: Sonraki videoya geç
- **`p`**: Duraklat/Devam et

## 📊 Çıktılar

### 1. CSV Formatı
**Dosya:** `results/test_predictions.csv`

| video_id | num_frames | true_class_id | true_class_name | pred_class_id | pred_class_name | confidence | is_correct |
|----------|------------|---------------|-----------------|---------------|-----------------|------------|------------|
| signer6_sample8 | 85 | 5 | agac | 5 | agac | 0.98 | True |
| signer6_sample11 | 72 | 2 | acikmak | 2 | acikmak | 0.95 | True |

**Kolonlar:**
- `video_id`: Video tanımlayıcısı
- `num_frames`: Videodaki frame sayısı
- `true_class_id`: Gerçek sınıf ID (1=acele, 2=acikmak, 5=agac)
- `true_class_name`: Gerçek sınıf adı
- `pred_class_id`: Tahmin edilen sınıf ID
- `pred_class_name`: Tahmin edilen sınıf adı
- `confidence`: Tahmin güveni (0-1 arası)
- `is_correct`: Tahmin doğru mu? (True/False)

### 2. JSON Formatı
**Dosya:** `results/test_predictions.json`

```json
[
    {
        "video_id": "signer6_sample8",
        "video_path": "../Data/Test Data & Valid, Labels/test/signer6_sample8_color.mp4",
        "num_frames": 85,
        "true_class_id": 5,
        "true_class_name": "agac",
        "pred_class_id": 5,
        "pred_class_name": "agac",
        "confidence": 0.9823,
        "is_correct": true,
        "probabilities": {
            "acele": 0.0012,
            "acikmak": 0.0165,
            "agac": 0.9823
        }
    }
]
```

**Ek Bilgiler:**
- Tüm sınıflar için olasılık dağılımı
- Video dosya yolu
- Detaylı tahmin bilgileri

### 3. Konsol Çıktısı

Script çalışırken konsola detaylı bilgi verir:

```
================================================================================
🎬 TRANSFORMER TEST VIDEO INFERENCE
================================================================================

🖥️  Device: CUDA (NVIDIA GeForce RTX 3060)

📂 Model ve scaler yükleniyor...
   ✅ Model yüklendi!
      - Epoch: 13
      - Val Acc: 1.0000
      - Val F1: 1.0000

📊 Test Seti:
   - Toplam video: 50

📊 Sınıf Dağılımı:
   - acele (ClassId 1): 16 video
   - acikmak (ClassId 2): 17 video
   - agac (ClassId 5): 17 video

================================================================================
⌨️  KONTROLLER:
   - 'q': Çıkış
   - 'n': Sonraki video
   - 'p': Duraklat/Devam
================================================================================

▶️  Videoları göstermek ister misiniz? (y/n) [y]: y

================================================================================
🎯 TEST BAŞLIYOR
================================================================================

[1/50] signer6_sample8 (ClassId: 5)
   ✅ DOĞRU: agac (98.23%) | Gerçek: agac

[2/50] signer6_sample11 (ClassId: 2)
   ✅ DOĞRU: acikmak (95.12%) | Gerçek: acikmak

...

================================================================================
💾 SONUÇLAR KAYDEDİLİYOR
================================================================================

   ✅ CSV kaydedildi: results/test_predictions.csv
   ✅ JSON kaydedildi: results/test_predictions.json

================================================================================
📊 ÖZET İSTATİSTİKLER
================================================================================

📈 Genel Performans:
   - Toplam video: 50
   - Doğru tahmin: 45
   - Yanlış tahmin: 5
   - Accuracy: 90.00%

📊 Sınıf Bazlı Performans:
   - acele     : 11/16 (68.75%)
   - acikmak   : 17/17 (100.00%)
   - agac      : 17/17 (100.00%)

🎯 Confidence İstatistikleri:
   - Ortalama: 92.45%
   - Doğru tahminler: 95.23%
   - Yanlış tahminler: 78.42%

📋 Karışıklık Özeti (Yanlış Tahminler):
   - acele     → agac       (82.15%) [signer34_sample126]
   - acele     → agac       (79.34%) [signer34_sample405]
   - acele     → agac       (81.56%) [signer34_sample412]
   - acele     → agac       (83.92%) [signer6_sample162]
   - acele     → agac       (80.47%) [signer6_sample521]

================================================================================
✅ TEST TAMAMLANDI
================================================================================

📁 Sonuçlar kaydedildi:
   - results/test_predictions.csv
   - results/test_predictions.json
```

## 📈 Video Görselleştirme

Video oynatılırken ekranda gösterilenler:

```
┌──────────────────────────────────────────────────┐
│ Video: signer6_sample8                           │
│ Tahmin: agac (98%)                               │
│ Gercek: agac                            DOGRU    │
│                                                  │
│                                                  │
│          [MediaPipe Skeleton Overlay]            │
│                                                  │
│                                                  │
│ [████████████████████░░░░░] Progress             │
└──────────────────────────────────────────────────┘
```

**Görsel Elementler:**
- ✅ Video ID
- ✅ Model tahmini + confidence
- ✅ Gerçek etiket
- ✅ Durum (DOĞRU/YANLIŞ - yeşil/kırmızı)
- ✅ MediaPipe iskelet overlay
- ✅ Progress bar

## 🔍 Teknik Detaylar

### Veri İşleme Pipeline

1. **Keypoint Extraction** (MediaPipe)
   - Pose: 33 × 3 = 99 boyut
   - Face (key points): 11 × 3 = 33 boyut
   - Hands: 2 × 21 × 3 = 126 boyut
   - **Toplam:** 258 boyut

2. **Normalization** (StandardScaler)
   - Training'de fit edilen scaler kullanılır
   - Z-score normalization

3. **Sequence Processing**
   - Max length: 200 frame (config'den)
   - Padding: Başa sıfır ekleme (kısa videolar)
   - Truncation: Son 200 frame (uzun videolar)

4. **Model Inference**
   - Transformer encoder (6 layer, 8 head, 256 d_model)
   - Batch size: 1 (video başına)
   - Softmax output → probabilities

### Performans

**Hız:**
- CUDA GPU: ~5-10 FPS (real-time)
- MPS (Apple): ~3-7 FPS
- CPU: ~1-3 FPS

**Bellek:**
- GPU VRAM: ~500 MB (model)
- RAM: ~2 GB (video buffering)

## 🛠️ Sorun Giderme

### Model bulunamadı hatası

```
❌ HATA: Model checkpoint bulunamadı: checkpoints/best_model.pth
```

**Çözüm:** Önce modeli eğitin:
```bash
python train.py
```

### Scaler bulunamadı hatası

```
❌ HATA: Scaler bulunamadı: data/scaler.pkl
```

**Çözüm:** Veri hazırlama pipeline'ını çalıştırın:
```bash
python scripts/01_select_videos.py
python scripts/02_extract_keypoints.py
python scripts/03_normalize_data.py
```

### Video gösterilmiyor

**Neden:** 
- Headless sunucu (GUI yok)
- `show_video=False` seçildi

**Çözüm:**
- Local makinede çalıştırın veya
- `n` seçeneğiyle sadece tahmin yapın (görselleştirme olmadan)

### MediaPipe hatası

```
MediaPipe initialization failed
```

**Çözüm:**
```bash
pip install --upgrade mediapipe opencv-python
```

## 📝 Notlar

### Class ID Mapping

Model 3 sınıfı öğrendi:
- **ClassId 1** → acele (index 0)
- **ClassId 2** → acikmak (index 1)
- **ClassId 5** → agac (index 2)

Model output'u (0, 1, 2) gerçek ClassId'lere (1, 2, 5) dönüştürülür.

### Video Formatı

Test videoları:
- Format: MP4 (H.264)
- İsim: `signerX_sampleY_color.mp4`
- FPS: ~30 (değişken)
- Frame sayısı: 40-120 (değişken)

### Confidence Threshold

Script herhangi bir confidence threshold uygulamaz.
Tüm tahminler kaydedilir. İsterseniz CSV'yi filtreleyin:

```python
import pandas as pd

df = pd.read_csv('results/test_predictions.csv')

# Sadece yüksek confidence (>0.9)
high_conf = df[df['confidence'] > 0.9]

# Düşük confidence tahminler
low_conf = df[df['confidence'] < 0.7]
```

## 🎓 Best Practices

1. **İlk çalıştırma:** Videoları göstererek (`y`) çalıştırın, sonuçları görsel olarak kontrol edin.

2. **Batch processing:** Çok sayıda video için `n` seçerek hızlı işlem yapın.

3. **Hata analizi:** Yanlış tahminlerin video görsellerini inceleyin (confusion patterns).

4. **Performance monitoring:** JSON çıktısındaki `probabilities` alanını inceleyin (model ne kadar emin?).

## 📚 İlgili Scriptler

- `train.py` - Model eğitimi
- `evaluate.py` - Test seti değerlendirmesi (batch)
- `visualize_attention.py` - Attention haritaları
- `scripts/02_extract_keypoints.py` - Keypoint extraction

---

**Son Güncelleme:** Ekim 2025  
**Versiyon:** 1.0.0

