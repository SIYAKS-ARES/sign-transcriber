# Anlamlandırma Sistemi - Demo Çalıştırma Rehberi

## 🎯 Sistem Özeti

Transformer tabanlı işaret dili tanıma sistemi başarıyla entegre edildi:

- **Model**: 226 sınıf PyTorch Transformer (88% doğruluk)
- **Input**: 258 boyutlu MediaPipe keypoints (Pose + Face + Hands)
- **Sequence Length**: 200 frame
- **Device**: CPU (MPS uyumluluk sorunu nedeniyle)

## 🚀 Hızlı Başlangıç

### 1. Conda Environment Aktif Et

```bash
conda activate anlamlandirma
```

### 2. Flask Uygulamasını Başlat

```bash
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi
python app.py
```

Uygulama `http://localhost:5005` adresinde çalışacak.

### 3. Demo Sayfasına Git

Tarayıcıda: `http://localhost:5005/demo`

### 4. Test Videosu Yükle

`test_videos/` klasöründeki videolardan birini yükleyin:
- `acikmak_1.mp4`
- `acikmak_2.mp4`
- `acikmak_3.mp4`
- `acikmak_4.mp4`
- `acikmak_5.mp4`

## 📊 Test Sonuçları

### Test Komutu:
```bash
python test_transformer_model.py
```

### Örnek Sonuç (acikmak_2.mp4):

```
🏆 Tahmin: acikmak
📈 Güven: 0.8468 (84.68%)
✅ Eşik karşılandı: EVET

📋 Top-5:
   1. acikmak    - 0.8468 (84.68%)
   2. ben        - 0.0008 (0.08%)
   3. arkadas    - 0.0008 (0.08%)
   4. kemer      - 0.0008 (0.08%)
   5. yakin      - 0.0008 (0.08%)
```

## 🎬 Ekran Görüntüsü İçin

1. Flask uygulamasını çalıştırın
2. `/demo` sayfasında bir test videosu yükleyin
3. "Test Model" butonu ile sadece model çıktısını görün
4. Veya "Translate" butonu ile LLM entegrasyonunu test edin
5. Ekran görüntüsü alın (Cmd+Shift+4 - Mac)

## 📁 Dosya Yapısı

```
anlamlandirma-sistemi/
├── app.py                      # Flask uygulaması
├── local_model_handler.py      # Transformer model handler (GÜNCEL)
├── test_transformer_model.py   # Test script
├── test_videos/                # Test videoları
│   ├── acikmak_1.mp4
│   ├── acikmak_2.mp4
│   └── ...
├── requirements.txt            # Dependencies (GÜNCEL)
└── templates/
    └── demo.html               # Web arayüzü
```

## ⚙️ Sistem Gereksinimleri

- **Python**: 3.10
- **Conda Environment**: anlamlandirma
- **Temel Kütüphaneler**:
  - PyTorch 2.9.1
  - MediaPipe 0.10.14
  - OpenCV 4.12.0
  - Flask 3.1.2
  - scikit-learn 1.7.2
  - pandas 2.3.3

## 🔧 Model Detayları

### Transformer Checkpoint:
- **Path**: `transformer-signlang/checkpoints/best_model.pth`
- **Epoch**: 98
- **Val Accuracy**: 0.8787
- **Val F1**: 0.8756

### Scaler:
- **Path**: `transformer-signlang/data/scaler.pkl`
- **Type**: StandardScaler (Z-score normalization)

### Class Names:
- **Path**: `Data/Class ID/SignList_ClassId_TR_EN.csv`
- **Count**: 226 sınıf

## 📝 Notlar

- MPS (Apple Silicon GPU) transformer mask operasyonlarını desteklemediği için CPU kullanılıyor
- MediaPipe tespit oranları genellikle %90+ (Pose, Face, Hands)
- Video uzunluğu 200 frame'e normalize ediliyor (padding/truncation)
- Minimum güven eşiği: 0.3 (varsayılan)

## 🐛 Sorun Giderme

### Model yüklenemezse:
```bash
# Checkpoint ve scaler kontrolü
ls -lh transformer-signlang/checkpoints/best_model.pth
ls -lh transformer-signlang/data/scaler.pkl
```

### MediaPipe hatası:
```bash
# MediaPipe yeniden yükle
pip install --upgrade mediapipe
```

### PyTorch hatası:
```bash
# PyTorch yeniden yükle
pip install --upgrade torch torchvision
```

