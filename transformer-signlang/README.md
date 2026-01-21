# Transformer-SignLang (226 Sınıf) — Eğitim & İnferans Rehberi

PyTorch tabanlı **Temporal Transformer** ile Türk İşaret Dili / AUTSL işaretlerini sınıflandırma. Girdi: MediaPipe Holistic keypoint dizileri (258 boyut, 200 frame). Çıktı: 226 sınıf etiketi.

> Ayrıntılı eski planlar için `Docs/` klasörünü inceleyin. Bu README güncel kullanım özetidir.

---

## Özeti Hızlı Gör

| Özellik | Değer |
|---------|-------|
| Mimari | Temporal Transformer Encoder |
| Girdi | 200×258 keypoint (Pose+Face+Hands) |
| Sınıf | 226 (AUTSL) |
| Checkpoint | `checkpoints/best_model.pth` |
| Normalizasyon | `Data/scaler.pkl` (train setinden) |
| Sonuçlar | `results/` (confusion matrix, per-class metrics, test_metrics.json) |

---

## Kurulum

```bash
cd transformer-signlang
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Klasörleri hazırla
mkdir -p data/keypoints data/processed checkpoints results logs
```

GPU için PyTorch (örnek):  
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## Veri Hazırlama Pipeline

> AUTSL benzeri dizin ve etiketlere sahip olmanız gerekir (`Data/Train Data/train`, `train_labels.csv`, `Data/Class ID/SignList_ClassId_TR_EN.csv`). Mevcut `Data/scaler.pkl` yalnızca referans içindir; kendi verinizle yeniden üretin.

1) **Video seçimi**  
`python scripts/01_select_videos.py`  
Çıktı: `data/selected_videos_*.csv`

2) **Keypoint çıkarımı (MediaPipe Holistic)**  
`python scripts/02_extract_keypoints.py`  
Çıktı: `data/keypoints/<video_id>.npy` (T×258)

3) **Normalize + Padding**  
`python scripts/03_normalize_data.py`  
- Scaler yalnızca train verisinde fit edilir → `data/scaler.pkl`  
- `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy` (+ label/id dosyaları) oluşturulur.

---

## Eğitim

```bash
python train.py

# Checkpoint'ten devam
python train.py --resume checkpoints/last_model.pth
python train.py --resume-from-best
```

- En iyi model: `checkpoints/best_model.pth`
- Son model: `checkpoints/last_model.pth`
- Early stopping (patience=10), cosine scheduler + warmup
- Opsiyonel izleme: TensorBoard (`logs/`), wandb (train.py içinde açılabilir)

---

## Değerlendirme

```bash
python evaluate.py
```

Çıktılar (`results/`):
- `test_metrics.json`
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `per_class_metrics.png`
- (varsa) `test_predictions.*`

---

## Hızlı İnferans

Hazır checkpoint + scaler ile kısa video testi:

```bash
python inference_test_videos.py --video path/to/video.mp4
```

Script, keypoint çıkarımı ve 226 sınıf tahmini yapar (MediaPipe’ın kurulu olduğundan emin olun).

---

## Dizin Yapısı

```
transformer-signlang/
├── config.py                  # Model/egitim konfigürasyonu
├── train.py                   # Eğitim
├── evaluate.py                # Test/değerlendirme
├── inference_test_videos.py   # Hızlı inferans
├── validate_setup.py          # Ortam kontrolü
│
├── models/
│   └── transformer_model.py   # Temporal Transformer
├── scripts/
│   ├── 01_select_videos.py
│   ├── 02_extract_keypoints.py
│   ├── 03_normalize_data.py
│   └── (plot/inspect/test yardımcıları)
├── Data/
│   └── scaler.pkl             # Normalizasyon (örnek)
├── checkpoints/
│   └── best_model.pth         # Eğitilmiş ağırlıklar
├── results/
│   ├── evaluation_report.json
│   ├── confusion_matrix*.png/.csv
│   ├── per_class_metrics*.
│   └── attention/ (rollout görselleri)
├── logs/
└── Docs/                      # Ayrıntılı/legacy dokümanlar
```

---

## Temel Parametreler (config.py)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| INPUT_DIM | 258 | Pose+Face+Hands keypoint boyutu |
| MAX_SEQ_LENGTH | 200 | Frame sayısı (padding/truncation) |
| NUM_CLASSES | 226 | AUTSL sınıf sayısı |
| D_MODEL | 256 | Transformer embedding |
| NHEAD | 8 | Attention head sayısı |
| NUM_ENCODER_LAYERS | 6 | Encoder blokları |
| DIM_FEEDFORWARD | 1024 | FFN gizli boyutu |
| DROPOUT | 0.1 | Dropout |
| BATCH_SIZE | 32 | Batch |
| LEARNING_RATE | 1e-4 | Başlangıç LR |
| WARMUP_EPOCHS | 10 | LR ısınma |
| LABEL_SMOOTHING | 0.1 | CE smoothing |

---

## Sorun Giderme

- **CUDA OOM**: `BATCH_SIZE` düşür, `D_MODEL`/`NUM_ENCODER_LAYERS` azalt.  
- **MediaPipe yavaş/bozuk**: `scripts/02_extract_keypoints.py` içinde `model_complexity=0`; `mediapipe==0.10.14` kurulu olsun.  
- **Checkpoint yok**: Eğitimden sonra `checkpoints/best_model.pth` oluşur; yoksa eğitimi tamamlayın.  
- **Veri uyumsuz**: `data/scaler.pkl` ve `data/processed/*` aynı sürümle üretilmiş olmalı; scaler yalnızca train setinde fit edilir.  

---

## Referans Dokümanlar

- Ayrıntılı/önceki planlar: `Docs/`
- Ekran görüntüleri: `ss/`
- Benchmark görselleri/raporları: `results/`

---

**Son Güncelleme:** 2026-01-20  
**Sürüm:** 2.0 (226 sınıf, güncel checkpoint)  
**Yazar:** sign-transcriber ekibi
# Transformer Tabanlı İşaret Dili Tanıma Projesi İş Planı

## 1. Proje Hedefi

Bu proje, Türk İşaret Dili (TİD) videolarını sınıflandırmak için **Transformer** tabanlı bir derin öğrenme modeli geliştirmeyi amaçlamaktadır. Proje kapsamında:

- **Amaç:** Video dizilerinden çıkarılan MediaPipe keypoint'leri kullanarak işaret dilindeki kelimeleri otomatik olarak tanıyan bir model oluşturmak.
- **Kapsam:** İlk aşamada **3 kelime** (abla, acele, acikmak) üzerinde proof-of-concept modeli geliştirilecek.
- **Metodoloji:** Transformer encoder mimarisini temporal (zamansal) veri işleme için uyarlayarak, LSTM tabanlı yaklaşımlara göre daha iyi performans hedeflenmektedir.

**Temel Avantajlar:**
- ✅ Uzun menzilli bağımlılıkları (long-range dependencies) daha iyi yakalama
- ✅ Paralel işleme ile hızlı eğitim
- ✅ Attention mekanizması ile yorumlanabilirlik
- ✅ Transfer learning potansiyeli

---

## 2. Veri Hazırlama Süreci

### 2.1. Veri Kaynağı ve Yapısı

**Ana Veri Dizini:** `/Data/Train Data/train/`

**Veri Formatı:**
- Video dosyaları: `signerX_sampleY_color.mp4` ve `signerX_sampleY_depth.mp4`
- Etiket dosyası: `/Data/Train Data/train_labels.csv`
- Sınıf eşleştirme: `/Data/Class ID/SignList_ClassId_TR_EN.csv`

**İlk 3 Kelime:**
```
ClassId | TR       | EN
--------|----------|----------
0       | abla     | sister
1       | acele    | hurry
2       | acikmak  | hungry
```

### 2.2. Adım 1: İlk 3 Kelimeye Ait Videoların Seçilmesi

**Dosya:** `scripts/01_select_videos.py`

Bu script aşağıdaki işlemleri gerçekleştirir:

1. **CSV Okuma:**
   ```python
   import pandas as pd
   
   # Etiket dosyasını oku
   labels_df = pd.read_csv('../Data/Train Data/train_labels.csv', 
                           header=None, 
                           names=['video_id', 'class_id'])
   
   # İlk 3 kelimeye ait videoları filtrele
   target_classes = [0, 1, 2]  # abla, acele, acikmak
   filtered_df = labels_df[labels_df['class_id'].isin(target_classes)]
   ```

2. **Video Yollarını Oluşturma:**
   ```python
   video_paths = []
   for idx, row in filtered_df.iterrows():
       video_id = row['video_id']
       class_id = row['class_id']
       color_path = f'../Data/Train Data/train/{video_id}_color.mp4'
       video_paths.append({
           'video_id': video_id,
           'class_id': class_id,
           'path': color_path
       })
   ```

3. **Seçilmiş Videoların Kaydedilmesi:**
   ```python
   selected_df = pd.DataFrame(video_paths)
   selected_df.to_csv('data/selected_videos.csv', index=False)
   ```

**Beklenen Çıktı:**
- `data/selected_videos.csv` dosyası oluşturulur
- Her satır: `video_id, class_id, path`

### 2.3. Adım 2: MediaPipe ile Keypoint Çıkarımı

**Dosya:** `scripts/02_extract_keypoints.py`

**MediaPipe Holistic Kullanımı:**
MediaPipe Holistic, aşağıdaki keypoint'leri çıkarır:

| Bileşen | Keypoint Sayısı | Koordinat Boyutu | Toplam |
|---------|----------------|------------------|---------|
| Yüz (Face) | 468 | x, y, z | 1404 |
| Poz (Pose) | 33 | x, y, z, visibility | 132 |
| Sol El (Left Hand) | 21 | x, y, z | 63 |
| Sağ El (Right Hand) | 21 | x, y, z | 63 |

**Keypoint Vektörü Boyutu Hesaplama:**

Projede kullanılacak olan **258 boyutlu** vektör şu şekilde oluşturulur:

```python
# Poz: 33 nokta × (x, y, z) = 99 boyut
pose_landmarks = 33 * 3  # 99

# Sol El: 21 nokta × (x, y, z) = 63 boyut
left_hand_landmarks = 21 * 3  # 63

# Sağ El: 21 nokta × (x, y, z) = 63 boyut
right_hand_landmarks = 21 * 3  # 63

# Yüz: İşaret dili için sadece key noktalar (11 nokta × 3) = 33 boyut
# Örn: göz çevreleri, kaş, burun, ağız köşeleri
face_key_landmarks = 11 * 3  # 33

# TOPLAM: 99 + 63 + 63 + 33 = 258 boyut
```

**İmplementasyon:**

```python
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_frame(results):
    """Bir frame'den 258 boyutlu keypoint vektörü çıkarır"""
    
    # Pose keypoints (33 × 3 = 99)
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*3)
    
    # Yüz keypoints (sadece key noktalar: 11 × 3 = 33)
    face_key_indices = [33, 133, 362, 263, 61, 291, 78, 308, 13, 14, 17]  # Göz, kaş, burun, ağız
    face = np.array([[results.face_landmarks.landmark[i].x, 
                      results.face_landmarks.landmark[i].y,
                      results.face_landmarks.landmark[i].z] 
                     for i in face_key_indices]).flatten() \
           if results.face_landmarks else np.zeros(11*3)
    
    # Sol el keypoints (21 × 3 = 63)
    left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
                if results.left_hand_landmarks else np.zeros(21*3)
    
    # Sağ el keypoints (21 × 3 = 63)
    right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
                 if results.right_hand_landmarks else np.zeros(21*3)
    
    # Birleştir: 99 + 33 + 63 + 63 = 258
    keypoints = np.concatenate([pose, face, left_hand, right_hand])
    
    return keypoints

def process_video(video_path, max_frames=None):
    """Video dosyasından keypoint sekansı çıkarır"""
    cap = cv2.VideoCapture(video_path)
    keypoint_sequence = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # MediaPipe işleme
            results = holistic.process(image)
            
            # Keypoint çıkarımı
            keypoints = extract_keypoints_from_frame(results)
            keypoint_sequence.append(keypoints)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
    
    cap.release()
    return np.array(keypoint_sequence)  # Shape: (num_frames, 258)
```

**Tüm Videoları İşleme:**

```python
def main():
    # Seçilmiş videoları oku
    selected_df = pd.read_csv('data/selected_videos.csv')
    
    # Keypoint'leri saklamak için dizin oluştur
    os.makedirs('data/keypoints', exist_ok=True)
    
    # Her videoyu işle
    for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df)):
        video_id = row['video_id']
        video_path = row['path']
        class_id = row['class_id']
        
        # Video kontrolü
        if not os.path.exists(video_path):
            print(f"Video bulunamadı: {video_path}")
            continue
        
        # Keypoint çıkarımı
        keypoints = process_video(video_path)
        
        # Kaydet: .npy formatında
        save_path = f'data/keypoints/{video_id}.npy'
        np.save(save_path, keypoints)
        
        # Metadata kaydet
        metadata = {
            'video_id': video_id,
            'class_id': class_id,
            'num_frames': len(keypoints),
            'keypoint_shape': keypoints.shape
        }
    
    print(f"✅ Toplam {len(selected_df)} video işlendi.")

if __name__ == '__main__':
    main()
```

**Beklenen Çıktı:**
- `data/keypoints/` dizini altında her video için `.npy` dosyası
- Her `.npy` dosyası: shape = `(num_frames, 258)`

### 2.4. Adım 3: Veri Normalizasyonu ve Augmentasyon

**Dosya:** `scripts/03_normalize_data.py`

**Normalizasyon Teknikleri:**

1. **Temporal Normalizasyon (Zaman Boyutu):**
   - Problem: Her video farklı uzunlukta
   - Çözüm: Padding veya dinamik uzunluk

2. **Spatial Normalizasyon (Özellik Boyutu):**
   - Z-score normalization (StandardScaler)
   - Min-Max normalization

**İmplementasyon:**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def normalize_keypoints(keypoint_sequences):
    """
    Tüm sekansları normalize eder
    
    Args:
        keypoint_sequences: List of arrays [(T1, 258), (T2, 258), ...]
    
    Returns:
        normalized_sequences: List of normalized arrays
        scaler: Fitted StandardScaler objesi (test için kaydedilecek)
    """
    # Tüm frame'leri birleştir
    all_frames = np.vstack([seq for seq in keypoint_sequences])
    
    # StandardScaler fit et
    scaler = StandardScaler()
    scaler.fit(all_frames)
    
    # Her sekansı ayrı ayrı normalize et
    normalized_sequences = []
    for seq in keypoint_sequences:
        normalized_seq = scaler.transform(seq)
        normalized_sequences.append(normalized_seq)
    
    return normalized_sequences, scaler

def pad_sequences(sequences, max_length=None, padding='post', truncating='post', value=0.0):
    """
    Sekansları aynı uzunluğa getirir (Keras pad_sequences benzeri)
    
    Args:
        sequences: List of arrays
        max_length: Hedef uzunluk (None ise en uzun sekans)
        padding: 'pre' veya 'post'
        truncating: 'pre' veya 'post'
        value: Padding değeri
    
    Returns:
        padded_array: (batch_size, max_length, feature_dim)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    feature_dim = sequences[0].shape[1]
    padded_array = np.full((len(sequences), max_length, feature_dim), value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        
        if seq_len > max_length:
            # Truncate
            if truncating == 'post':
                padded_array[i] = seq[:max_length]
            else:  # 'pre'
                padded_array[i] = seq[-max_length:]
        else:
            # Pad
            if padding == 'post':
                padded_array[i, :seq_len] = seq
            else:  # 'pre'
                padded_array[i, -seq_len:] = seq
    
    return padded_array

def main():
    # Keypoint dosyalarını yükle
    selected_df = pd.read_csv('data/selected_videos.csv')
    
    keypoint_sequences = []
    labels = []
    video_ids = []
    
    for idx, row in selected_df.iterrows():
        video_id = row['video_id']
        class_id = row['class_id']
        
        keypoint_path = f'data/keypoints/{video_id}.npy'
        if not os.path.exists(keypoint_path):
            continue
        
        keypoints = np.load(keypoint_path)
        keypoint_sequences.append(keypoints)
        labels.append(class_id)
        video_ids.append(video_id)
    
    # Normalizasyon
    normalized_sequences, scaler = normalize_keypoints(keypoint_sequences)
    
    # Scaler'ı kaydet (test için gerekli)
    with open('data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Padding (max_length hesapla)
    sequence_lengths = [len(seq) for seq in normalized_sequences]
    max_length = np.percentile(sequence_lengths, 95)  # %95 percentile
    max_length = int(max_length)
    
    print(f"Sekans uzunlukları - Min: {np.min(sequence_lengths)}, "
          f"Max: {np.max(sequence_lengths)}, "
          f"Mean: {np.mean(sequence_lengths):.2f}, "
          f"95th percentile: {max_length}")
    
    # Padding uygula
    padded_sequences = pad_sequences(
        normalized_sequences,
        max_length=max_length,
        padding='post',
        truncating='post',
        value=0.0
    )
    
    print(f"Padded sequences shape: {padded_sequences.shape}")
    # Expected: (num_videos, max_length, 258)
    
    # Kaydet
    np.save('data/X_normalized.npy', padded_sequences)
    np.save('data/y_labels.npy', np.array(labels))
    np.save('data/video_ids.npy', np.array(video_ids))
    
    print("✅ Normalizasyon ve padding tamamlandı!")

if __name__ == '__main__':
    main()
```

**Data Augmentation (Opsiyonel):**

```python
def augment_sequence(sequence):
    """Temporal augmentation teknikleri"""
    
    # 1. Zaman ölçekleme (temporal scaling)
    # Hız değiştirme: %80-120 arası
    
    # 2. Gaussian noise ekleme
    noise = np.random.normal(0, 0.01, sequence.shape)
    augmented = sequence + noise
    
    # 3. Frame dropping (bazı frame'leri atla)
    
    return augmented
```

### 2.5. Güncellenmiş Veri Pipeline

**ÖNEMLİ DEĞİŞİKLİK:** Train/Val/Test setleri zaten ayrı klasörlerde bulunduğu için veri bölümleme scripti kaldırıldı.

**Güncellenmiş Strateji:**

```python
from sklearn.model_selection import train_test_split
import numpy as np

def stratified_split(X, y, video_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Veri setini stratified (sınıf dengeli) olarak böler
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, 
                  train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Oranlar toplamı 1 olmalı"
    
    # Önce train ve temp (val+test) olarak böl
    X_train, X_temp, y_train, y_temp, train_ids, temp_ids = train_test_split(
        X, y, video_ids,
        test_size=(1 - train_ratio),
        stratify=y,
        random_state=random_state
    )
    
    # Temp'i val ve test olarak böl
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test, val_ids, test_ids = train_test_split(
        X_temp, y_temp, temp_ids,
        test_size=(1 - val_ratio_adjusted),
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids

def main():
    # Normalize edilmiş veriyi yükle
    X = np.load('data/X_normalized.npy')
    y = np.load('data/y_labels.npy')
    video_ids = np.load('data/video_ids.npy', allow_pickle=True)
    
    print(f"Toplam veri: {X.shape}")
    print(f"Sınıf dağılımı: {np.bincount(y)}")
    
    # Veri setini böl
    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids = stratified_split(
        X, y, video_ids,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )
    
    # İstatistikleri yazdır
    print(f"\n📊 Veri Bölümleme İstatistikleri:")
    print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  - Sınıf dağılımı: {np.bincount(y_train)}")
    print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  - Sınıf dağılımı: {np.bincount(y_val)}")
    print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  - Sınıf dağılımı: {np.bincount(y_test)}")
    
    # Kaydet
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/y_test.npy', y_test)
    
    # Video ID'lerini de kaydet (debugging için)
    np.save('data/processed/train_ids.npy', train_ids)
    np.save('data/processed/val_ids.npy', val_ids)
    np.save('data/processed/test_ids.npy', test_ids)
    
    print("\n✅ Veri bölümleme tamamlandı!")
    print(f"📁 Kaydedilen dosyalar: data/processed/")

if __name__ == '__main__':
    main()
```

**Beklenen Dizin Yapısı:**

```
transformer-signlang/
├── data/
│   ├── selected_videos.csv
│   ├── keypoints/
│   │   ├── signer0_sample1.npy
│   │   ├── signer0_sample2.npy
│   │   └── ...
│   ├── scaler.pkl
│   ├── X_normalized.npy
│   ├── y_labels.npy
│   ├── video_ids.npy
│   └── processed/
│       ├── X_train.npy    # Shape: (N_train, max_length, 258)
│       ├── X_val.npy      # Shape: (N_val, max_length, 258)
│       ├── X_test.npy     # Shape: (N_test, max_length, 258)
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
```

---

## 3. Model Mimarisi

### 3.1. Transformer Encoder Mimarisi Tasarımı

**Dosya:** `models/transformer_model.py`

Modelimiz, video keypoint dizilerini işlemek için özel olarak tasarlanmış bir **Temporal Transformer Encoder** mimarisini kullanacaktır.

**Mimari Bileşenleri:**

```
Input (Batch, Seq_Length, 258)
    ↓
[1] Input Projection Layer
    ↓
(Batch, Seq_Length, d_model)
    ↓
[2] Positional Encoding
    ↓
(Batch, Seq_Length, d_model)
    ↓
[3] Transformer Encoder Blocks (x N)
    ↓
(Batch, Seq_Length, d_model)
    ↓
[4] Global Average Pooling
    ↓
(Batch, d_model)
    ↓
[5] Classification Head
    ↓
(Batch, num_classes)
```

### 3.2. PyTorch İmplementasyonu

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding matrisi oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSignLanguageClassifier(nn.Module):
    """
    Temporal Transformer Encoder for Sign Language Recognition
    """
    def __init__(
        self,
        input_dim=258,          # Keypoint feature dimension
        d_model=256,            # Transformer embedding dimension
        nhead=8,                # Number of attention heads
        num_encoder_layers=6,   # Number of Transformer encoder layers
        dim_feedforward=1024,   # Dimension of feedforward network
        dropout=0.1,            # Dropout rate
        num_classes=3,          # Number of sign classes
        max_seq_length=200      # Maximum sequence length
    ):
        super(TransformerSignLanguageClassifier, self).__init__()
        
        self.d_model = d_model
        
        # [1] Input Projection: (batch, seq, 258) -> (batch, seq, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # [2] Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # [3] Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU aktivasyon (BERT-style)
            batch_first=True    # Input shape: (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # [4] Pooling Strategy
        # Opsiyonlar: 
        # - Global Average Pooling (GAP)
        # - [CLS] token (BERT-style)
        # - Last hidden state
        self.pooling_type = 'gap'  # 'gap', 'cls', or 'last'
        
        if self.pooling_type == 'cls':
            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # [5] Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length, input_dim=258)
            mask: (batch_size, seq_length) - Padding mask (optional)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_length, _ = x.shape
        
        # [1] Input projection
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # [2] Add CLS token if needed
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)
            
            if mask is not None:
                # CLS token için mask genişlet
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # [3] Positional encoding
        x = self.pos_encoder(x)
        
        # [4] Transformer encoding
        # mask: True pozisyonlar IGNORE edilir
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # [5] Pooling
        if self.pooling_type == 'gap':
            # Global Average Pooling
            if mask is not None:
                # Masked positions hariç ortalama al
                mask_expanded = (~mask).unsqueeze(-1).float()  # (batch, seq, 1)
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                x = x.mean(dim=1)  # (batch, d_model)
        
        elif self.pooling_type == 'cls':
            # [CLS] token'ı kullan
            x = x[:, 0, :]  # (batch, d_model)
        
        elif self.pooling_type == 'last':
            # Son pozisyondaki hidden state
            if mask is not None:
                # Her örnek için son valid pozisyonu bul
                lengths = (~mask).sum(dim=1) - 1  # (batch,)
                x = x[torch.arange(batch_size), lengths, :]
            else:
                x = x[:, -1, :]  # (batch, d_model)
        
        # [6] Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits
```

### 3.3. Model Hiperparametreleri

**Temel Konfigürasyon:**

```python
# config.py

class TransformerConfig:
    # Data parameters
    INPUT_DIM = 258           # MediaPipe keypoint dimension
    MAX_SEQ_LENGTH = 200      # Maximum video length (frames)
    NUM_CLASSES = 3           # abla, acele, acikmak
    
    # Model architecture
    D_MODEL = 256             # Embedding dimension
    NHEAD = 8                 # Multi-head attention heads
    NUM_ENCODER_LAYERS = 6    # Transformer blocks
    DIM_FEEDFORWARD = 1024    # FFN hidden size
    DROPOUT = 0.1             # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 10
    
    # Optimization
    OPTIMIZER = 'adamw'       # 'adam', 'adamw', 'sgd'
    SCHEDULER = 'cosine'      # 'cosine', 'step', 'plateau'
    LABEL_SMOOTHING = 0.1     # Label smoothing for cross-entropy
    
    # Regularization
    GRADIENT_CLIP = 1.0       # Gradient clipping
    
    # Paths
    DATA_DIR = 'data/processed'
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
```

**Alternatif Konfigürasyonlar:**

| Model Size | d_model | heads | layers | FFN | Params |
|-----------|---------|-------|--------|-----|--------|
| Tiny | 128 | 4 | 3 | 512 | ~1M |
| Small | 256 | 8 | 4 | 1024 | ~5M |
| Base | 256 | 8 | 6 | 1024 | ~8M |
| Large | 512 | 16 | 12 | 2048 | ~40M |

İlk denemeler için **Base** konfigürasyonu önerilir.

### 3.4. Model Özeti

```python
def print_model_summary(model, input_shape=(32, 200, 258)):
    """Model özeti ve parametre sayısı"""
    from torchinfo import summary
    
    summary(
        model,
        input_size=input_shape,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        depth=4
    )
    
    # Toplam parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model İstatistikleri:")
    print(f"Toplam parametreler: {total_params:,}")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")
    print(f"Model boyutu: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
```

---

## 4. Eğitim Stratejisi

### 4.1. Kayıp Fonksiyonu (Loss Function)

**Cross-Entropy Loss with Label Smoothing:**

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing ile Cross-Entropy Loss
    
    Overfitting'i azaltır ve model kalibrasyonunu iyileştirir.
    """
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        """
        Args:
            preds: (batch_size, num_classes) - Model logits
            target: (batch_size,) - Ground truth labels
        """
        num_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        # Label smoothing
        # True label: 1 - epsilon + epsilon/K
        # Other labels: epsilon/K
        targets = torch.zeros_like(log_preds).scatter_(
            1, target.unsqueeze(1), 1
        )
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        
        loss = (-targets * log_preds).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 4.2. Optimizer (AdamW)

```python
def create_optimizer(model, config):
    """
    AdamW optimizer with weight decay
    
    Transformer modelleri için önerilen optimizer
    """
    # Farklı katmanlar için farklı learning rate (optional)
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'classifier' not in n],
            'lr': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'classifier' in n],
            'lr': config.LEARNING_RATE * 10,  # Classifier için daha yüksek LR
            'weight_decay': 0  # Classifier'da weight decay yok
        }
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer
```

### 4.3. Learning Rate Scheduler

```python
def create_scheduler(optimizer, config, num_training_steps):
    """
    Cosine Annealing with Warmup
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup_steps = config.WARMUP_EPOCHS * num_training_steps
    
    # Warmup phase: Linear increase
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Main phase: Cosine annealing
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # Combine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler
```

### 4.4. Eğitim Loop'u

**Dosya:** `train.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import wandb  # Experiment tracking (optional)

class SignLanguageDataset(Dataset):
    """PyTorch Dataset for sign language keypoints"""
    
    def __init__(self, X, y):
        """
        Args:
            X: (N, seq_len, 258) numpy array
            y: (N,) numpy array
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_padding_mask(X):
    """
    Padding pozisyonları için mask oluştur
    
    Args:
        X: (batch, seq_len, features)
    
    Returns:
        mask: (batch, seq_len) - True for padding positions
    """
    # Eğer tüm feature'lar 0 ise padding
    mask = (X.sum(dim=-1) == 0)
    return mask


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config):
    """Bir epoch eğitim"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Padding mask oluştur
        mask = create_padding_mask(X_batch)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X_batch, mask=mask)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_targets, all_preds)
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for X_batch, y_batch in tqdm(dataloader, desc='Validation'):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Padding mask
        mask = create_padding_mask(X_batch)
        
        # Forward pass
        logits = model(X_batch, mask=mask)
        loss = criterion(logits, y_batch)
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())
    
    # Metrics
    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1


def main():
    # Configuration
    from config import TransformerConfig
    config = TransformerConfig()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data
    print("📂 Veri yükleniyor...")
    X_train = np.load(f'{config.DATA_DIR}/X_train.npy')
    y_train = np.load(f'{config.DATA_DIR}/y_train.npy')
    X_val = np.load(f'{config.DATA_DIR}/X_val.npy')
    y_val = np.load(f'{config.DATA_DIR}/y_val.npy')
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create datasets
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("🏗️  Model oluşturuluyor...")
    model = TransformerSignLanguageClassifier(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES,
        max_seq_length=config.MAX_SEQ_LENGTH
    ).to(device)
    
    print_model_summary(model)
    
    # Loss, optimizer, scheduler
    criterion = LabelSmoothingCrossEntropy(epsilon=config.LABEL_SMOOTHING)
    optimizer = create_optimizer(model, config)
    
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Experiment tracking (optional)
    # wandb.init(project='sign-language-transformer', config=vars(config))
    
    # Training loop
    print("\n🚀 Eğitim başlıyor...\n")
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Print metrics
        print(f"\n📊 Epoch {epoch} Sonuçları:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': vars(config)
            }
            
            torch.save(checkpoint, f'{config.CHECKPOINT_DIR}/best_model.pth')
            print(f"  ✅ Best model kaydedildi! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
        
        # Log to wandb
        # wandb.log({
        #     'train/loss': train_loss,
        #     'train/accuracy': train_acc,
        #     'val/loss': val_loss,
        #     'val/accuracy': val_acc,
        #     'val/f1': val_f1,
        #     'lr': optimizer.param_groups[0]['lr']
        # })
    
    print(f"\n🎉 Eğitim tamamlandı!")
    print(f"📈 En iyi validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
```

### 4.5. Eğitim İzleme Metrikleri

Eğitim sırasında izlenecek metrikler:

1. **Loss:** Train ve validation loss
2. **Accuracy:** Genel doğruluk
3. **F1-Score:** Sınıf dengesizliği için
4. **Learning Rate:** Scheduler takibi
5. **Gradient Norm:** Gradient explosion kontrolü

---

## 5. Değerlendirme Metrikleri

### 5.1. Test Seti Değerlendirmesi

**Dosya:** `evaluate.py`

```python
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names):
    """Test seti üzerinde model değerlendirmesi"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    for X_batch, y_batch in tqdm(test_loader, desc='Evaluating'):
        X_batch = X_batch.to(device)
        
        # Padding mask
        mask = create_padding_mask(X_batch)
        
        # Inference
        logits = model(X_batch, mask=mask)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='macro'),
        'recall': recall_score(all_targets, all_preds, average='macro'),
        'f1_score': f1_score(all_targets, all_preds, average='macro'),
    }
    
    # Per-class metrics
    print("\n📊 Classification Report:\n")
    print(classification_report(
        all_targets, all_preds,
        target_names=class_names,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return metrics, cm, cm_normalized, all_probs


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    """Confusion matrix görselleştirme"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion matrix kaydedildi: {save_path}")


def plot_normalized_confusion_matrix(cm_norm, class_names, save_path='results/confusion_matrix_normalized.png'):
    """Normalized confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Ratio'}
    )
    
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Normalized confusion matrix kaydedildi: {save_path}")


def plot_per_class_metrics(metrics_dict, class_names, save_path='results/per_class_metrics.png'):
    """Sınıf bazlı metrikler görselleştirme"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        metrics_dict['targets'],
        metrics_dict['preds'],
        average=None
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Per-class metrics kaydedildi: {save_path}")


def main():
    from config import TransformerConfig
    config = TransformerConfig()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    print("📂 Test verisi yükleniyor...")
    X_test = np.load(f'{config.DATA_DIR}/X_test.npy')
    y_test = np.load(f'{config.DATA_DIR}/y_test.npy')
    
    test_dataset = SignLanguageDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    print("🏗️  Model yükleniyor...")
    model = TransformerSignLanguageClassifier(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES,
        max_seq_length=config.MAX_SEQ_LENGTH
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(f'{config.CHECKPOINT_DIR}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model yüklendi (Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")
    
    # Class names
    class_names = ['abla', 'acele', 'acikmak']
    
    # Evaluate
    print("\n🧪 Test seti değerlendiriliyor...")
    metrics, cm, cm_norm, probs = evaluate_model(model, test_loader, device, class_names)
    
    # Print results
    print(f"\n{'='*50}")
    print("📈 TEST SETİ SONUÇLARI")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    # Save metrics JSON
    with open('results/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrices
    plot_confusion_matrix(cm, class_names)
    plot_normalized_confusion_matrix(cm_norm, class_names)
    
    # Plot per-class metrics
    plot_per_class_metrics(
        {'targets': y_test, 'preds': np.argmax(probs, axis=1)},
        class_names
    )
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv('results/confusion_matrix.csv')
    
    print("\n✅ Tüm sonuçlar 'results/' klasörüne kaydedildi!")

if __name__ == '__main__':
    main()
```

### 5.2. Değerlendirme Metrikleri Açıklaması

#### 5.2.1. Accuracy (Doğruluk)
```
Accuracy = (True Positives + True Negatives) / Total Samples
```
- Tüm doğru tahminlerin oranı
- Dengeli veri setlerinde iyi bir metrik

#### 5.2.2. Precision (Kesinlik)
```
Precision = True Positives / (True Positives + False Positives)
```
- Pozitif tahmin edilen örneklerden kaçının gerçekten pozitif olduğu
- "Modelin söylediği ne kadar güvenilir?"

#### 5.2.3. Recall (Duyarlılık)
```
Recall = True Positives / (True Positives + False Negatives)
```
- Gerçek pozitiflerin kaçının yakalandığı
- "Tüm pozitif örneklerin ne kadarını bulduk?"

#### 5.2.4. F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Precision ve Recall'un harmonik ortalaması
- Dengesiz veri setlerinde önemli

#### 5.2.5. Confusion Matrix (Karışıklık Matrisi)

```
                Predicted
              abla  acele  acikmak
Actual abla    [[TP   FP    FP   ]
      acele    [FP   TP    FP   ]
      acikmak  [FP   FP    TP   ]]
```

**Yorumlama:**
- Diagonal (köşegen): Doğru tahminler
- Off-diagonal: Karışan sınıflar
- En yüksek off-diagonal değerler: Model bu iki sınıfı karıştırıyor

### 5.3. Model Yorumlama: Attention Visualization

**Dosya:** `visualize_attention.py`

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_weights(model, video_sequence, save_path='results/attention_map.png'):
    """
    Transformer attention ağırlıklarını görselleştirir
    
    Bu, modelin videonun hangi frame'lerine odaklandığını gösterir
    """
    model.eval()
    
    with torch.no_grad():
        # Hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            # output[1] contains attention weights
            attention_weights.append(output[1].cpu().numpy())
        
        # Register hook on last encoder layer
        hook = model.transformer_encoder.layers[-1].self_attn.register_forward_hook(hook_fn)
        
        # Forward pass
        video_tensor = torch.FloatTensor(video_sequence).unsqueeze(0)  # (1, seq_len, 258)
        _ = model(video_tensor)
        
        hook.remove()
    
    # Plot attention
    attn = attention_weights[0][0]  # (num_heads, seq_len, seq_len)
    
    # Average over heads
    attn_avg = attn.mean(axis=0)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(attn_avg, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position (Frame)')
    plt.ylabel('Query Position (Frame)')
    plt.title('Attention Map (Averaged over Heads)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Attention map kaydedildi: {save_path}")
```

---

## 6. Gerekli Kütüphaneler

### 6.1. Python Gereksinimleri

**Dosya:** `requirements.txt`

```txt
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Computer Vision & MediaPipe
opencv-python>=4.8.0
mediapipe>=0.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress Bar
tqdm>=4.65.0

# Model Summary
torchinfo>=1.8.0

# Experiment Tracking (Optional)
# wandb>=0.15.0
# tensorboard>=2.13.0

# Utilities
pyyaml>=6.0
joblib>=1.3.0
```

### 6.2. Kurulum Komutları

```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Gereksinimleri kur
pip install --upgrade pip
pip install -r requirements.txt

# GPU için PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 6.3. Sistem Gereksinimleri

**Minimum:**
- RAM: 8 GB
- GPU: 4 GB VRAM (önerilir)
- Disk: 10 GB boş alan

**Önerilen:**
- RAM: 16 GB
- GPU: 8 GB+ VRAM (NVIDIA RTX 2060+)
- Disk: 20 GB SSD

---

## 7. Proje Dizin Yapısı

```
transformer-signlang/
│
├── README.md                    # Bu dosya
├── requirements.txt             # Python paketleri
├── config.py                    # Konfigürasyon parametreleri
│
├── data/                        # Veri dosyaları
│   ├── selected_videos_train.csv
│   ├── selected_videos_val.csv
│   ├── selected_videos_test.csv
│   ├── keypoints/               # MediaPipe keypoint'ler (.npy)
│   ├── scaler.pkl               # Normalizasyon scaler (SADECE train'den)
│   └── processed/               # Normalize edilmiş veriler
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── train_ids.npy
│       ├── val_ids.npy
│       └── test_ids.npy
│
├── scripts/                     # Veri hazırlama scriptleri
│   ├── 01_select_videos.py     # Train/Val/Test video seçimi (3 CSV)
│   ├── 02_extract_keypoints.py # MediaPipe keypoint çıkarımı
│   └── 03_normalize_data.py    # Normalize + Pad → processed/
│
├── models/                      # Model tanımları
│   ├── __init__.py
│   └── transformer_model.py    # Transformer model sınıfı
│
├── train.py                     # Eğitim scripti
├── evaluate.py                  # Değerlendirme scripti
├── visualize_attention.py       # Attention görselleştirme
│
├── checkpoints/                 # Eğitilmiş model checkpoints
│   └── best_model.pth
│
├── results/                     # Değerlendirme sonuçları
│   ├── test_metrics.json
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   └── per_class_metrics.png
│
└── logs/                        # Eğitim logları
    └── training.log
```

---

## 8. Adım Adım Kullanım Kılavuzu

### 8.1. Proje Kurulumu

```bash
cd transformer-signlang

# Virtual environment
python -m venv venv
source venv/bin/activate

# Paketleri kur
pip install -r requirements.txt

# Gerekli dizinleri oluştur
mkdir -p data/keypoints data/processed checkpoints results logs
```

### 8.2. Veri Hazırlama Pipeline

**ÖNEMLİ:** Train/Val/Test setleri zaten ayrı klasörlerde mevcut!

**Adım 1: Video Seçimi (Train/Val/Test)**
```bash
python scripts/01_select_videos.py
```
Çıktı: 
- `data/selected_videos_train.csv`
- `data/selected_videos_val.csv`
- `data/selected_videos_test.csv`

**Adım 2: Keypoint Çıkarımı**
```bash
python scripts/02_extract_keypoints.py
```
Çıktı: `data/keypoints/*.npy` dosyaları (tüm setler için)

**Adım 3: Normalizasyon ve Padding (Train/Val/Test)**
```bash
python scripts/03_normalize_data.py
```
**Kritik:** Scaler SADECE train setinde fit edilir!

Çıktı: 
- `data/processed/X_train.npy`, `y_train.npy`, `train_ids.npy`
- `data/processed/X_val.npy`, `y_val.npy`, `val_ids.npy`
- `data/processed/X_test.npy`, `y_test.npy`, `test_ids.npy`
- `data/scaler.pkl` (sadece train'den)

### 8.3. Model Eğitimi

```bash
# Normal eğitim (sıfırdan)
python train.py

# Checkpoint'ten devam etme (NEW!)
python train.py --resume checkpoints/last_model.pth
python train.py --resume-from-best
```

**Eğitim sırasında:**
- Progress bar ile epoch ilerlemesi
- Her epoch sonunda train/val metrikleri
- En iyi model otomatik kaydedilir (`checkpoints/best_model.pth`)
- Her 5 epoch'ta `last_model.pth` kaydedilir
- Early stopping (10 epoch patience)

**🔄 Checkpoint Resume Özelliği (NEW!):**

Eğitim kesintiye uğrarsa kaldığı yerden devam edebilirsiniz:

| Senaryo | Komut |
|---------|-------|
| Normal eğitim | `python train.py` |
| Last checkpoint'ten devam | `python train.py --resume checkpoints/last_model.pth` |
| Best model'den devam | `python train.py --resume-from-best` |

**Resume edilenler:**
- ✅ Model ağırlıkları
- ✅ Optimizer state (momentum, variance)
- ✅ Learning rate scheduler position
- ✅ Best accuracy tracking
- ✅ Training history (grafikler kopuksuz)
- ✅ Early stopping patience counter

**Faydaları:**
- 🔴 Elektrik kesintisi/sistem çökmesi koruması
- 🎯 GPU timeout'larında bölümleyebilme
- ⚡ Hiperparametre değişiklikleriyle devam

**Eğitim İzleme (Opsiyonel):**
```bash
# TensorBoard
tensorboard --logdir=logs

# Weights & Biases
wandb login
# train.py içinde wandb.init() satırını uncomment et
```

### 8.4. Model Değerlendirmesi

```bash
python evaluate.py
```

**Çıktılar:**
- Console'da classification report
- `results/test_metrics.json`
- `results/confusion_matrix.png`
- `results/confusion_matrix_normalized.png`
- `results/per_class_metrics.png`

### 8.5. Attention Görselleştirme

```bash
python visualize_attention.py --video_id signer0_sample1
```

---

## 9. Hiperparametre Optimizasyonu

### 9.1. Önerilen Deneyler

**Experiment 1: Model Boyutu**
- Tiny, Small, Base, Large konfigürasyonları
- Overfit mi? → Daha küçük model
- Underfit mi? → Daha büyük model

**Experiment 2: Learning Rate**
- [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
- Learning rate finder kullan

**Experiment 3: Augmentation**
- Temporal scaling
- Gaussian noise
- Frame dropping

**Experiment 4: Pooling Strategy**
- Global Average Pooling
- [CLS] token
- Last hidden state

### 9.2. Grid Search Örneği

```python
# scripts/hyperparameter_search.py

import itertools

param_grid = {
    'd_model': [128, 256, 512],
    'num_layers': [4, 6, 8],
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'dropout': [0.1, 0.2, 0.3]
}

for params in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    print(f"Training with: {config}")
    # Train model...
```

---

## 10. Gelecek Geliştirmeler

### 10.1. Model Mimarisi
- [ ] Multi-Scale Temporal Transformer
- [ ] Cross-Modal Transformer (RGB + Depth)
- [ ] Pre-training with self-supervised learning

### 10.2. Veri
- [ ] Tüm 226 kelimeye genişletme
- [ ] Data augmentation teknikleri
- [ ] Temporal super-resolution

### 10.3. Deployment
- [ ] ONNX export
- [ ] TorchScript conversion
- [ ] Quantization (INT8)
- [ ] Real-time inference pipeline

---

## 11. Sorun Giderme

### 11.1. Yaygın Hatalar

**1. CUDA Out of Memory**
```python
# config.py içinde:
BATCH_SIZE = 16  # 32 yerine
```

**2. MediaPipe Keypoint Çıkarımı Yavaş**
```python
# 02_extract_keypoints.py içinde:
model_complexity=0  # 1 yerine (daha hızlı ama daha az hassas)
```

**3. Model Overfit Oluyor**
```python
# Daha fazla regularization:
DROPOUT = 0.3  # 0.1 yerine
LABEL_SMOOTHING = 0.2  # 0.1 yerine
# Data augmentation ekle
```

**4. Model Converge Olmuyor**
```python
# Learning rate düşür:
LEARNING_RATE = 1e-5  # 1e-4 yerine
# Warmup epochs artır:
WARMUP_EPOCHS = 20  # 10 yerine
```

---

## 12. Referanslar

### 12.1. Akademik Makaleler

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper
   
2. **Video Transformer Network** (Bertasius et al., 2021)
   - Video için transformer adaptasyonu

3. **Sign Language Recognition with Transformers** (Özdemir et al., 2022)
   - İşaret dili tanıma için transformer kullanımı

### 12.2. Kod Referansları

- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 13. İletişim ve Destek

**Proje Sahibi:** [Your Name]  
**E-posta:** [your.email@example.com]  
**GitHub:** [github.com/yourusername/sign-language-transformer]

---

## 14. Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

---

**Son Güncelleme:** Ekim 2025  
**Versiyon:** 1.0.0

---

## 🚀 Hızlı Başlangıç Özeti

```bash
# 0. Conda Environment
conda activate transformers

# 1. Kurulum
pip install -r requirements.txt

# 2. Veri Hazırlama (Train/Val/Test ayrı setler)
python scripts/01_select_videos.py      # 3 CSV oluşturur
python scripts/02_extract_keypoints.py   # Tüm setler için keypoint
python scripts/03_normalize_data.py      # Normalize + Pad → processed/

# 3. Eğitim
python train.py

# 4. Değerlendirme
python evaluate.py
```

**Not:** Train/Val/Test setleri zaten ayrı klasörlerde mevcut, %100 train verisi kullanılıyor!

**Başarılar! 🎉**

