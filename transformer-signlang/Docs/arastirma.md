# Örnek Transformer Tabanlı İşaret Dili Tanıma Projesi İş Planı

## 1. Proje Hedefi

Bu proje, **Türk İşaret Dili (TİD)** video verileri üzerinde **Transformer tabanlı derin öğrenme** modeli kullanarak işaret dili tanıma sistemi geliştirmeyi amaçlamaktadır.

### 1.1 Kapsam

-**Hedef Kelimeler:** İlk üç kelime (abla, acele, acikmak)

-**Veri Kaynağı:**`TID-N/videos/` dizini altındaki MediaPipe keypoint verileri

-**Model Türü:** Temporal Transformer (Sequence-to-Classification)

-**Özellik Vektörü:** Frame başına 258 boyutlu MediaPipe keypoint'leri

- 33 pose keypoints × 3 (x, y, z) = 99 boyut
- 21 sol el keypoints × 3 = 63 boyut
- 21 sağ el keypoints × 3 = 63 boyut
- 468 yüz keypoints (opsiyonel, DROP_FACE=True ise kullanılmaz)

### 1.2 Proje Çıktıları

- ✅ Eğitim için hazırlanmış veri seti (train/validation/test split)
- ✅ Transformer tabanlı işaret dili tanıma modeli
- ✅ Eğitilmiş model checkpoint'leri
- ✅ Kapsamlı değerlendirme raporları ve görselleştirmeler
- ✅ Gerçek zamanlı tahmin altyapısı

---

## 2. Veri Hazırlama Süreci

### 2.1 Mevcut Veri Yapısı

Proje dizini altında zaten işlenmiş keypoint verileri bulunmaktadır:

```

TID-N/videos/

├── abla/

│   ├── signer0_sample1034/

│   │   ├── frame_0001.npy  # 258 boyutlu keypoint

│   │   ├── frame_0002.npy

│   │   └── ... (değişken sayıda frame)

│   ├── signer0_sample1044/

│   └── ... (toplam ~128 örnek)

├── acele/

│   └── ... (toplam ~100 örnek)

└── acikmak/

    └── ... (toplam ~107 örnek)

```

Her `.npy` dosyası bir frame için 258 boyutlu numpy array içerir:

-**Shape:**`(258,)` veya `(1, 258)`

-**Data Type:**`float32` veya `float64`

### 2.2 Veri Yükleme ve Organizasyon

#### 2.2.1 Keypoint Dosyalarının Okunması

Her video örneği için keypoint'ler sequential olarak yüklenir:

```python

import numpy as np

import os

from glob import glob


defload_sequence_keypoints(sequence_path):

"""

    Bir video örneğine ait tüm frame keypoint'lerini yükler


    Args:

        sequence_path: signer_sample klasör yolu


    Returns:

        keypoints: (num_frames, 258) shape'inde numpy array

    """

# Frame dosyalarını sıralı şekilde al

    frame_files =sorted(glob(os.path.join(sequence_path, "frame_*.npy")))


# Her frame'i yükle

    frames = []

for frame_file in frame_files:

        keypoint = np.load(frame_file)

# Shape'i normalize et (258,) veya (1,258) -> (258,)

if keypoint.ndim >1:

            keypoint = keypoint.flatten()

        frames.append(keypoint)


# (num_frames, 258) shape'inde array oluştur

    keypoints = np.array(frames, dtype=np.float32)

return keypoints

```

#### 2.2.2 Tüm Veri Setinin Oluşturulması

```python

defbuild_dataset(video_root, class_names):

"""

    Tüm video örneklerini yükleyip etiketleriyle eşleştirir


    Args:

        video_root: TID-N/videos/ dizini

        class_names: ['abla', 'acele', 'acikmak']


    Returns:

        sequences: List of (num_frames, 258) arrays

        labels: List of integer class labels

        metadata: List of dicts with signer/sample info

    """

    sequences = []

    labels = []

    metadata = []


for class_id, class_name inenumerate(class_names):

        class_path = os.path.join(video_root, class_name)


# Tüm signer_sample klasörlerini bul

        sample_dirs =sorted(glob(os.path.join(class_path, "signer*_sample*")))


print(f"[{class_name}] {len(sample_dirs)} örnek bulundu")


for sample_dir in sample_dirs:

try:

# Keypoint'leri yükle

                keypoints = load_sequence_keypoints(sample_dir)


# Çok kısa veya çok uzun sekansları filtrele

if keypoints.shape[0] <10or keypoints.shape[0] >200:

print(f"⚠️ Filtrelendi: {sample_dir} (frame count: {keypoints.shape[0]})")

continue


                sequences.append(keypoints)

                labels.append(class_id)


# Metadata ekle (debugging için)

                sample_name = os.path.basename(sample_dir)

                metadata.append({

'class_name': class_name,

'class_id': class_id,

'sample_name': sample_name,

'num_frames': keypoints.shape[0]

                })


exceptExceptionas e:

print(f"❌ Hata: {sample_dir} - {str(e)}")

continue


return sequences, labels, metadata

```

### 2.3 Sekans Uzunluğu Normalizasyonu

Transformer modelleri sabit uzunlukta giriş bekler. Farklı uzunluklardaki videoları normalize etmek için iki yöntem:

#### 2.3.1 Yöntem 1: Padding/Truncation (Önerilen)

```python

defnormalize_sequence_length(sequences, target_length=60, mode='pad'):

"""

    Sekansları hedef uzunluğa normalize eder


    Args:

        sequences: List of (num_frames, 258) arrays

        target_length: Hedef frame sayısı

        mode: 'pad' (padding) veya 'interpolate' (yeniden örnekleme)


    Returns:

        normalized: (num_samples, target_length, 258) array

        masks: (num_samples, target_length) binary mask (padding tespiti için)

    """

    normalized = []

    masks = []


for seq in sequences:

        num_frames = seq.shape[0]


if mode =='pad':

if num_frames >= target_length:

# Truncate: İlk target_length frame'i al

                new_seq = seq[:target_length]

                mask = np.ones(target_length, dtype=np.float32)

else:

# Pad: Sıfırlarla doldur

                pad_length = target_length - num_frames

                new_seq = np.vstack([seq, np.zeros((pad_length, 258), dtype=np.float32)])

                mask = np.concatenate([np.ones(num_frames), np.zeros(pad_length)], dtype=np.float32)


elif mode =='interpolate':

# Temporal interpolation (her frame'i yeniden örnekle)

from scipy.interpolate import interp1d

            old_indices = np.linspace(0, num_frames -1, num_frames)

            new_indices = np.linspace(0, num_frames -1, target_length)


            interpolator = interp1d(old_indices, seq, axis=0, kind='linear')

            new_seq = interpolator(new_indices)

            mask = np.ones(target_length, dtype=np.float32)


        normalized.append(new_seq)

        masks.append(mask)


return np.array(normalized, dtype=np.float32), np.array(masks, dtype=np.float32)

```

**Önerilen Parametre:**

-`target_length = 60` (config.py'deki SEQ_LEN ile uyumlu)

-`mode = 'pad'` (Transformer'lar masking ile padding'i doğal olarak destekler)

#### 2.3.2 Yöntem 2: Temporal Interpolation

Video'nun hızına göre yeniden örnekleme yapar. Daha smooth ama hesaplama maliyeti yüksek.

### 2.4 Veri Normalizasyonu

Keypoint koordinatlarını normalize etmek model stabilitesi için kritik:

```python

defnormalize_keypoints(sequences, method='z-score'):

"""

    Keypoint değerlerini normalize eder


    Args:

        sequences: (num_samples, target_length, 258) array

        method: 'z-score' veya 'min-max'


    Returns:

        normalized_sequences: Normalize edilmiş array

        stats: {mean, std} veya {min, max} (inference için gerekli)

    """

if method =='z-score':

# Tüm veri seti üzerinden mean ve std hesapla

        mean = sequences.mean(axis=(0, 1), keepdims=True)  # (1, 1, 258)

        std = sequences.std(axis=(0, 1), keepdims=True) +1e-8


        normalized = (sequences - mean) / std

        stats = {'mean': mean, 'std': std}


elif method =='min-max':

        min_val = sequences.min(axis=(0, 1), keepdims=True)

        max_val = sequences.max(axis=(0, 1), keepdims=True)


        normalized = (sequences - min_val) / (max_val - min_val +1e-8)

        stats = {'min': min_val, 'max': max_val}


return normalized, stats

```

**⚠️ Önemli:** Normalizasyon istatistikleri (`stats`) mutlaka kaydedilmelidir! Inference sırasında yeni videolar aynı istatistiklerle normalize edilecek.

### 2.5 Veri Setinin Bölümlenmesi (Train/Val/Test Split)

```python

from sklearn.model_selection import train_test_split


defsplit_dataset(sequences, labels, metadata, 

                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,

                  stratify=True, random_seed=42):

"""

    Veri setini train/val/test olarak böler


    Args:

        sequences: (num_samples, seq_len, 258) array

        labels: (num_samples,) array

        metadata: List of dicts

        train_ratio: Eğitim seti oranı (0.8 = %80)

        val_ratio: Doğrulama seti oranı (0.1 = %10)

        test_ratio: Test seti oranı (0.1 = %10)

        stratify: Her sınıftan eşit oranda örnek al

        random_seed: Reproducibility için seed


    Returns:

        train_data: (X_train, y_train, meta_train)

        val_data: (X_val, y_val, meta_val)

        test_data: (X_test, y_test, meta_test)

    """

assertabs(train_ratio + val_ratio + test_ratio -1.0) <1e-6, "Oranlar toplamı 1 olmalı"


# İlk split: train vs (val+test)

    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(

        sequences, labels, metadata,

test_size=(1- train_ratio),

stratify=labels if stratify elseNone,

random_state=random_seed

    )


# İkinci split: val vs test

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(

        X_temp, y_temp, meta_temp,

test_size=(1- val_ratio_adjusted),

stratify=y_temp if stratify elseNone,

random_state=random_seed

    )


# Split istatistiklerini yazdır

print("\n📊 Veri Seti Bölünme İstatistikleri:")

print(f"  Train: {len(X_train)} örnek ({train_ratio*100:.0f}%)")

print(f"  Val:   {len(X_val)} örnek ({val_ratio*100:.0f}%)")

print(f"  Test:  {len(X_test)} örnek ({test_ratio*100:.0f}%)")


# Sınıf dağılımlarını kontrol et

print("\n📈 Sınıf Dağılımları:")

for split_name, split_labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:

        unique, counts = np.unique(split_labels, return_counts=True)

print(f"  {split_name}:")

forcls, cnt inzip(unique, counts):

print(f"    Class {cls}: {cnt} örnek")


return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)

```

**Önerilen Split Oranları:**

-**Train:** 80% (~268 örnek)

-**Validation:** 10% (~34 örnek)

-**Test:** 10% (~34 örnek)

### 2.6 Veri Kaydetme ve Yükleme

Hazırlanan veriyi disk'e kaydetmek eğitim sürecini hızlandırır:

```python

import pickle


defsave_processed_dataset(output_dir, train_data, val_data, test_data, 

                           norm_stats, class_names):

"""

    İşlenmiş veri setini disk'e kaydeder


    Args:

        output_dir: Kaydedilecek dizin (örn: TID-N/processed_data/)

        train_data, val_data, test_data: (X, y, meta) tuples

        norm_stats: Normalizasyon istatistikleri

        class_names: ['abla', 'acele', 'acikmak']

    """

    os.makedirs(output_dir, exist_ok=True)


# Veri setlerini kaydet

    np.save(os.path.join(output_dir, 'X_train.npy'), train_data[0])

    np.save(os.path.join(output_dir, 'y_train.npy'), train_data[1])


    np.save(os.path.join(output_dir, 'X_val.npy'), val_data[0])

    np.save(os.path.join(output_dir, 'y_val.npy'), val_data[1])


    np.save(os.path.join(output_dir, 'X_test.npy'), test_data[0])

    np.save(os.path.join(output_dir, 'y_test.npy'), test_data[1])


# Metadata ve config kaydet

withopen(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:

        pickle.dump({

'train_meta': train_data[2],

'val_meta': val_data[2],

'test_meta': test_data[2],

'norm_stats': norm_stats,

'class_names': class_names

        }, f)


print(f"✅ Veri seti başarıyla kaydedildi: {output_dir}")


defload_processed_dataset(data_dir):

"""

    Kaydedilmiş veri setini yükler


    Returns:

        train_data, val_data, test_data, metadata_dict

    """

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))

    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))


    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))

    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))


    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))

    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))


withopen(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:

        metadata = pickle.load(f)


return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata

```

### 2.7 Veri Hazırlama Pipeline'ı - Ana Script

Tüm adımları bir araya getiren `prepare_data.py` scripti:

```python

# prepare_data.py

import os

import numpy as np

from config importVIDEOS_ROOT, CLASS_NAMES, SEQ_LEN


defmain():

print("🚀 Transformer Veri Hazırlama Pipeline Başladı")

print("="*60)


# 1. Ham keypoint'leri yükle

print("\n[1/6] Keypoint'ler yükleniyor...")

    sequences, labels, metadata = build_dataset(VIDEOS_ROOT, CLASS_NAMES)

print(f"✅ Toplam {len(sequences)} örnek yüklendi")


# 2. Sekans uzunluklarını normalize et

print(f"\n[2/6] Sekanslar {SEQ_LEN} frame'e normalize ediliyor...")

    sequences_norm, masks = normalize_sequence_length(sequences, target_length=SEQ_LEN)

print(f"✅ Shape: {sequences_norm.shape}, Masks: {masks.shape}")


# 3. Keypoint normalizasyonu

print("\n[3/6] Keypoint normalizasyonu yapılıyor...")

    sequences_norm, norm_stats = normalize_keypoints(sequences_norm, method='z-score')

print(f"✅ Mean: {norm_stats['mean'].shape}, Std: {norm_stats['std'].shape}")


# 4. Train/Val/Test split

print("\n[4/6] Veri seti bölünüyor...")

    train_data, val_data, test_data = split_dataset(

        sequences_norm, np.array(labels), metadata,

train_ratio=0.8, val_ratio=0.1, test_ratio=0.1

    )


# 5. Disk'e kaydet

print("\n[5/6] Veri seti disk'e kaydediliyor...")

    output_dir = os.path.join(os.path.dirname(__file__), 'processed_data')

    save_processed_dataset(output_dir, train_data, val_data, test_data, 

                          norm_stats, CLASS_NAMES)


# 6. Özet istatistikler

print("\n[6/6] Veri hazırlama tamamlandı!")

print("="*60)

print(f"📁 Kaydedilen dizin: {output_dir}")

print(f"📊 Train: {train_data[0].shape}")

print(f"📊 Val:   {val_data[0].shape}")

print(f"📊 Test:  {test_data[0].shape}")


if__name__=="__main__":

    main()

```

**Kullanım:**

```bash

cdTID-N

pythonprepare_data.py

```

**Çıktı Dosya Yapısı:**

```

TID-N/processed_data/

├── X_train.npy          # (268, 60, 258)

├── y_train.npy          # (268,)

├── X_val.npy            # (34, 60, 258)

├── y_val.npy            # (34,)

├── X_test.npy           # (34, 60, 258)

├── y_test.npy           # (34,)

└── metadata.pkl         # normalization stats + class names

```

---

## 3. Model Mimarisi

### 3.1 Transformer Mimarisi Genel Bakış

Transformer modeli, video sekanslarındaki temporal (zamansal) bağımlılıkları yakalamak için **Multi-Head Self-Attention** mekanizmasını kullanır.

```

Input Keypoints (60, 258)

        ↓

[Input Projection] → (60, d_model)

        ↓

[Positional Encoding] → (60, d_model)

        ↓

[Transformer Encoder Block 1]

        ↓

[Transformer Encoder Block 2]

        ↓

[Transformer Encoder Block N]

        ↓

[Global Average Pooling] → (d_model,)

        ↓

[Classification Head] → (3,)

        ↓

Output: Softmax probabilities

```

### 3.2 Model Bileşenleri (PyTorch)

#### 3.2.1 Input Projection Layer

Keypoint vektörlerini Transformer'ın iç boyutuna (`d_model`) project eder:

```python

import torch

import torch.nn as nn


classInputProjection(nn.Module):

def__init__(self, input_dim=258, d_model=256, dropout=0.1):

"""

        Args:

            input_dim: Keypoint boyutu (258)

            d_model: Transformer hidden dim (256, 512, etc.)

            dropout: Regularization

        """

super().__init__()

self.projection = nn.Linear(input_dim, d_model)

self.dropout = nn.Dropout(dropout)

self.norm = nn.LayerNorm(d_model)


defforward(self, x):

# x: (batch, seq_len, 258)

        x =self.projection(x)  # (batch, seq_len, d_model)

        x =self.norm(x)

        x =self.dropout(x)

return x

```

#### 3.2.2 Positional Encoding

Transformer'lar sequential bilgiyi doğal olarak yakalayamaz, bu nedenle pozisyon bilgisi eklenir:

```python

classPositionalEncoding(nn.Module):

def__init__(self, d_model=256, max_len=100, dropout=0.1):

"""

        Sinusoidal Positional Encoding


        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))

        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        """

super().__init__()

self.dropout = nn.Dropout(dropout)


# Pozisyonel encoding tablosunu oluştur

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *

                            (-np.log(10000.0) / d_model))


        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)


# Buffer olarak kaydet (trainable değil)

self.register_buffer('pe', pe)


defforward(self, x):

# x: (batch, seq_len, d_model)

        x = x +self.pe[:, :x.size(1), :]

returnself.dropout(x)

```

**Alternatif:** Learnable positional embeddings (parametre sayısını artırır):

```python

self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

x = x +self.pos_embedding[:, :x.size(1), :]

```

#### 3.2.3 Transformer Encoder Block

Her Transformer block içerir:

1.**Multi-Head Self-Attention**

2.**Feed-Forward Network**

3.**Residual Connections + Layer Normalization**

```python

classTransformerEncoderBlock(nn.Module):

def__init__(self, d_model=256, num_heads=8, d_ff=1024, dropout=0.1):

"""

        Args:

            d_model: Model dimensiyonu

            num_heads: Attention head sayısı (d_model % num_heads == 0 olmalı)

            d_ff: Feed-forward hidden dim (genelde 4 * d_model)

            dropout: Dropout oranı

        """

super().__init__()


# Multi-Head Self-Attention

self.self_attn = nn.MultiheadAttention(

embed_dim=d_model,

num_heads=num_heads,

dropout=dropout,

batch_first=True# (batch, seq, feature) formatı için

        )


# Feed-Forward Network

self.ffn = nn.Sequential(

            nn.Linear(d_model, d_ff),

            nn.GELU(),  # GELU aktivasyonu (ReLU'dan daha smooth)

            nn.Dropout(dropout),

            nn.Linear(d_ff, d_model)

        )


# Layer Normalization (Pre-LN yapısı daha stabil)

self.norm1 = nn.LayerNorm(d_model)

self.norm2 = nn.LayerNorm(d_model)


self.dropout1 = nn.Dropout(dropout)

self.dropout2 = nn.Dropout(dropout)


defforward(self, x, mask=None):

"""

        Args:

            x: (batch, seq_len, d_model)

            mask: (batch, seq_len) - True/1 for valid tokens, False/0 for padding


        Returns:

            x: (batch, seq_len, d_model)

        """

# Self-Attention bloğu (Pre-LN)

        residual = x

        x =self.norm1(x)


# Attention mask'i PyTorch formatına çevir (opsiyonel)

        attn_mask =None

if mask isnotNone:

# mask: (batch, seq_len) -> attn_mask: (batch, seq_len)

# False/0 olan yerlere -inf atanır (attention'da kullanılmasın)

            attn_mask =~mask.bool()  # Invert: padding=True, valid=False


        x_attn, _ =self.self_attn(x, x, x, key_padding_mask=attn_mask)

        x = residual +self.dropout1(x_attn)


# Feed-Forward bloğu (Pre-LN)

        residual = x

        x =self.norm2(x)

        x_ffn =self.ffn(x)

        x = residual +self.dropout2(x_ffn)


return x

```

#### 3.2.4 Classification Head

Transformer encoder çıktısını sınıf olasılıklarına dönüştürür:

```python

classClassificationHead(nn.Module):

def__init__(self, d_model=256, num_classes=3, dropout=0.5, pooling='mean'):

"""

        Args:

            d_model: Transformer çıktı boyutu

            num_classes: Sınıf sayısı (3: abla, acele, acikmak)

            dropout: Regularization

            pooling: 'mean', 'max', 'cls' (CLS token kullanımı)

        """

super().__init__()

self.pooling = pooling


self.classifier = nn.Sequential(

            nn.LayerNorm(d_model),

            nn.Dropout(dropout),

            nn.Linear(d_model, d_model //2),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(d_model //2, num_classes)

        )


defforward(self, x, mask=None):

"""

        Args:

            x: (batch, seq_len, d_model)

            mask: (batch, seq_len) - padding mask


        Returns:

            logits: (batch, num_classes)

        """

# Temporal pooling

ifself.pooling =='mean':

# Masking-aware average pooling

if mask isnotNone:

                mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)

                x_sum = (x * mask_expanded).sum(dim=1)  # (batch, d_model)

                x_mean = x_sum / mask_expanded.sum(dim=1).clamp(min=1)  # Avoid div by zero

else:

                x_mean = x.mean(dim=1)

            x = x_mean


elifself.pooling =='max':

            x, _ = x.max(dim=1)


elifself.pooling =='cls':

# İlk token'ı CLS token olarak kullan

            x = x[:, 0, :]


# Classification

        logits =self.classifier(x)  # (batch, num_classes)

return logits

```

#### 3.2.5 Tam Transformer Model

Tüm bileşenleri bir araya getiren ana model:

```python

classSignLanguageTransformer(nn.Module):

def__init__(self, 

                 input_dim=258,

                 d_model=256,

                 num_heads=8,

                 num_layers=6,

                 d_ff=1024,

                 num_classes=3,

                 max_seq_len=100,

                 dropout=0.1,

                 pooling='mean'):

"""

        Transformer tabanlı işaret dili tanıma modeli


        Args:

            input_dim: Keypoint boyutu (258)

            d_model: Transformer hidden dim

            num_heads: Attention head sayısı

            num_layers: Encoder block sayısı

            d_ff: Feed-forward hidden dim

            num_classes: Sınıf sayısı

            max_seq_len: Maksimum sekans uzunluğu

            dropout: Dropout oranı

            pooling: Temporal pooling stratejisi

        """

super().__init__()


# Input projection

self.input_proj = InputProjection(input_dim, d_model, dropout)


# Positional encoding

self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)


# Transformer encoder blocks

self.encoder_layers = nn.ModuleList([

            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)

for _ inrange(num_layers)

        ])


# Classification head

self.classifier = ClassificationHead(d_model, num_classes, dropout, pooling)


defforward(self, x, mask=None):

"""

        Args:

            x: (batch, seq_len, input_dim=258)

            mask: (batch, seq_len) - binary mask (1=valid, 0=padding)


        Returns:

            logits: (batch, num_classes)

        """

# Input projection: (batch, seq_len, 258) -> (batch, seq_len, d_model)

        x =self.input_proj(x)


# Positional encoding

        x =self.pos_encoding(x)


# Transformer encoder layers

for encoder inself.encoder_layers:

            x = encoder(x, mask)


# Classification

        logits =self.classifier(x, mask)


return logits


defget_attention_weights(self, x, mask=None):

"""

        Attention haritalarını görselleştirme için çıkart

        """

# Bu fonksiyon analysis için kullanılır (opsiyonel)

pass

```

### 3.3 Model Konfigürasyonları

Farklı model boyutları için önerilen hiperparametreler:

#### 3.3.1 Small Model (Hızlı deney için)

```python

model = SignLanguageTransformer(

input_dim=258,

d_model=128,

num_heads=4,

num_layers=3,

d_ff=512,

num_classes=3,

dropout=0.1

)

# Parametre sayısı: ~500K

```

#### 3.3.2 Base Model (Önerilen)

```python

model = SignLanguageTransformer(

input_dim=258,

d_model=256,

num_heads=8,

num_layers=6,

d_ff=1024,

num_classes=3,

dropout=0.15

)

# Parametre sayısı: ~5M

```

#### 3.3.3 Large Model (Daha fazla veri için)

```python

model = SignLanguageTransformer(

input_dim=258,

d_model=512,

num_heads=8,

num_layers=8,

d_ff=2048,

num_classes=3,

dropout=0.2

)

# Parametre sayısı: ~25M

```

### 3.4 TensorFlow/Keras Alternatifi

PyTorch yerine TensorFlow tercih ederseniz:

```python

import tensorflow as tf

from tensorflow import keras


defcreate_transformer_model(seq_len=60, input_dim=258, d_model=256, 

                             num_heads=8, num_layers=6, num_classes=3):

"""

    Keras ile Transformer model

    """

# Input

    inputs = keras.Input(shape=(seq_len, input_dim))

    mask_input = keras.Input(shape=(seq_len,), dtype='bool')


# Input projection

    x = keras.layers.Dense(d_model)(inputs)

    x = keras.layers.LayerNormalization()(x)


# Positional encoding (learnable)

    pos_embedding = keras.layers.Embedding(seq_len, d_model)(

        tf.range(seq_len)

    )

    x = x + pos_embedding


# Transformer blocks

for _ inrange(num_layers):

# Multi-head attention

        attn_output = keras.layers.MultiHeadAttention(

num_heads=num_heads, key_dim=d_model // num_heads

        )(x, x, attention_mask=mask_input)

        x = keras.layers.Add()([x, attn_output])

        x = keras.layers.LayerNormalization()(x)


# Feed-forward

        ffn = keras.Sequential([

            keras.layers.Dense(d_model *4, activation='gelu'),

            keras.layers.Dense(d_model)

        ])

        ffn_output = ffn(x)

        x = keras.layers.Add()([x, ffn_output])

        x = keras.layers.LayerNormalization()(x)


# Global pooling

    x = keras.layers.GlobalAveragePooling1D()(x)


# Classification head

    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)


    model = keras.Model(inputs=[inputs, mask_input], outputs=outputs)

return model

```

---

## 4. Eğitim Stratejisi

### 4.1 Loss Fonksiyonu

#### 4.1.1 Cross-Entropy Loss (Standart)

Çok sınıflı sınıflandırma için standart loss:

```python

criterion = nn.CrossEntropyLoss()

```

**Matematiksel Formül:**

```

L = -Σ y_true_i * log(y_pred_i)

```

#### 4.1.2 Label Smoothing

Overconfidence'ı önlemek için yumuşatılmiş etiketler:

```python

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

```

Etiketler: `[1, 0, 0]` → `[0.93, 0.033, 0.033]`

#### 4.1.3 Focal Loss (Class Imbalance için)

Dengesiz sınıflar varsa (abla sınıfı daha fazla):

```python

classFocalLoss(nn.Module):

def__init__(self, alpha=None, gamma=2.0):

"""

        Args:

            alpha: Class weights [w0, w1, w2]

            gamma: Focusing parameter (yüksek = zor örneklere focus)

        """

super().__init__()

self.alpha = alpha

self.gamma = gamma


defforward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_loss = ((1- pt) **self.gamma) * ce_loss


ifself.alpha isnotNone:

            alpha_t =self.alpha[targets]

            focal_loss = alpha_t * focal_loss


return focal_loss.mean()


# Kullanım

criterion = FocalLoss(alpha=torch.tensor([1.5, 1.0, 1.0]), gamma=2.0)

```

### 4.2 Optimizer

#### 4.2.1 AdamW (Önerilen)

Weight decay ile düzenlenmiş Adam:

```python

from torch.optim import AdamW


optimizer = AdamW(

    model.parameters(),

lr=1e-4,              # Learning rate

betas=(0.9, 0.999),   # Momentum parametreleri

weight_decay=0.01,    # L2 regularization

eps=1e-8

)

```

**Learning Rate Önerileri:**

- Small model: `1e-3`
- Base model: `5e-4` veya `1e-4`
- Large model: `1e-4` veya `5e-5`

#### 4.2.2 Adam (Alternatif)

```python

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

```

### 4.3 Learning Rate Scheduler

Eğitim ilerledikçe learning rate'i azaltmak:

#### 4.3.1 ReduceLROnPlateau (Adaptif)

```python

from torch.optim.lr_scheduler import ReduceLROnPlateau


scheduler = ReduceLROnPlateau(

    optimizer,

mode='min',         # 'min' for loss, 'max' for accuracy

factor=0.5,         # LR'yi yarıya indir

patience=10,        # 10 epoch'ta iyileşme yoksa

verbose=True,

min_lr=1e-7

)


# Her epoch sonunda:

scheduler.step(val_loss)

```

#### 4.3.2 Cosine Annealing (Döngüsel)

```python

from torch.optim.lr_scheduler import CosineAnnealingLR


scheduler = CosineAnnealingLR(

    optimizer,

T_max=50,          # 50 epoch'luk döngü

eta_min=1e-6# Minimum LR

)


# Her epoch sonunda:

scheduler.step()

```

#### 4.3.3 Warmup + Cosine Decay (En İyi)

```python

defget_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):

"""

    İlk birkaç epoch warmup, sonra cosine decay

    """

deflr_lambda(current_step):

if current_step < num_warmup_steps:

returnfloat(current_step) /float(max(1, num_warmup_steps))

        progress =float(current_step - num_warmup_steps) /float(max(1, num_training_steps - num_warmup_steps))

returnmax(0.0, 0.5* (1.0+ math.cos(math.pi * progress)))


return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Kullanım

num_epochs =100

steps_per_epoch =len(train_loader)

scheduler = get_cosine_schedule_with_warmup(

    optimizer,

num_warmup_steps=5* steps_per_epoch,  # 5 epoch warmup

num_training_steps=num_epochs * steps_per_epoch

)


# Her batch sonunda:

scheduler.step()

```

### 4.4 Data Augmentation (Opsiyonel)

Keypoint verileri için augmentation teknikleri:

```python

classKeypointAugmentation:

"""

    Temporal ve spatial augmentation

    """

def__init__(self, 

                 temporal_jitter=0.1,

                 spatial_noise=0.02,

                 rotation_angle=5,

                 scale_range=(0.9, 1.1)):

self.temporal_jitter = temporal_jitter

self.spatial_noise = spatial_noise

self.rotation_angle = rotation_angle

self.scale_range = scale_range


def__call__(self, keypoints):

"""

        Args:

            keypoints: (seq_len, 258)

        Returns:

            augmented: (seq_len, 258)

        """

        keypoints = keypoints.copy()


# 1. Temporal jittering (frame'leri hafif kaydır)

if np.random.rand() <0.5:

            shift =int(len(keypoints) *self.temporal_jitter * np.random.randn())

            keypoints = np.roll(keypoints, shift, axis=0)


# 2. Gaussian noise (spatial)

if np.random.rand() <0.5:

            noise = np.random.normal(0, self.spatial_noise, keypoints.shape)

            keypoints = keypoints + noise


# 3. Scaling

if np.random.rand() <0.5:

            scale = np.random.uniform(*self.scale_range)

            keypoints = keypoints * scale


# 4. Horizontal flip (x koordinatlarını tersine çevir)

if np.random.rand() <0.3:

# Her 3. boyut x koordinatı (0, 3, 6, ...)

            keypoints[:, 0::3] =1.0- keypoints[:, 0::3]

# Sol ve sağ eli değiştir (99:162 ile 162:225)

            left_hand = keypoints[:, 99:162].copy()

            right_hand = keypoints[:, 162:225].copy()

            keypoints[:, 99:162] = right_hand

            keypoints[:, 162:225] = left_hand


return keypoints

```

**⚠️ Dikkat:** Augmentation'ı sadece training sırasında uygulayın, validation/test'te değil.

### 4.5 Training Loop

Ana eğitim döngüsü:

```python

import torch

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm


deftrain_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):

"""

    Bir epoch eğitim


    Returns:

        avg_loss, avg_acc

    """

    model.train()

    total_loss =0

    total_correct =0

    total_samples =0


    pbar = tqdm(train_loader, desc='Training')

for batch_idx, (data, targets, masks) inenumerate(pbar):

        data = data.to(device)       # (batch, 60, 258)

        targets = targets.to(device)  # (batch,)

        masks = masks.to(device)      # (batch, 60)


# Forward pass

        optimizer.zero_grad()

        outputs = model(data, masks)  # (batch, num_classes)

        loss = criterion(outputs, targets)


# Backward pass

        loss.backward()


# Gradient clipping (stability için)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        optimizer.step()


if scheduler isnotNone:

            scheduler.step()  # Batch-level scheduler için


# Metrics

        _, predicted = outputs.max(1)

        total_loss += loss.item() * data.size(0)

        total_correct += predicted.eq(targets).sum().item()

        total_samples += data.size(0)


# Progress bar güncelle

        pbar.set_postfix({

'loss': f'{loss.item():.4f}',

'acc': f'{100. * total_correct / total_samples:.2f}%'

        })


    avg_loss = total_loss / total_samples

    avg_acc =100. * total_correct / total_samples


return avg_loss, avg_acc


defvalidate(model, val_loader, criterion, device):

"""

    Validation


    Returns:

        avg_loss, avg_acc

    """

    model.eval()

    total_loss =0

    total_correct =0

    total_samples =0


with torch.no_grad():

for data, targets, masks in tqdm(val_loader, desc='Validation'):

            data = data.to(device)

            targets = targets.to(device)

            masks = masks.to(device)


            outputs = model(data, masks)

            loss = criterion(outputs, targets)


            _, predicted = outputs.max(1)

            total_loss += loss.item() * data.size(0)

            total_correct += predicted.eq(targets).sum().item()

            total_samples += data.size(0)


    avg_loss = total_loss / total_samples

    avg_acc =100. * total_correct / total_samples


return avg_loss, avg_acc

```

### 4.6 Full Training Script

```python

# train.py

import torch

from torch.utils.data import DataLoader, TensorDataset


defmain():

# Konfigürasyon

DEVICE= torch.device('cuda'if torch.cuda.is_available() else'cpu')

BATCH_SIZE=32

NUM_EPOCHS=100

LEARNING_RATE=1e-4


print(f"🚀 Training başlıyor - Device: {DEVICE}")

print("="*60)


# 1. Veriyi yükle

print("\n[1/5] Veri yükleniyor...")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_processed_dataset('processed_data')


# Masks oluştur (padding detection)

    masks_train = (X_train.sum(axis=-1) !=0).astype(np.float32)

    masks_val = (X_val.sum(axis=-1) !=0).astype(np.float32)


# PyTorch tensors

    train_dataset = TensorDataset(

        torch.FloatTensor(X_train),

        torch.LongTensor(y_train),

        torch.FloatTensor(masks_train)

    )

    val_dataset = TensorDataset(

        torch.FloatTensor(X_val),

        torch.LongTensor(y_val),

        torch.FloatTensor(masks_val)

    )


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# 2. Model oluştur

print("\n[2/5] Model oluşturuluyor...")

    model = SignLanguageTransformer(

input_dim=258,

d_model=256,

num_heads=8,

num_layers=6,

d_ff=1024,

num_classes=3,

dropout=0.15

    ).to(DEVICE)


print(f"✅ Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")


# 3. Optimizer & Loss

print("\n[3/5] Optimizer ve loss hazırlanıyor...")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


# 4. Training loop

print("\n[4/5] Eğitim başlıyor...")

    best_val_acc =0

    patience_counter =0

EARLY_STOP_PATIENCE=30


for epoch inrange(1, NUM_EPOCHS+1):

print(f"\n{'='*60}")

print(f"Epoch {epoch}/{NUM_EPOCHS}")

print(f"{'='*60}")


# Train

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)


# Validate

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)


# Scheduler step

        scheduler.step(val_loss)


# Logging

print(f"\n📊 Epoch {epoch} Sonuçları:")

print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")


# Model checkpoint (en iyi modeli kaydet)

if val_acc > best_val_acc:

            best_val_acc = val_acc

            patience_counter =0


            checkpoint = {

'epoch': epoch,

'model_state_dict': model.state_dict(),

'optimizer_state_dict': optimizer.state_dict(),

'val_acc': val_acc,

'val_loss': val_loss

            }

            torch.save(checkpoint, 'checkpoints/best_model.pth')

print(f"  ✅ En iyi model kaydedildi! (Val Acc: {val_acc:.2f}%)")

else:

            patience_counter +=1


# Early stopping

if patience_counter >=EARLY_STOP_PATIENCE:

print(f"\n⏹️ Early stopping! {EARLY_STOP_PATIENCE} epoch'ta iyileşme yok.")

break


# 5. Training tamamlandı

print("\n[5/5] Training tamamlandı!")

print(f"✅ En iyi validation accuracy: {best_val_acc:.2f}%")


if__name__=="__main__":

    main()

```

**Kullanım:**

```bash

cdTID-N

pythontrain.py

```

### 4.7 TensorBoard İzleme

Eğitim sürecini görselleştirmek için:

```python

from torch.utils.tensorboard import SummaryWriter


# Training script başında:

writer = SummaryWriter('runs/transformer_experiment_1')


# Her epoch sonunda:

writer.add_scalar('Loss/train', train_loss, epoch)

writer.add_scalar('Loss/val', val_loss, epoch)

writer.add_scalar('Accuracy/train', train_acc, epoch)

writer.add_scalar('Accuracy/val', val_acc, epoch)

writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)


# Training bitince:

writer.close()

```

**TensorBoard başlatma:**

```bash

tensorboard--logdir=runs

```

---

## 5. Değerlendirme Metrikleri

### 5.1 Test Seti Üzerinde Değerlendirme

```python

# evaluate.py

import torch

import numpy as np

from sklearn.metrics import (

    accuracy_score, precision_score, recall_score, f1_score,

    classification_report, confusion_matrix

)

import matplotlib.pyplot as plt

import seaborn as sns


defevaluate_model(model, test_loader, device, class_names):

"""

    Model'i test seti üzerinde kapsamlı değerlendir


    Returns:

        metrics: Dictionary of evaluation metrics

        predictions: Array of predictions

        targets: Array of ground truth

    """

    model.eval()


    all_preds = []

    all_targets = []

    all_probs = []


with torch.no_grad():

for data, targets, masks in tqdm(test_loader, desc='Testing'):

            data = data.to(device)

            targets = targets.to(device)

            masks = masks.to(device)


            outputs = model(data, masks)

            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)


            all_preds.extend(predicted.cpu().numpy())

            all_targets.extend(targets.cpu().numpy())

            all_probs.extend(probs.cpu().numpy())


    all_preds = np.array(all_preds)

    all_targets = np.array(all_targets)

    all_probs = np.array(all_probs)


# Metrikleri hesapla

    metrics = {

'accuracy': accuracy_score(all_targets, all_preds),

'precision_macro': precision_score(all_targets, all_preds, average='macro'),

'recall_macro': recall_score(all_targets, all_preds, average='macro'),

'f1_macro': f1_score(all_targets, all_preds, average='macro'),

'precision_per_class': precision_score(all_targets, all_preds, average=None),

'recall_per_class': recall_score(all_targets, all_preds, average=None),

'f1_per_class': f1_score(all_targets, all_preds, average=None)

    }


return metrics, all_preds, all_targets, all_probs

```

### 5.2 Metrik Açıklamaları

#### 5.2.1 Accuracy (Doğruluk)

```

Accuracy = (Doğru Tahminler) / (Toplam Tahminler)

```

**Örnek:**

- 34 test örneğinden 30'u doğru → Accuracy = 30/34 = 88.2%

**Kısıtlamalar:**

- Dengesiz sınıflar varsa yanıltıcı olabilir
- Örnek: %90 abla, %10 diğerleri → Hep "abla" tahmin et → %90 accuracy!

#### 5.2.2 Precision (Kesinlik)

```

Precision = True Positives / (True Positives + False Positives)

```

**Anlamı:** Model "abla" dediğinde ne kadar haklı?

**Örnek (abla sınıfı için):**

- Model 12 kez "abla" dedi
- Bunlardan 10'u gerçekten abla idi
- Precision = 10/12 = 0.833

#### 5.2.3 Recall (Duyarlılık / Sensitivity)

```

Recall = True Positives / (True Positives + False Negatives)

```

**Anlamı:** Gerçek "abla" örneklerinin kaçını bulduk?

**Örnek (abla sınıfı için):**

- Test setinde 11 tane abla var
- Model bunlardan 10'unu buldu
- Recall = 10/11 = 0.909

#### 5.2.4 F1-Score

```

F1 = 2 * (Precision * Recall) / (Precision + Recall)

```

**Anlamı:** Precision ve Recall'ın harmonik ortalaması

**Örnek:**

- Precision = 0.833, Recall = 0.909
- F1 = 2 * (0.833 * 0.909) / (0.833 + 0.909) = 0.870

### 5.3 Classification Report

```python

defprint_classification_report(targets, predictions, class_names):

"""

    Sınıf bazında detaylı rapor

    """

    report = classification_report(

        targets, predictions,

target_names=class_names,

digits=4

    )


print("\n📋 Classification Report:")

print("="*60)

print(report)


# JSON formatında kaydet

    report_dict = classification_report(

        targets, predictions,

target_names=class_names,

output_dict=True

    )


withopen('evaluation_report.json', 'w') as f:

        json.dump(report_dict, f, indent=4)


print("✅ Rapor kaydedildi: evaluation_report.json")

```

**Örnek Çıktı:**

```

              precision    recall  f1-score   support


        abla     0.8333    0.9091    0.8696        11

       acele     0.9000    0.8182    0.8571        11

     acikmak     0.9167    0.9167    0.9167        12


    accuracy                         0.8824        34

   macro avg     0.8833    0.8813    0.8812        34

weighted avg     0.8840    0.8824    0.8820        34

```

### 5.4 Confusion Matrix

Hangi sınıfların birbirine karıştığını gösterir:

```python

defplot_confusion_matrix(targets, predictions, class_names, save_path='confusion_matrix.png'):

"""

    Confusion matrix görselleştirmesi

    """

    cm = confusion_matrix(targets, predictions)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# Raw counts

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 

xticklabels=class_names, yticklabels=class_names, ax=ax1)

    ax1.set_title('Confusion Matrix (Raw Counts)')

    ax1.set_ylabel('True Label')

    ax1.set_xlabel('Predicted Label')


# Normalized

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',

xticklabels=class_names, yticklabels=class_names, ax=ax2)

    ax2.set_title('Confusion Matrix (Normalized)')

    ax2.set_ylabel('True Label')

    ax2.set_xlabel('Predicted Label')


    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"✅ Confusion matrix kaydedildi: {save_path}")

    plt.show()

```

**Örnek Confusion Matrix:**

```

           abla  acele  acikmak

abla         10      1        0

acele         1      9        1

acikmak       0      1       11

```

**Yorumlama:**

- Diagonal (köşegen) yüksek = İyi!
- abla → acele: 1 kez karıştırıldı
- acele → acikmak: 1 kez karıştırıldı

### 5.5 Per-Class Visualizations

Sınıf bazlı performans grafikleri:

```python

defplot_per_class_metrics(metrics, class_names, save_path='per_class_metrics.png'):

"""

    Her sınıf için precision, recall, f1-score grafiği

    """

    x = np.arange(len(class_names))

    width =0.25


    fig, ax = plt.subplots(figsize=(12, 6))


    ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', alpha=0.8)

    ax.bar(x, metrics['recall_per_class'], width, label='Recall', alpha=0.8)

    ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', alpha=0.8)


    ax.set_ylabel('Score')

    ax.set_title('Per-Class Performance Metrics')

    ax.set_xticks(x)

    ax.set_xticklabels(class_names)

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    ax.set_ylim([0, 1.0])


    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"✅ Per-class metrics grafiği kaydedildi: {save_path}")

    plt.show()

```

### 5.6 Confidence Distribution

Model ne kadar güvenli tahmin yapıyor?

```python

defplot_confidence_distribution(probs, targets, save_path='confidence_dist.png'):

"""

    Tahmin güven dağılımı

    """

# Her örnek için maximum probability

    max_probs = probs.max(axis=1)


# Doğru ve yanlış tahminler için ayrı ayrı

    predictions = probs.argmax(axis=1)

    correct_mask = (predictions == targets)


    correct_probs = max_probs[correct_mask]

    incorrect_probs = max_probs[~correct_mask]


    fig, ax = plt.subplots(figsize=(10, 6))


    ax.hist(correct_probs, bins=20, alpha=0.7, label='Correct Predictions', color='green', range=(0, 1))

    ax.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect Predictions', color='red', range=(0, 1))


    ax.set_xlabel('Confidence (Max Probability)')

    ax.set_ylabel('Count')

    ax.set_title('Prediction Confidence Distribution')

    ax.legend()

    ax.grid(axis='y', alpha=0.3)


    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"✅ Confidence distribution grafiği kaydedildi: {save_path}")

    plt.show()


# İstatistikler

print("\n📊 Confidence İstatistikleri:")

print(f"  Doğru tahminler - Ortalama güven: {correct_probs.mean():.4f} ± {correct_probs.std():.4f}")

print(f"  Yanlış tahminler - Ortalama güven: {incorrect_probs.mean():.4f} ± {incorrect_probs.std():.4f}")

```

### 5.7 Tam Evaluation Script

```python

# evaluate.py

defmain():

DEVICE= torch.device('cuda'if torch.cuda.is_available() else'cpu')


print("🔍 Model Değerlendirme Başlıyor")

print("="*60)


# 1. Test verisini yükle

print("\n[1/6] Test verisi yükleniyor...")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_processed_dataset('processed_data')

    class_names = metadata['class_names']


    masks_test = (X_test.sum(axis=-1) !=0).astype(np.float32)

    test_dataset = TensorDataset(

        torch.FloatTensor(X_test),

        torch.LongTensor(y_test),

        torch.FloatTensor(masks_test)

    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2. Model yükle

print("\n[2/6] Model yükleniyor...")

    model = SignLanguageTransformer(

input_dim=258, d_model=256, num_heads=8,

num_layers=6, d_ff=1024, num_classes=3

    ).to(DEVICE)


    checkpoint = torch.load('checkpoints/best_model.pth', map_location=DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])

print(f"✅ Model yüklendi (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")


# 3. Evaluate

print("\n[3/6] Test seti üzerinde değerlendirme yapılıyor...")

    metrics, predictions, targets, probs = evaluate_model(model, test_loader, DEVICE, class_names)


# 4. Metrikleri yazdır

print("\n[4/6] Metrikler hesaplandı:")

print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

print(f"  Precision: {metrics['precision_macro']:.4f}")

print(f"  Recall:    {metrics['recall_macro']:.4f}")

print(f"  F1-Score:  {metrics['f1_macro']:.4f}")


# 5. Görselleştirmeler

print("\n[5/6] Görselleştirmeler oluşturuluyor...")

    print_classification_report(targets, predictions, class_names)

    plot_confusion_matrix(targets, predictions, class_names, 'results/confusion_matrix.png')

    plot_per_class_metrics(metrics, class_names, 'results/per_class_metrics.png')

    plot_confidence_distribution(probs, targets, 'results/confidence_distribution.png')


# 6. Sonuçları kaydet

print("\n[6/6] Sonuçlar kaydediliyor...")

    results = {

'test_accuracy': float(metrics['accuracy']),

'test_precision': float(metrics['precision_macro']),

'test_recall': float(metrics['recall_macro']),

'test_f1': float(metrics['f1_macro']),

'per_class_metrics': {

            class_names[i]: {

'precision': float(metrics['precision_per_class'][i]),

'recall': float(metrics['recall_per_class'][i]),

'f1': float(metrics['f1_per_class'][i])

            }

for i inrange(len(class_names))

        }

    }


withopen('results/evaluation_results.json', 'w') as f:

        json.dump(results, f, indent=4)


print("\n✅ Değerlendirme tamamlandı!")

print(f"📁 Sonuçlar 'results/' klasörüne kaydedildi")


if__name__=="__main__":

    main()

```

**Kullanım:**

```bash

cdTID-N

pythonevaluate.py

```

---

## 6. Gerekli Kütüphaneler

### 6.1 requirements.txt

```txt

# Deep Learning Framework

torch>=2.0.0

torchvision>=0.15.0

torchaudio>=2.0.0


# Alternative: TensorFlow

# tensorflow>=2.12.0


# Data Processing

numpy>=1.24.0

pandas>=2.0.0

scipy>=1.10.0


# Computer Vision

opencv-python>=4.7.0

mediapipe>=0.10.0


# Machine Learning Utilities

scikit-learn>=1.2.0


# Visualization

matplotlib>=3.7.0

seaborn>=0.12.0

tensorboard>=2.12.0


# Progress Bars

tqdm>=4.65.0


# Utilities

pillow>=9.5.0

pyyaml>=6.0

```

### 6.2 Kurulum

```bash

# Virtual environment oluştur

python3-mvenvvenv

sourcevenv/bin/activate# Linux/Mac

# veya

venv\Scripts\activate# Windows


# Kütüphaneleri kur

pipinstall-rrequirements.txt


# GPU desteği için (CUDA 11.8)

pipinstalltorchtorchvisiontorchaudio--index-urlhttps://download.pytorch.org/whl/cu118

```

### 6.3 Sistem Gereksinimleri

**Minimum:**

- CPU: 4 cores
- RAM: 8 GB
- Disk: 5 GB boş alan
- GPU: Opsiyonel (CPU ile de çalışır)

**Önerilen:**

- CPU: 8+ cores
- RAM: 16 GB
- Disk: 20 GB boş alan
- GPU: NVIDIA GTX 1660 veya üzeri (6GB+ VRAM)
- CUDA: 11.8 veya üzeri

---

## 7. Proje Dosya Yapısı

```

TID-N/

├── README.md                    # Bu dosya

├── requirements.txt             # Kütüphane bağımlılıkları

├── config.py                    # Konfigürasyon parametreleri

│

├── prepare_data.py              # Veri hazırlama scripti

├── train.py                     # Eğitim scripti

├── evaluate.py                  # Değerlendirme scripti

├── infer_videos.py              # Inference scripti

│

├── models/

│   ├── __init__.py

│   ├── transformer.py           # Transformer model tanımı

│   └── utils.py                 # Model yardımcı fonksiyonları

│

├── utils/

│   ├── __init__.py

│   ├── data_loader.py           # Veri yükleme fonksiyonları

│   ├── augmentation.py          # Data augmentation

│   └── visualization.py         # Görselleştirme fonksiyonları

│

├── videos/                      # Ham keypoint verileri

│   ├── abla/

│   ├── acele/

│   └── acikmak/

│

├── processed_data/              # İşlenmiş veri seti

│   ├── X_train.npy

│   ├── y_train.npy

│   ├── X_val.npy

│   ├── y_val.npy

│   ├── X_test.npy

│   ├── y_test.npy

│   └── metadata.pkl

│

├── checkpoints/                 # Model checkpoint'leri

│   ├── best_model.pth

│   └── ...

│

├── results/                     # Değerlendirme sonuçları

│   ├── confusion_matrix.png

│   ├── per_class_metrics.png

│   ├── confidence_distribution.png

│   └── evaluation_results.json

│

└── runs/                        # TensorBoard logları

    └── transformer_experiment_1/

```

---

## 8. Kullanım Talimatları

### 8.1 Adım Adım Çalıştırma

```bash

# 1. Repo'yu klonla

cdTID-N


# 2. Virtual environment oluştur

python3-mvenvvenv

sourcevenv/bin/activate


# 3. Bağımlılıkları kur

pipinstall-rrequirements.txt


# 4. Veriyi hazırla

pythonprepare_data.py

# Çıktı: processed_data/ klasörü oluşturulur


# 5. Model eğit

pythontrain.py

# Çıktı: checkpoints/best_model.pth kaydedilir


# 6. Modeli değerlendir

pythonevaluate.py

# Çıktı: results/ klasöründe görselleştirmeler


# 7. TensorBoard ile izle (opsiyonel)

tensorboard--logdir=runs

```

### 8.2 Hiperparametre Tuning

`config.py` veya `train.py` içinde ayarlanabilir:

```python

# Model boyutu

D_MODEL=256# 128, 256, 512

NUM_HEADS=8# 4, 8, 16

NUM_LAYERS=6# 3, 6, 9, 12

D_FF=1024# 512, 1024, 2048


# Training

BATCH_SIZE=32# 16, 32, 64

LEARNING_RATE=1e-4# 5e-5, 1e-4, 5e-4

DROPOUT=0.15# 0.1, 0.15, 0.2


# Data

SEQ_LEN=60# 30, 60, 90

```

---

## 9. Sonraki Adımlar ve İyileştirmeler

### 9.1 Model İyileştirmeleri

1.**Daha Fazla Kelime:** İlk 3'ten tüm 226 kelimeye genişletin

2.**Ensemble:** Birden fazla model'in tahminlerini birleştirin

3.**Temporal Attention Visualization:** Hangi frame'lere odaklanıyor?

4.**Multi-Modal:** Video + Optik Akış + Yüz ifadeleri

### 9.2 Deployment

1.**ONNX Export:** Model'i ONNX formatına çevirerek platform bağımsız hale getirin

2.**Quantization:** Model boyutunu küçültün (INT8)

3.**Real-Time Inference:** Webcam üzerinden gerçek zamanlı tahmin

4.**Web/Mobile App:** Flask/FastAPI ile API oluşturun

### 9.3 Veri Artırma

1.**Video Augmentation:** Hız değiştirme, perspektif dönüşümü

2.**Mixup/CutMix:** Farklı örnekleri karıştırın

3.**Synthetic Data:** GAN ile sentetik işaret dili videoları

---

## 10. Kaynaklar ve Referanslar

### 10.1 Transformer Makaleleri

1.**Attention Is All You Need** (Vaswani et al., 2017)

- Orijinal Transformer makalesi
- https://arxiv.org/abs/1706.03762

2.**Video Action Recognition Transformer** (VoVNet, 2021)

- Video sınıflandırma için Transformer
- https://arxiv.org/abs/2103.15691

3.**TimeSformer** (Facebook AI, 2021)

- Divided Space-Time Attention
- https://arxiv.org/abs/2102.05095

### 10.2 İşaret Dili Tanıma

1.**Sign Language Recognition Survey** (2020)

- İşaret dili tanıma teknikleri
- https://arxiv.org/abs/2008.09918

2.**MediaPipe Holistic** (Google, 2020)

- Keypoint extraction
- https://google.github.io/mediapipe/

### 10.3 Faydalı Linkler

- PyTorch Transformer Tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Papers with Code (Sign Language): https://paperswithcode.com/task/sign-language-recognition

---

## 11. Lisans ve Teşekkürler

Bu proje, Türk İşaret Dili (TİD) araştırmaları için geliştirilmiştir.

**Veri Seti:** TİD (Turkish Sign Language) Dataset

**Geliştirici:** [İsminiz]

**Tarih:** Ekim 2025

---

## 12. İletişim

Sorularınız veya katkılarınız için:

- Email: [email@example.com]
- GitHub Issues: [repo-link]

**Mutlu Kodlamalar! 🚀🤟**
