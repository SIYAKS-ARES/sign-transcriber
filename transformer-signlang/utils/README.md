# Utility Functions

Bu klasör, projedeki tekrarlayan sorunları önlemek için merkezi utility fonksiyonlarını içerir.

## 📂 Modüller

### 1. `device_utils.py` - Device/GPU Yönetimi

**Sorun:** MPS (Apple Silicon), CUDA ve CPU desteğinin her yerde manuel kontrolü

**Çözüm:** Merkezi device selection

```python
from utils import get_device, print_device_info

# Device seç (otomatik: CUDA > MPS > CPU)
device, device_name = get_device(verbose=True)
# Output: 
# 🖥️  Device: MPS (Apple Silicon GPU)
#    ⚡ Using Metal Performance Shaders

# Detaylı device bilgisi
print_device_info()
```

**Fonksiyonlar:**
- `get_device(verbose=True)`: En iyi device'ı otomatik seçer
- `print_device_info()`: Tüm device'ları listeler
- `check_device_compatibility(config)`: Config ile uyumluluğu kontrol eder

---

### 2. `class_utils.py` - Class ID Mapping

**Sorun:** ClassId (1,2,5) ile Label Index (0,1,2) arasında karışıklık

**Çözüm:** Merkezi mapping fonksiyonları

```python
from utils import get_class_mapping, remap_labels, validate_class_mapping
from config import TransformerConfig

config = TransformerConfig()

# ClassId -> Index mapping al
mapping = get_class_mapping(config.TARGET_CLASS_IDS)
# Output: {1: 0, 2: 1, 5: 2}

# Label'ları remap et
original_labels = [1, 2, 5, 1, 2]  # ClassId'ler
remapped = remap_labels(original_labels, config.TARGET_CLASS_IDS, to_index=True)
# Output: [0, 1, 2, 0, 1]

# Geri dönüştür
back = remap_labels(remapped, config.TARGET_CLASS_IDS, to_index=False)
# Output: [1, 2, 5, 1, 2]

# Validation
validate_class_mapping(remapped, config.TARGET_CLASS_IDS, config.NUM_CLASSES)
# Hata varsa ValueError raise eder
```

**Fonksiyonlar:**
- `get_class_mapping(target_class_ids)`: ClassId -> Index mapping
- `get_reverse_mapping(target_class_ids)`: Index -> ClassId mapping
- `remap_labels(labels, target_class_ids, to_index=True)`: Label dönüşümü
- `get_original_class_id(index, target_class_ids)`: Index'ten ClassId
- `validate_class_mapping(...)`: Mapping doğrulama
- `print_class_distribution(...)`: Güzel formatlı dağılım yazdırma

---

## 🚀 Kullanım Örnekleri

### Data Processing Script'lerinde

```python
# scripts/03_normalize_data.py
from config import TransformerConfig
from utils import get_class_mapping

config = TransformerConfig()
class_id_to_idx = get_class_mapping(config.TARGET_CLASS_IDS)

for video in videos:
    original_class_id = video['class_id']  # 1, 2 veya 5
    label = class_id_to_idx[original_class_id]  # 0, 1 veya 2
    labels.append(label)
```

### Training Script'lerinde

```python
# train.py
from utils import get_device

device, device_name = get_device(verbose=True)
model = model.to(device)

# MPS için pin_memory otomatik düzeltilir
```

### Evaluation Script'lerinde

```python
# evaluate.py
from utils import get_original_class_id, print_class_distribution
from config import TransformerConfig

config = TransformerConfig()

# Predictions'ları orijinal class ID'lere dönüştür
for idx in predictions:
    class_id = get_original_class_id(idx, config.TARGET_CLASS_IDS)
    print(f"Predicted: {config.CLASS_NAMES[idx]} (ClassId: {class_id})")

# Dağılımı yazdır
print_class_distribution(y_test, config.TARGET_CLASS_IDS, 
                        config.CLASS_NAMES, split_name="TEST")
```

---

## ✅ Validation

Tüm sistem kontrollerini yapmak için:

```bash
python validate_setup.py
```

Bu script şunları kontrol eder:
- ✅ Python version (3.8+)
- ✅ Dependencies (torch, mediapipe, vb.)
- ✅ Project structure
- ✅ Configuration
- ✅ **Device compatibility (CUDA/MPS/CPU)** ← YENİ!
- ✅ Data availability
- ✅ **Class mapping validation** ← YENİ!

---

## 🔧 Best Practices

### 1. Her Zaman Utils Kullan

❌ **Kötü:**
```python
# Her script'te tekrar et
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Manual mapping
if class_id == 1:
    label = 0
elif class_id == 2:
    label = 1
# ...
```

✅ **İyi:**
```python
from utils import get_device, get_class_mapping

device, _ = get_device()
class_mapping = get_class_mapping(config.TARGET_CLASS_IDS)
label = class_mapping[class_id]
```

### 2. Validation Yap

Her data processing sonrası:
```python
from utils import validate_class_mapping

validate_class_mapping(labels, config.TARGET_CLASS_IDS, config.NUM_CLASSES)
# ValueError raise ederse sorun var
```

### 3. Setup Validation Çalıştır

Yeni environment'ta ilk iş:
```bash
python validate_setup.py
```

---

## 🐛 Önlenen Sorunlar

Bu utils sayesinde artık şu hatalar olmayacak:

| Hata | Neden | Çözüm |
|------|-------|-------|
| `index 5 is out of bounds for dimension 1 with size 3` | ClassId mapping yapılmamış | `remap_labels()` kullan |
| CPU kullanıyor (MPS olmasına rağmen) | MPS kontrolü eksik | `get_device()` kullan |
| `pin_memory warning on MPS` | MPS pin_memory desteklemiyor | `check_device_compatibility()` otomatik düzeltir |
| Label validation hatası | 0-indexed kontrol yok | `validate_class_mapping()` kullan |

---

## 📊 Test

Utils'leri test etmek için:

```bash
cd transformer-signlang
python -c "from utils import *; print_device_info()"
python -c "from utils import *; print(get_class_mapping([1,2,5]))"
```

