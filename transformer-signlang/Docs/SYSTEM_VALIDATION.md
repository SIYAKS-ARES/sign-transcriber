# 🛡️ Sistem Validation ve Error Prevention

Bu doküman, yaşanan sorunları ve bunları önlemek için eklenen güvenlik katmanlarını açıklar.

## 🔴 Yaşanan Kritik Sorunlar

### Sorun 1: MPS (Apple Silicon GPU) Desteği Eksikliği
**Belirti:**
```
🖥️  Device: cpu
```
M3 MacBook Pro olmasına rağmen CPU kullanılıyordu.

**Kök Neden:**
```python
# Sadece CUDA kontrolü yapılıyordu:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Etki:** 
- GPU'suz eğitim = 10-20x daha yavaş
- Gereksiz zaman kaybı

---

### Sorun 2: Class ID Mapping Hatası
**Belirti:**
```python
RuntimeError: index 5 is out of bounds for dimension 1 with size 3
```

**Kök Neden:**
- Veri: ClassId 1, 2, 5 (orijinal dataset)
- Model: 3 sınıf bekliyor → indeksler 0, 1, 2 olmalı
- Mapping yapılmamış → ClassId 5 direkt kullanılmış

**Etki:**
- Training crash
- Model çalışmıyor

---

## ✅ Eklenen Çözümler

### 1. Merkezi Utility Fonksiyonları

#### `utils/device_utils.py`
```python
from utils import get_device

# Otomatik device seçimi: CUDA > MPS > CPU
device, device_name = get_device()
# ✅ M3 Mac'te: device='mps'
# ✅ NVIDIA'da: device='cuda'
# ✅ Fallback: device='cpu'
```

**Özellikler:**
- ✅ CUDA desteği (NVIDIA GPU)
- ✅ MPS desteği (Apple Silicon M1/M2/M3)
- ✅ CPU fallback
- ✅ Detaylı bilgi yazdırma
- ✅ Otomatik uyumluluk kontrolü

#### `utils/class_utils.py`
```python
from utils import get_class_mapping, remap_labels, validate_class_mapping

# ClassId -> Index mapping
mapping = get_class_mapping([1, 2, 5])
# {1: 0, 2: 1, 5: 2}

# Label dönüşümü
labels = [1, 2, 5, 1]
remapped = remap_labels(labels, [1, 2, 5], to_index=True)
# [0, 1, 2, 0] ✅

# Validation
validate_class_mapping(remapped, [1, 2, 5], num_classes=3)
# ✅ veya ValueError
```

**Özellikler:**
- ✅ Bidirectional mapping (ClassId ↔ Index)
- ✅ Otomatik validation
- ✅ Detaylı hata mesajları
- ✅ Distribution printing

---

### 2. Güncellenmiş Scriptler

#### `scripts/03_normalize_data.py`
```python
# ÖNCE (HATA):
labels.append(class_id)  # 1, 2, 5 → Model crash!

# SONRA (DOĞRU):
from utils import get_class_mapping
mapping = get_class_mapping(config.TARGET_CLASS_IDS)
labels.append(mapping[class_id])  # 0, 1, 2 ✅
```

#### `train.py`
```python
# ÖNCE (CPU kullanıyordu):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SONRA (MPS destekli):
from utils import get_device
device, device_name = get_device(verbose=True)
# ✅ M3'te MPS kullanıyor!
```

#### `validate_setup.py`
```python
# Yeni kontroller eklendi:

1. Device Compatibility Check
   - ✅ CUDA availability
   - ✅ MPS availability (Apple Silicon)
   - ✅ CPU fallback
   - ✅ GPU memory info

2. Class Mapping Validation
   - ✅ Labels 0-indexed mi?
   - ✅ Ardışık mı? [0, 1, 2]
   - ✅ Sınıf sayısı doğru mu?
   - ✅ Mapping tutarlı mı?
```

---

### 3. Otomatik Validation Sistemi

#### Setup Validation
```bash
python validate_setup.py
```

**Kontrol Edilen:**
1. ✅ Python version (3.8+)
2. ✅ Dependencies (torch, mediapipe, etc.)
3. ✅ Project structure (files & dirs)
4. ✅ Configuration (config.py)
5. ✅ **Device compatibility** ← YENİ!
6. ✅ Data availability
7. ✅ **Class mapping** ← YENİ!

**Çıktı Örneği:**
```
🔢 CHECKING CLASS MAPPING
================================================================================
   Found labels: [0, 1, 2]
   Expected: [0, 1, 2, ...] (0-indexed)
   Target class IDs: [1, 2, 5]
   Class names: ['acele', 'acikmak', 'agac']
   ✅ Class mapping is correct:
      Label 0 → ClassId 1 (acele)
      Label 1 → ClassId 2 (acikmak)
      Label 2 → ClassId 5 (agac)

🎮 CHECKING DEVICE COMPATIBILITY
================================================================================
   ❌ CUDA not available
   ✅ MPS (Apple Silicon GPU) is available
      Device: M1/M2/M3 GPU detected
      ⚡ Metal Performance Shaders enabled
   ✅ CPU is always available (fallback)
   
   🎯 GPU acceleration available!
```

---

## 🔒 Error Prevention Matrix

| Potansiyel Hata | Önlem | Lokasyon |
|-----------------|-------|----------|
| CPU kullanımı (MPS varken) | `get_device()` MPS kontrolü | `utils/device_utils.py` |
| ClassId mapping hatası | `remap_labels()` otomatik dönüşüm | `utils/class_utils.py` |
| Label validation | `validate_class_mapping()` | `utils/class_utils.py` |
| pin_memory MPS hatası | `check_device_compatibility()` | `utils/device_utils.py` |
| 0-indexed olmayan labels | `validate_class_mapping()` raise error | `validate_setup.py` |
| Sınıf sayısı mismatch | Validation kontrol eder | `validate_setup.py` |

---

## 📋 Checklist: Yeni Script Yazarken

Her yeni script yazarken bu adımları takip edin:

### Device Selection
```python
- [ ] from utils import get_device kullan
- [ ] device, _ = get_device() ile device al
- [ ] Manuel CUDA/CPU kontrolü YAPMA
```

### Class Mapping
```python
- [ ] from utils import get_class_mapping, remap_labels
- [ ] ClassId'leri kullanmadan önce remap et
- [ ] Display'de original ClassId'yi göster
- [ ] Validation yap: validate_class_mapping()
```

### Data Loading
```python
- [ ] Labels'ı yükledikten sonra validate et
- [ ] print_class_distribution() ile dağılımı göster
- [ ] 0-indexed olduğunu doğrula
```

---

## 🧪 Test Senaryoları

### 1. Device Test
```bash
# Test 1: Device detection
python -c "from utils import print_device_info; print_device_info()"

# Beklenen (M3 Mac):
# ✅ MPS (Apple Silicon GPU) is available

# Test 2: Device usage
python -c "from utils import get_device; d, n = get_device(); print(f'Using: {n}')"

# Beklenen:
# Using: MPS (Apple Silicon GPU)
```

### 2. Class Mapping Test
```bash
# Test 1: Mapping
python -c "from utils import get_class_mapping; print(get_class_mapping([1,2,5]))"

# Beklenen:
# {1: 0, 2: 1, 5: 2}

# Test 2: Remapping
python -c "from utils import remap_labels; print(remap_labels([1,2,5], [1,2,5]))"

# Beklenen:
# [0 1 2]
```

### 3. Validation Test
```bash
# Full system check
python validate_setup.py

# Beklenen:
# 7/7 checks PASSED
```

---

## 📚 Dokümantasyon

- **`utils/README.md`**: Utility fonksiyonları detaylı kullanım
- **`SYSTEM_VALIDATION.md`** (bu dosya): Error prevention
- **`ilerleme.md`**: Bug fix history
- **`CALISTIRMA_REHBERI.md`**: Pipeline rehberi

---

## 🎯 Özet

### Önce (Sorunlu):
```python
❌ CPU kullanıyordu (MPS olmasına rağmen)
❌ ClassId mapping crash
❌ Her scriptte manuel kontrol
❌ Validation yok
```

### Sonra (Güvenli):
```python
✅ Otomatik device selection (CUDA/MPS/CPU)
✅ Merkezi class mapping utilities
✅ Otomatik validation
✅ Detaylı error messages
✅ Best practices enforcement
```

---

## 🚀 Kullanım

Yeni bir çalışmaya başlarken:

```bash
# 1. Sistemi validate et
python validate_setup.py

# 2. Data processing (utils kullanarak)
python scripts/03_normalize_data.py

# 3. Training (utils kullanarak)
python train.py

# 4. Evaluation (utils kullanarak)
python evaluate.py
```

Her adımda utils fonksiyonları otomatik olarak doğru device'ı seçer ve class mapping'i kontrol eder.

---

**🎉 Artık bu tür sorunlar tekrar yaşanmayacak!**

