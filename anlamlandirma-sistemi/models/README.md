# Models Rehberi (`anlamlandirma-sistemi/models/`)

Bu klasör, `anlamlandirma-sistemi` içinde kullanılan **eğitilmiş model dosyalarını** içerir. Bu dosyalar genelde büyük (binary) olduğu için sürümleme ve kullanım şekli açık olmalıdır.

---

## Dosyalar

| Dosya | Tür | Amaç |
|------|-----|------|
| `best_model.pth` | PyTorch checkpoint | Transformer tabanlı işaret sınıflandırma modeli (en iyi checkpoint) |
| `labels.csv` | CSV | Model sınıf etiketleri / isimleri (id → label) |
| `sign_classifier_haveface.h5` | Keras / TF model | Yüz keypoint’leri dahil varyant (legacy/alternatif) |
| `sign_classifier_NoneFace_best_89225.h5` | Keras / TF model | Yüz olmadan varyant (legacy/alternatif) |

> Not: Güncel production akışında ana kullanım PyTorch Transformer tarafındadır (`best_model.pth`). Keras `.h5` modelleri eski/alternatif denemeler olarak tutuluyor olabilir.

---

## Uygulamada Kullanım

Model yükleme ve inference akışı:
- `anlamlandirma-sistemi/local_model_handler.py`
  - MediaPipe keypoint çıkarımı
  - Normalizasyon (scaler)
  - Sekans padding/truncation
  - Model inference

UI/API üzerinden test:
- `POST /api/test_model` (LLM olmadan sadece model çıktısı)
- `POST /api/process_video` (model + opsiyonel LLM/RAG)

---

## Model Dosyası Notları

- `.pth` dosyası tipik olarak şu bilgileri içerir:
  - `model_state_dict`
  - eğitim epoch’u ve validation metrikleri (varsa)
  - config snapshot (varsa)

- Eğer farklı bir checkpoint ile denemek isterseniz:
  - `local_model_handler.py` içindeki load path’i güncellenir.

---

## Versiyonleme Önerisi

Bu klasörde yeni modeller eklenecekse şu şema önerilir:

```
best_model_<dataset>_<date>.pth
labels_<dataset>_<date>.csv
```

Örn:
- `best_model_autsl226_2026-01-21.pth`
- `labels_autsl226_2026-01-21.csv`

Ve bu README’ye kısa bir “Model Changelog” bölümü eklenebilir:
- Eğitim verisi değişti mi?
- Sınıf sayısı değişti mi?
- Accuracy/F1 kaç?

---

## Depolama / Git Notu

Model dosyaları büyük olabileceği için:
- `.gitignore` ile takip dışı bırakma
- veya Git LFS kullanma

