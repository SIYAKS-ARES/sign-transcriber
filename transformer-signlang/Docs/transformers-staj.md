### Transformer-Tabanlı TİD Tanıma — 5 Günlük Staj Defteri

Bu rapor, `transformer-signlang` projesi kapsamında Transformer tabanlı işaret dili tanıma hattının (MediaPipe keypoint → Temporal Transformer → Değerlendirme/Görselleştirme) 5 günlük staj çalışmasını ayrıntılı biçimde anlatır. Kod referansları, komutlar ve örnek çıktı şemaları dahildir.

---

### 1) Proje Özeti ve Hedefler

- **Amaç**: MediaPipe Holistic ile çıkarılan 258‑boyutlu anahtar nokta dizilerinden Türk İşaret Dili (TİD) kelime sınıflandırması yapan bir Transformer modeli geliştirmek ve 226 kelimelik genişleme ile ölçeklemek.
- **Çıktılar**: Eğitim/değerlendirme scriptleri (`train.py`, `evaluate.py`), test videolarında uçtan uca çıkarım (`inference_test_videos.py`), attention görselleştirme (`visualize_attention.py`), kayıtlı sonuçlar (`results/`) ve checkpoint’ler (`checkpoints/`).

---

### 1.1) Deneme Aşamaları ve Süreler

- **Deneme 1 — 3 Kelime (POC):** `abla`, `acele`, `acikmak` sınıflarıyla uçtan uca hattın ilk doğrulaması yapıldı. Veri hazırlama ve eğitim hızla tamamlandı; mimari doğrulandı.
- **Deneme 2 — 10 Kelime:** Sınıf sayısı 10’a çıkarılarak model kapasitesi, label smoothing, dropout ve pooling stratejileri gözlemlendi; veri boru hattının ölçeklenebilirliği test edildi.
- **Deneme 3 — 226 Kelime (Tüm AUTSL):** Tüm sınıflar için tam ölçekli deney gerçekleştirildi.
  - Keypoint extraction (MediaPipe Holistic) süresi: **≈ 19 saat** (tüm videoların 258‑boyutlu anahtar nokta çıkarımı ve kaydı).
  - Model eğitimi süresi: **≈ 5 saat** (Base konfigürasyon; optimizasyon: AdamW + cosine annealing + warmup).
  - Not: Süreler donanım (GPU/CPU), disk I/O ve eşzamanlı iş sayısına göre değişebilir; raporlanan değerler bu kurulumda gözlemlenen yaklaşık sürelerdir.

### 2) Günlük Plan ve Çıktılar

#### Gün 1 — Literatür, Mimari ve Kurulum

- Transformer yaklaşımı: Uzun menzilli bağımlılıklar, paralel eğitim, attention ile yorumlanabilirlik.
- Girdi temsili: Pose(33×3=99) + Face(11×3=33) + Eller(2×21×3=126) = **258** özellik/frame.
- Model iskeleti: InputProjection → PositionalEncoding → N×TransformerEncoder → Pooling (GAP/CLS/Last) → Classifier.

Kod referansı (konfigürasyon ve hiperparametreler):
```28:58:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/config.py
class TransformerConfig:
    INPUT_DIM = 258
    MAX_SEQ_LENGTH = 200
    NUM_CLASSES = 226
    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    LABEL_SMOOTHING = 0.15
    POOLING_TYPE = 'gap'
```

Kurulum:
```bash
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/keypoints data/processed checkpoints results logs
```

#### Gün 2 — Veri Hazırlama: Seçim, Keypoint Çıkarımı, Normalizasyon

- Seçim CSV’leri oluşturma (train/val/test), MediaPipe ile 258‑boyutlu vektör çıkarma, scaler fit etme (yalnız train), pad/truncate ile sabit uzunluğa getirme.

Çerçeveden 258‑boyut çıkarımı:
```50:96:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/inference_test_videos.py
def extract_keypoints_from_frame(results):
    # Pose: 33×3, Face: 11×3 (seçilmiş kilit noktalar), Sol/sağ el: 21×3
    # Birleştir → 99 + 33 + 63 + 63 = 258
    ...
```

Normalizasyon ve pad/truncate:
```147:184:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/inference_test_videos.py
def normalize_sequence(sequence, scaler):
    return scaler.transform(sequence)

def pad_or_truncate_sequence(sequence, target_length):
    # Uzunsa son target_length; kısaysa başa sıfır pad
    ...
```

Çalıştırma (örnek):
```bash
python scripts/01_select_videos.py
python scripts/02_extract_keypoints.py
python scripts/03_normalize_data.py
```

Beklenen veriler: `data/scaler.pkl`, `data/processed/X_{train,val,test}.npy`, `y_{...}.npy`.

#### Gün 3 — Eğitim: Optimizasyon ve Takip

- Label smoothing’li CE loss, AdamW + Cosine annealing (warmup), gradient clipping, early stopping.
- Cihaz seçimi: CUDA > MPS > CPU; özet ve batch istatistikleri loglanır; en iyi model `checkpoints/best_model.pth`.

Eğitim döngüsü:
```423:471:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/train.py
def main():
    config = TransformerConfig()
    # Data yükleme → DataLoader
    model = TransformerSignLanguageClassifier(...)
    criterion = LabelSmoothingCrossEntropy(epsilon=config.LABEL_SMOOTHING)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    ...  # train_epoch / validate_epoch ve checkpoint kaydı
```

Komutlar:
```bash
python train.py                     # sıfırdan
python train.py --resume-from-best  # en iyi modelden devam
```

Örnek epoch özeti (konsol):
```text
📊 Epoch 12/100 Summary:
   Train Loss: 0.8421 | Train Acc: 0.7634
   Val   Loss: 0.9153 | Val   Acc: 0.7420 | Val F1: 0.7351
   Learning Rate: 0.000073
   ✅ Best model saved! (Val Acc: 0.7420)
```

#### Gün 4 — Değerlendirme ve Sonuçların Kaydı

- Test seti üzerinde accuracy, macro/micro F1, karışıklık matrisi, sınıf bazlı metrikler, confidence dağılımı; görseller `results/` altına kaydedilir.

Değerlendirme akışı:
```407:551:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/evaluate.py
def main():
    config = TransformerConfig()
    X_test = np.load(...)
    model = TransformerSignLanguageClassifier(...)
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    all_preds, all_probs, all_targets = evaluate_model(...)
    metrics = compute_metrics(all_targets, all_preds, config.CLASS_NAMES)
    save_results(metrics, config, config.RESULTS_DIR)
    # confusion_matrix_raw/normalized.png, per_class_metrics.png, prediction_confidence.png
```

Komut:
```bash
python evaluate.py
```

Beklenen kayıtlar: `evaluation_report.json`, `confusion_matrix_*.{csv,png}`, `per_class_metrics.{csv,png}`, `prediction_confidence.png`.

#### Gün 5 — Uçtan Uca Çıkarım ve Attention Görselleştirme

- Test videolarında uçtan uca: video → MediaPipe → normalizasyon → pad → Transformer → tahmin; interaktif oynatma, sonuç CSV/JSON.
- Attention analizleri: katman/başlık bazlı çoklu ısı haritaları, ortalama attention ve rollout.

Video çıkarımı:
```457:654:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/inference_test_videos.py
model, scaler, checkpoint = load_model_and_scaler(config, device)
for row in test_df.itertuples():
    result = process_and_display_video(...)
    # results/test_predictions.{csv,json}
```

Attention görselleştirme:
```382:522:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/transformer-signlang/visualize_attention.py
extractor = AttentionExtractor(model)
attention_weights = extractor.get_attention_weights(x, mask=None)
plot_multi_head_attention(...)
plot_averaged_attention(...)
plot_attention_rollout(...)
plot_attention_statistics(...)
```

Komutlar:
```bash
python inference_test_videos.py
python visualize_attention.py --num_samples 6
```

Örnek özet istatistik (konsol):
```text
📈 Genel Performans:
   - Toplam video: 150
   - Doğru tahmin: 118
   - Accuracy: 78.67%
🎯 Confidence İstatistikleri:
   - Ortalama: 0.73
   - Doğru/yanlış ayrımı belirgin, yanlışlarda ~0.48 ortalama
```

---

### 3) Teknik Notlar ve İyileştirme Önerileri

- Mask uyumluluğu: MPS üzerinde bazı mask sınırlamaları; eğitimde/çıkarımda gerekli yerlerde devre dışı bırakma iş akışları uygulanmıştır.
- Veri tarafı: Scaler yalnızca train setinde fit edilmeli; pad stratejisi (pre/post) kararlılık için sabit tutulmalı.
- Hiperparametre denemeleri: `D_MODEL/NHEAD/layers`, label smoothing, dropout ve pooling türü; sınıf dengesizliği için ağırlıklı loss veya focal loss alternatifleri.
- Gelecek iş: Multi‑scale temporal attention, self‑supervised pretraining, ONNX/TorchScript export, gerçek zamanlı pipeline optimizasyonu.

---

### 4) Hızlı Komut Özeti

```bash
cd transformer-signlang
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Veri hazırlama
python scripts/01_select_videos.py
python scripts/02_extract_keypoints.py
python scripts/03_normalize_data.py

# Eğitim
python train.py

# Değerlendirme
python evaluate.py

# Uçtan uca çıkarım ve görselleştirme
python inference_test_videos.py
python visualize_attention.py --num_samples 6
```

---

### 5) Sonuç

Beş gün sonunda, 258‑boyutlu MediaPipe anahtar noktalarıyla beslenen Temporal Transformer modeli; eğitim, değerlendirme, test videolarında çıkarım ve attention görselleştirme fonksiyonlarıyla birlikte uçtan uca çalışır hale getirilmiştir. `results/` klasörüne metrikler ve görseller kaydedilmekte; `checkpoints/` altında en iyi model saklanmaktadır. Ölçeklenebilirlik ve yorumlanabilirlik hedeflerine yönelik iyileştirme alanları belirlenmiştir.


