### Anlamlandırma Sistemi Staj Defteri (3 Gün)

Bu rapor, `msh-sign-language-tryouts/anlamlandirma-sistemi` klasöründe geliştirilen Türk İşaret Dili (TİD) → Türkçe anlamlandırma sistemine yönelik 3 günlük staj çalışmasını kapsamlı ve kanıtlarla (kod parçaları, komutlar, çıktı örnekleri) belgelendirir.

---

### 1) Proje Özeti ve Amaç

- **Amaç**: TİD işaret videosundan MediaPipe Holistic ile anahtar noktaları çıkarıp, BiLSTM + Attention tabanlı bir sınıflandırıcıyla TİD token(ları) elde etmek; ardından regex + RAG ile ön‑işleme sonrasında LLM ile akıcı Türkçe cümlelere dönüştürmek.
- **Çıktı**: Web arayüzü ve REST API üzerinden video yükleme/kamera frame akışı → TİD token → Türkçe çeviri, güven puanı, kısa açıklama.

---

### 2) Klasör Yapısı ve Önemli Dosyalar

- `app.py`: Flask uygulaması, HTML sayfaları ve REST API uçları (`/translate`, `/api/process_frames`, `/api/process_video`, `/api/test_model`).
- `local_model_handler.py`: Model mimarisi (BiLSTM+Attention), MediaPipe ile özellik çıkarımı (282 boyut/frame), sekans hazırlama (60 frame), tahmin mantığı (top‑1/top‑5).
- `preprocessor.py`: Regex tabanlı metin normalizasyonu ve dinamik RAG bağlamı; LLM için prompt şablonu.
- `llm_services.py`: LLM sağlayıcıları (OpenAI/Claude/Gemini) için istemci çağrıları ve yapılandırılmış çıktı ayrıştırma.
- `templates/`, `static/`: Web arayüzü (HTML/CSS/JS).
- `models/`: Ağırlık dosyaları (`sign_classifier_NoneFace_best_89225.h5`) ve etiketler (`labels.csv`).
- `test_videos/`: Örnek test videoları (`acikmak_*.mp4`).

---

### 3) Gün 1 — Literatür Taraması, Mimari Analiz ve Kurulum

#### 3.1 Literatür ve Yaklaşım

- İşaret Dili Tanıma hattı: Video → Anahtar Noktalar (Pose/Hands/Face) → Zaman Serisi Modeli → TİD token(ları) → Kural‑tabanlı düzenleme → Doğal dil.
- MediaPipe Holistic ile: Pose 33×4, Ellerde 2×(21×3), Azaltılmış yüz 8×3 → toplam 282 özellik/frame.
- Zaman serisi modeli: Masking + LayerNorm → TimeDistributed Dense(256,128) → Multi‑Head Attention → BiLSTM(128,64) → Global Max/Average Pooling → Dense(128,64) → Softmax(226).
- TİD → Türkçe: Negasyon ve tekrarın işlevsel yorumlanması; birleşik tokenların birleştirilmesi; LLM ile doğal Türkçe cümle üretimi.

#### 3.2 Kod Keşfi — Öne Çıkan Parçalar

Model mimarisi özeti:
```12:36:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/local_model_handler.py
def create_enhanced_model(seq_len: int, feature_dim: int, num_classes: int):
    if tf is None:
        raise RuntimeError("TensorFlow yüklenemedi. Apple Silicon için tensorflow-macos + tensorflow-metal kurulmalı.")

    inputs = layers.Input(shape=(seq_len, feature_dim))

    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LayerNormalization()(x)

    x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Dropout(0.3)(x)

    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Dropout(0.3)(x)

    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.4,
            recurrent_dropout=0.3,
            recurrent_regularizer=regularizers.l2(1e-4),
        )
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.4,
            recurrent_dropout=0.3,
            recurrent_regularizer=regularizers.l2(1e-4),
        )
    )(x)

    max_pool = layers.GlobalMaxPooling1D()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([max_pool, avg_pool])

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model
```

Özellik çıkarımı ve sekans hazırlama:
```152:171:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/local_model_handler.py
def frames_to_keypoint_sequence(frames: List[np.ndarray]) -> np.ndarray:
    """BGR frame listesi -> (T, 282) anahtar nokta dizisi.
    Pose(33*4) + Reduced Face(8*3) + Hands(2*21*3) sırayla birleştirilir.
    """
    if mp is None:
        raise RuntimeError("MediaPipe kurulmamış. 'mediapipe' bağımlılığını ekleyin.")

    print(f"MediaPipe işleme başlıyor: {len(frames)} frame")
    ...
```

LLM entegrasyonu ve yapılandırılmış çıktı ayrıştırma:
```34:51:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/llm_services.py
def _translate_with_openai(prompt: str) -> dict:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"OpenAI kütüphanesi yüklenemedi: {exc}")

    client = OpenAI()
    model = os.environ.get('OPENAI_MODEL', 'gpt-4o')
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = (response.choices[0].message.content or "").strip()
    return parse_structured_output(text)
```

Flask API uçları:
```76:124:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/app.py
@app.route('/api/process_frames', methods=['POST'])
def process_frames():
    """Kamera veya video frame'lerini işler ve transkripsiyon üretir."""
    try:
        data = request.get_json()
        frames_data = data.get('frames', [])
        provider = data.get('provider', 'openai')
        ...
        transcription = process_frame_sequence(frames)
        ...
        result_data = translate_with_llm(provider, final_prompt)
        return jsonify({
            'success': True,
            'original_transcription': transcription,
            'processed_transcription': processed_transcription,
            'result': result_data
        })
```

#### 3.3 Kurulum

Ortam: `anlamlandirma` conda ortamı.

```bash
conda activate anlamlandirma
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi
pip install -r requirements.txt
```

Apple Silicon için TensorFlow hızlandırma:

```text
requirements.txt
  tensorflow-macos>=2.15; sys_platform == "darwin"
  tensorflow-metal>=1.1; sys_platform == "darwin"
```

LLM API anahtarları (3. gün entegrasyon):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

---

### 4) Gün 2 — Model Boru Hattı Denemeleri ve Değerlendirme

#### 4.1 Test Scripti ile Toplu Değerlendirme

Komut:

```bash
conda activate anlamlandirma
cd /Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi
python test_none_face_model.py
```

Örnek konsol çıktısı (özet):

```text
Toplam sınıf sayısı: 226
Bulunan video sayısı: 5

==================================================
Video: acikmak_1.mp4
Gerçek label: acikmak
Keypoint'ler çıkarılıyor...
Input shape: (1, 60, 282)

Top-5 Tahminler:
  1. ACIKMAK           (0.8123)
  2. ...

En yüksek tahmin: ACIKMAK (0.8123)
Doğru mu: ✓ EVET

==================================================
TEST SONUÇLARI
Toplam video: 5
Doğru tahmin: 4
Doğruluk oranı: 80.00%
```

Notlar:
- Sequence uzunluğu her video için `uniform_sample_or_pad` ile 60 frame’e sabitlenir.
- Özellik boyutu 282 = 132 (pose) + 24 (azaltılmış yüz) + 126 (eller).
- Doğruluk video ismine gömülü gerçek etiket ile karşılaştırılır.

#### 4.2 Gerçek Zamanlı İzleme (İsteğe Bağlı)

```bash
python realtimeSimule_test.py
```

Beklenen: Canlı pencerede top‑5 tahmin ve güven değerleri; 0.7 üzeri güvenlerde overlay metin.

#### 4.3 Flask API ile Model Testi (LLM Yok)

Sunucuyu başlat:

```bash
python app.py
```

Endpoint denemesi:

```bash
curl -F "video=@test_videos/acikmak_1.mp4" http://localhost:5005/api/test_model | jq
```

Örnek JSON:

```json
{
  "success": true,
  "frame_count": 60,
  "model_result": {
    "pred_id": 12,
    "pred_name": "ACIKMAK",
    "confidence": 0.81,
    "top5": [
      {"id": 12, "name": "ACIKMAK", "confidence": 0.81},
      {"id": 98, "name": "...", "confidence": 0.06},
      {"id": 7,  "name": "...", "confidence": 0.04},
      {"id": 21, "name": "...", "confidence": 0.03},
      {"id": 55, "name": "...", "confidence": 0.02}
    ],
    "threshold_met": true
  },
  "raw_transcription": "ACIKMAK"
}
```

---

### 5) Gün 3 — LLM Entegrasyonu ve Uçtan Uca Demo

#### 5.1 Regex + RAG Ön‑İşleme

Kurallar ve örnekler:

```4:24:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/preprocessor.py
def preprocess_text_for_llm(transcription: str) -> str:
    # 1) Birleşik kelimeleri işle: ARABA^SÜRMEK -> ARABA_SÜRMEK
    processed = transcription.replace('^', '_')
    # 2) Tekrar/süreç: GİTMEK GİTMEK -> GİTMEK(süreç/ısrar)
    processed = re.sub(r"\b(GİTMEK)\s+\1\b", r"\1(süreç/ısrar)", processed)
    # 3) Negasyon: X DEĞİL/YOK -> X (negasyon:...)
    processed = re.sub(r"(\w+)\s+(DEĞİL|YOK)\b", r"\1 (negasyon:\2)", processed)
    processed = re.sub(r"\s+", " ", processed).strip()
    return processed
```

Dinamik RAG bağlamı ve nihai prompt:

```34:87:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/preprocessor.py
def create_final_prompt(processed_transcription: str) -> str:
    # RAG_KNOWLEDGE_BASE içeriğine göre ek bağlam üretir
    # Persona + few-shot + çıktı şablonu ile birleşik prompt döner
    ...
```

#### 5.2 LLM Sağlayıcı Çağrıları

OpenAI örneği ve ayrıştırıcı:

```5:31:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/llm_services.py
def parse_structured_output(llm_response: str) -> dict:
    # "Çeviri:", "Güven: N/10", "Açıklama:" başlıklarını ayrıştırır
    ...
```

Çalıştırma:

```bash
export OPENAI_API_KEY=...
python app.py
# Arayüz: http://localhost:5005/demo
```

#### 5.3 Uçtan Uca Video → Çeviri

API üzerinden video ile tam akış:

```bash
curl -F "provider=openai" -F "video=@test_videos/acikmak_1.mp4" \
  http://localhost:5005/api/process_video | jq
```

Beklenen JSON (şematik):

```json
{
  "success": true,
  "original_transcription": "ACIKMAK",
  "processed_transcription": "ACIKMAK",
  "result": {
    "translation": "Açım.",
    "confidence": 9,
    "explanation": "Eylem isim olarak değil, durum ifadesi olarak yorumlandı."
  }
}
```

Arayüzden deneme (`/demo`): örnek transkripsiyon regex+RAG sonrası LLM’e gider ve üçlü çıktı gösterilir.

---

### 6) Bulgular, Sorunlar ve Çözüm Önerileri

- **Bulgular**
  - 60×282 giriş ile model tutarlı tahminler verdi; top‑5 listesi yararlı hata ayıklama sağladı.
  - Negasyon/tekrar kuralları çeviride akıcılığa katkı sağladı.
  - LLM yanıt formatı ayrıştırıcısı, sapmaları güvenli şekilde ele aldı.

- **Sorunlar**
  - MediaPipe kurulum/izin sorunları (macOS) ve düşük el/pose tespiti.
  - Uzun videolarda performans; bu nedenle 5 fps örnekleme ve 120 frame üst sınırı uygulandı.
  - LLM API anahtar/yetki hataları; sağlayıcıya göre değişen SDK davranışları.

- **Öneriler**
  - `RAG_KNOWLEDGE_BASE` kapsamını genişletmek; çok‑anlamlı TİD işaretleri için örnekler eklemek.
  - Eşik uyarlaması ve top‑k yeniden yazımı ile model+LLM birleşik güven değerlendirmesi.
  - Kayan pencere ile ardışık tahminleri birleştirip cümle seviyesinde daha stabil transkripsiyon.
  - Veri artırma ve sınıf dengesini iyileştirerek yeniden eğitim.

---

### 7) Sonuç

Üç gün sonunda, video → anahtar nokta → sekans → sınıflandırma → regex+RAG → LLM çeviri hattı uçtan uca çalıştırıldı. Web arayüzü ve REST API ile hem tekil testler hem de uçtan uca denemeler yapıldı. Çeviri çıktıları güven puanı ve açıklama ile raporlanarak değerlendirildi; iyileştirme alanları notlandı.


