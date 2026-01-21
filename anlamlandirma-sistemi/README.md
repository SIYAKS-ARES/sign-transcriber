# Anlamlandirma Sistemi

Isaret dili videolarindan (kamera/video) **transkripsiyon (gloss/etiket)** ureten ve bunu opsiyonel olarak **LLM** ve/veya **RAG (Retrieval-Augmented Generation)** ile Turkce'ye ceviren Flask tabanli bir uygulama.

Bu klasor, tek bir uygulama altinda uc bileseni birlestirir:

- **Gorsel tanima (Transformer model)**: Video frame'lerinden MediaPipe keypoint cikarma + PyTorch Transformer siniflandirici ile kelime/etiket tahmini
- **Metin isleme + LLM ceviri**: Transkripsiyonu LLM'e uygun hale getirme, coklu alternatifli ciktiyi parse etme
- **RAG entegrasyonu (opsiyonel)**: `rag/` modulu ile sozluk + ceviri hafizasi tabanli baglamsal uretim

---

## Hizli Baslangic

### 1) Ortami hazirla

```bash
cd anlamlandirma-sistemi
pip install -r requirements.txt
```

### 2) Ortam degiskenlerini ayarla

Proje kokundeki `.env` dosyasindan okunur (uygulama `load_dotenv()` cagirir).

Ornek:

```bash
GEMINI_API_KEY=...
GEMINI_API_KEY_2=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

Not: `llm_services.py` Gemini icin birden fazla anahtari otomatik rotasyonla kullanabilir (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, ...).

### 3) Flask uygulamasini calistir

```bash
python app.py
```

Varsayilan:
- Host: `0.0.0.0` (env `HOST` ile degisir)
- Port: `5005` (env `PORT` ile degisir)
- Debug: `FLASK_DEBUG=1` ise acik

Tarayici:
- Ana sayfa: `http://localhost:5005/`
- Demo: `http://localhost:5005/demo`
- Deneyler: `http://localhost:5005/experiments`

---

## Sistem Akisi (High-Level)

1. **Video yukleme / kamera frame akisi** gelir.
2. Video, belirli FPS ile **frame**'lere ayrilir (`extract_frames_from_video`, varsayilan `target_fps=5`).
3. Frame'lerden **MediaPipe Holistic** ile **258 boyutlu keypoint** dizisi uretilir.
4. Keypoint dizisi **Transformer** modeline girer ve **etiket/transkripsiyon** uretilir.
5. Transkripsiyon:
   - **RAG varsa**: `rag/` pipeline'i ile sozluk + hafiza baglami ile prompt olusturulur ve LLM'e gonderilir.
   - **RAG yoksa**: Basit fallback preprocessing + prompt kullanilir.
6. LLM cevabi, **3 alternatif** formati destekleyecek sekilde parse edilir.
7. Sonuclar SQLite DB'ye kaydedilir (`anlamlandirma.db`).

---

## Proje Yapisi

```
anlamlandirma-sistemi/
  app.py                 # Flask UI + API endpoint'leri
  requirements.txt        # Bagimliliklar
  database.py             # SQLite (anlamlandirma.db) kayitlari
  llm_services.py         # OpenAI/Claude/Gemini adaptoru + cikti parser
  preprocessor.py         # RAG'e baglanan preprocessing ve prompt olusturma (fallback dahil)
  local_model_handler.py  # Transformer model + MediaPipe keypoint cikarma

  templates/              # Flask HTML sayfalari
  static/                 # JS/CSS
  test_videos/            # Ornek test videolari

  rag/                    # RAG modulu (sozluk + hafiza + prompt + llm + pipeline)
  vectorstore/            # ChromaDB verileri (rag icin)

  experiments/            # 3/4/5 kelimelik ceviri deneyleri + raporlama
  DEMO_CALISTIRMA.md      # Demo calistirma rehberi
  DENEY_KILAVUZU.md       # Deney (benchmark) rehberi
  RAG_ENTEGRASYON.md      # RAG entegrasyonu teknik dokumani
```

---

## Web UI

### Demo (`/demo`)

Amac:
- Video yukle
- Model tahminini gor
- (Opsiyonel) LLM ve/veya RAG ile Turkce ceviri al
- Sonuclari kaydet

Detayli adimlar icin `DEMO_CALISTIRMA.md` ve `HIZLI_TEST.md` dosyalarina bakin.

### Experiments UI (`/experiments`)

Amac:
- 3/4/5 kelimelik test setleri ile toplu deney calistirmak
- RAG acik/kapali, provider secimi, limit gibi parametrelerle karsilastirma yapmak

Detayli rehber: `DENEY_KILAVUZU.md`

---

## API Endpoint'leri

### Video / Frame isleme

- `POST /api/process_video`
  - Form-data: `video` (file), `provider` (default: gemini), `use_llm` (true/false)
  - Cikti: `original_transcription`, `processed_transcription`, `result`, `record_id`

- `POST /api/process_frames`
  - JSON: `{ "frames": ["data:image/...base64", ...], "provider": "openai|gemini|claude" }`

- `POST /api/test_model`
  - Sadece model tahmini (LLM yok)

### Kayit ve istatistik

- `GET /api/history?limit=10`
  - Son islenen kayitlar + istatistikler

### RAG durumu ve ceviri

- `GET /api/rag_status`
- `POST /api/translate_rag`
  - JSON: `{ "gloss": "BEN OKUL GITMEK" }`

### Human-in-the-Loop (HITL)

- `POST /api/approve_translation`
  - JSON: `{ "gloss": "...", "translation": "...", "reference": "..." }`
  - Onaylanan ceviri `tid_hafiza` koleksiyonuna kaydedilir.

### Kota/Rate-limit workaround (manuel LLM cevabi)

- `POST /api/get_prompt`
  - JSON: `{ "gloss": "...", "use_rag": true }`
  - Uretilen promptu dondurur (LLM'e manuel yapistir-kopyala icin)

- `POST /api/process_manual_response`
  - JSON: `{ "gloss": "...", "reference": "...", "llm_response": "...", "provider": "..." }`
  - Manuel girilen LLM yanitini parse eder ve deney formatina uygun sonuc uretir.

---

## LLM Entegrasyonu

`llm_services.py` su saglayicilari destekler:

- **OpenAI** (env: `OPENAI_API_KEY`, opsiyonel `OPENAI_MODEL`, varsayilan `gpt-4o`)
- **Anthropic Claude** (env: `ANTHROPIC_API_KEY`, opsiyonel `ANTHROPIC_MODEL`)
- **Gemini** (env: `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, opsiyonel `GEMINI_MODEL`)

LLM cikti formati:
- Tercihen **3 alternatif** format (Alternatif 1/2/3 + Guven + Aciklama)
- Fallback olarak tek ceviri formatini da parse eder.

---

## RAG Entegrasyonu

Bu projede RAG modulu `rag/` altinda gomulu gelir. Ayrintili teknik dokuman:
- `RAG_ENTEGRASYON.md`
- `RAG_GUIDE.md`

RAG'nin kullandigi vektorstore:
- `vectorstore/` (ChromaDB)

Vectorstore hazirlama/kontrol:
- `scripts/init_vectorstore.py` (bkz. `RAG_ENTEGRASYON.md`)

Not: RAG baslatilamazsa `preprocessor.py` otomatik olarak **fallback** moduna gecer (basit preprocessing + prompt).

---

## Deneyler (Benchmark)

CLI ile:

```bash
python experiments/run_experiments.py --all
python experiments/run_experiments.py --words 3 --limit 5
python experiments/run_experiments.py --all --output results.json --report benchmark.md
```

Test setleri:
- `experiments/test_sets/3_word_glosses.json`
- `experiments/test_sets/4_word_glosses.json`
- `experiments/test_sets/5_word_glosses.json`

Detayli rehber: `DENEY_KILAVUZU.md`

---

## Veritabani (SQLite)

Uygulama, isleme kayitlarini `anlamlandirma.db` icinde saklar.

Tablo: `video_records`
- `filename`, `filesize`, `duration`
- `upload_time`, `process_time`
- `transcription`, `translation`, `confidence`, `provider`
- `status`, `error_message`

Hizli kontrol:

```bash
python database.py
```

---

## Model Notlari (Transformer + MediaPipe)

Transformer bileseni `local_model_handler.py` icinde yonetilir:
- MediaPipe Holistic ile 258 boyutlu vektor: Pose(99) + Face(33) + Hands(126)
- Sekans uzunlugu: 200 frame (padding/truncation)
- Checkpoint ve scaler, repo yapisinda `transformer-signlang/` altindan okunur.

Detayli demo notlari:
- `DEMO_CALISTIRMA.md`
- `HIZLI_TEST.md`

---

## SSS / Sorun Giderme

### RAG calismiyor (fallback modunda)
- `vectorstore/` mevcut mu kontrol edin.
- Gerekirse `TRANSKRIPSIYON-RAG-VDB` altindaki vectorstore'u buraya kopyalayin (bkz. `RAG_ENTEGRASYON.md`).

### Gemini kota/rate limit hatalari
- `.env` icinde birden fazla anahtar tanimlayin: `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, ...
- `llm_services.py` otomatik anahtar rotasyonu yapar.
- Alternatif olarak manuel prompt/yanit endpoint'lerini kullanin: `/api/get_prompt`, `/api/process_manual_response`

---

## Dokuman Haritasi

- Calistirma: `DEMO_CALISTIRMA.md`, `HIZLI_TEST.md`
- Deney/benchmark: `DENEY_KILAVUZU.md`, `CLI-test-results-*.md`
- RAG: `RAG_ENTEGRASYON.md`, `RAG_GUIDE.md`, `rag_regex_overview.md`
- KB: `knowledge_base/README.md`

