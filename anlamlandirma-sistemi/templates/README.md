# Flask Templates Rehberi (`anlamlandirma-sistemi/templates/`)

Bu klasör, `anlamlandirma-sistemi/app.py` içindeki route’lara karşılık gelen Flask HTML şablonlarını içerir.

Amaç:
- Demo/deney ekranlarının hangi endpoint’lerle çalıştığını belgelemek
- UI akışını hızlıca anlamak

---

## Dosyalar ve İlişkili Route’lar

| Template | Route | Açıklama |
|----------|-------|----------|
| `base.html` | (layout) | Ortak layout: CSS/JS include, navbar, container |
| `index.html` | `GET /` | Ana giriş sayfası |
| `demo.html` | `GET /demo`, `POST /translate` | Video yükleme + model/LLM çıktısı demo ekranı |
| `test_video.html` | `GET /test-video` | Basit video oynatma test sayfası |
| `experiments.html` | `GET /experiments` | 3/4/5 kelimelik deneyleri UI üzerinden tetikleme |

---

## UI Akışı (Özet)

### 1) Demo Akışı

1. Kullanıcı `GET /demo` sayfasına gider.
2. Video yükleme/işleme aksiyonları genelde şu endpoint’lere bağlanır:
   - `POST /api/process_video` (video → frame → model → opsiyonel LLM)
   - `POST /api/test_model` (sadece model çıktısı)
3. Sonuçlar ekranda gösterilir ve işlem `anlamlandirma.db` içine kaydedilir.

### 2) Experiments Akışı

1. Kullanıcı `GET /experiments` sayfasına gider.
2. UI, deneyleri şu endpoint’lerle tetikler:
   - `POST /api/run_experiment`
   - `POST /api/run_all_experiments`
3. Sonuçlar JSON formatında dönüp UI’da tablo/özet olarak gösterilir.

---

## Notlar

- Stil/JS dosyaları: `anlamlandirma-sistemi/static/`
- API dönüş formatı ve kayıt/istatistik endpoint’leri için: `anlamlandirma-sistemi/README.md` ve `database.py`

---

**Son Güncelleme:** 2026-01-21

