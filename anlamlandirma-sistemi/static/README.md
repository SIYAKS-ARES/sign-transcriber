# Static Assets Rehberi (`anlamlandirma-sistemi/static/`)

Bu klasör, Flask web arayüzünün statik dosyalarını (CSS/JS) içerir.

---

## Dosyalar

| Dosya | Amaç |
|------|------|
| `style.css` | UI stil kuralları (layout, butonlar, renkler) |
| `main.js` | Frontend etkileşimleri (video yükleme, API çağrıları, sonuç gösterimi) |

---

## İlişkili Şablonlar

Bu dosyalar tipik olarak şu template’ler tarafından kullanılır:
- `templates/base.html` (global include)
- `templates/demo.html` (video → API akışı)
- `templates/experiments.html` (deney API çağrıları)

---

## API Entegrasyonu (Özet)

`main.js` üzerinden çağrılan başlıca endpoint’ler:
- `POST /api/process_video`
- `POST /api/test_model`
- `GET /api/history`
- `POST /api/run_experiment`
- `POST /api/run_all_experiments`

Bu endpoint’lerin sunucu tarafı implementasyonu: `anlamlandirma-sistemi/app.py`

---

## Notlar

- UI davranışlarını değiştirirken önce `templates/demo.html` / `templates/experiments.html` ile birlikte kontrol edin.
- API kota sorunları için uygulamada “LLM kapalı” mod bulunur (`use_llm=false` gibi). Detay: `HIZLI_TEST.md`

---

**Son Güncelleme:** 2026-01-21

