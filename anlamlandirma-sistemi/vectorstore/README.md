# Vectorstore Rehberi (`anlamlandirma-sistemi/vectorstore/`)

Bu klasör, `anlamlandirma-sistemi/rag` modülünün kullandığı **ChromaDB persistent storage** dizinidir.

İçerik:
- `chroma.sqlite3`: ChromaDB metadata / koleksiyon bilgileri
- UUID klasörleri (`*/data_level0.bin`, `header.bin`, vb.): vektör index dosyaları

---

## Koleksiyonlar

Varsayılan koleksiyon isimleri (`rag/config.py`):
- `tid_sozluk`: Statik sözlük koleksiyonu (kelime → anlam/örnek)
- `tid_hafiza`: Dinamik çeviri hafızası (cümle → çeviri örnekleri)

---

## Oluşturma / Kontrol

Uygulama içinde kontrol:
- Başlangıçta `app.py` RAG durumunu kontrol eder (`/api/rag_status`).

CLI ile kontrol:

```bash
cd anlamlandirma-sistemi
python scripts/init_vectorstore.py --check
python scripts/init_vectorstore.py --stats
```

Sıfırdan oluşturma (kaynak veri gerekir):

```bash
python scripts/init_vectorstore.py --init
```

---

## Önemli Notlar

- Bu dizin **binary** içerir ve büyük olabilir; genelde “data artifact” olarak düşünülmelidir.
- Yedek/taşıma için tüm `vectorstore/` klasörünü kopyalamak yeterlidir.
- Yol override:
  - `VECTORSTORE_PATH` env değişkeni ile farklı bir dizin kullanılabilir (bkz. `rag/config.py`).

---

**Son Güncelleme:** 2026-01-21

