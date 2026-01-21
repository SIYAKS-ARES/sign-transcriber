# Scripts Rehberi (TRANSKRIPSIYON-RAG-VDB/scripts)

Bu klasörde, TID RAG çeviri sisteminin **kurulum, test ve benchmark** işlemleri için yardımcı script’ler bulunur.

---

## İçerik

| Script | Amaç |
|--------|------|
| `init_sozluk.py` | `tid_sozluk` (statik sözlük) koleksiyonunu `TID_Sozluk_Verileri/` verisinden yükler |
| `init_hafiza.py` | `tid_hafiza` (dinamik hafıza) koleksiyonunu başlangıç verisiyle yükler |
| `test_rag.py` | Uçtan uca smoke test (konfig → preprocessing → retrieval → prompt → vb.) |
| `test_linguistic.py` | Dilbilgisel özellik testleri (TIDPreprocessor, parser, few-shot seçimleri) |
| `run_benchmark.py` | Örnek benchmark runner (not: mock translator kullanır) |

---

## Hızlı Başlangıç

```bash
cd TRANSKRIPSIYON-RAG-VDB

# 1) Sözlük koleksiyonunu hazırla
python scripts/init_sozluk.py

# 2) Hafıza koleksiyonunu hazırla
python scripts/init_hafiza.py

# 3) Smoke test
python scripts/test_rag.py

# 4) Dilbilgisi testleri
python scripts/test_linguistic.py
```

> Not: `init_*` script’leri ChromaDB’yi `vectorstore/` altında oluşturur/kullanır.

---

## Benchmark Notu

`run_benchmark.py` **mock translator** içerir. Gerçek LLM ile benchmark çalıştırmak için:
- `pipeline/translation_pipeline.py` ile gerçek çevirileri üretip
- `evaluation/benchmark.py` metrikleriyle (BLEU/BERTScore) hesaplayıp
- sonuçları JSON’a yazmak önerilir.

Detaylı raporlar:
- `evaluation/BENCHMARK_RAPORU.md`
- `evaluation/BENCHMARK_REPORT_EN.md`

---

## Sorun Giderme

- **Koleksiyon boş**: Önce `init_sozluk.py` ve `init_hafiza.py` çalıştırın.
- **API key yok / kota dolu**: `.env` değişkenlerini kontrol edin; gerekiyorsa küçük limitlerle test edin.

---

**Son Güncelleme:** 2026-01-21

