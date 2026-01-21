# TID Ceviri Deneyleri Kilavuzu

Bu belge, 3, 4 ve 5 kelimelik TID transkripsiyon cumlelerinin cevirisini test etmek icin deney sisteminin kullanimini aciklar.

## Hizli Baslangiç

### CLI ile Deney Calistirma

```bash
# Tum deneyleri calistir
cd anlamlandirma-sistemi
python experiments/run_experiments.py --all

# Sadece 3 kelimelik deneyleri calistir
python experiments/run_experiments.py --words 3

# Sinirli ornek sayisi ile test
python experiments/run_experiments.py --all --limit 5

# Sonuclari dosyaya kaydet
python experiments/run_experiments.py --all --output results.json --report benchmark.md
```

### Web Arayuzu

1. Flask uygulamasini baslatin:
   ```bash
   python app.py
   ```

2. Tarayicida acin: `http://localhost:5005/experiments`

3. Deney ayarlarini secin ve "Deneyleri Baslat" tusuna basin.

## Test Setleri

Test setleri `experiments/test_sets/` klasorunde bulunur:

| Dosya | Ornek Sayisi | Aciklama |
|-------|--------------|----------|
| 3_word_glosses.json | 20 | 3 kelimelik TID cumleleri |
| 4_word_glosses.json | 20 | 4 kelimelik TID cumleleri |
| 5_word_glosses.json | 20 | 5 kelimelik TID cumleleri (manuel olusturuldu) |

### Test Seti Formati

```json
{
    "gloss": "BEN OKUL GITMEK",
    "reference": "Okula gidiyorum."
}
```

## Deney Parametreleri

### Provider (LLM Saglayici)

- `gemini`: Google Gemini (varsayilan, onerilen)
- `openai`: OpenAI GPT-4
- `claude`: Anthropic Claude

### RAG Modu

- `--use-rag`: RAG sistemi etkin (varsayilan)
- `--no-rag`: RAG olmadan direkt LLM

### Limit

- `--limit N`: Her kelime sayisi icin N ornek test et

## Ciktilar

### JSON Ciktisi

```json
{
    "timestamp": "2026-01-21T08:00:00",
    "provider": "gemini",
    "rag_enabled": true,
    "results": {
        "3": {
            "word_count": 3,
            "total_samples": 20,
            "successful": 18,
            "failed": 2,
            "avg_confidence": 8.5,
            "avg_latency_ms": 1200,
            "results": [...]
        }
    }
}
```

### Markdown Raporu

Rapor su bolumlerden olusur:
- Ozet istatistikler tablosu
- Kelime sayisina gore detayli sonuclar
- Hata analizi

## Metrikler

### Basari Kriterleri

- **Basarili**: Ceviri uretildi ve hata yok
- **Basarisiz**: Ceviri uretilemedi veya hata olustu

### Guven Skoru

LLM'in 1-10 arasi ozdegerlendirmesi:
- 8-10: Yuksek guven (yesil)
- 6-7: Orta guven (sari)
- 1-5: Dusuk guven (kirmizi)

### Gecikme (Latency)

Tek bir ceviri icin gecen sure (milisaniye).

## Tipik Sonuclar

| Kelime Sayisi | Beklenen Basari | Beklenen Guven |
|---------------|-----------------|----------------|
| 3 kelime | %85-95 | 8-9/10 |
| 4 kelime | %75-85 | 7-8/10 |
| 5 kelime | %65-80 | 6-8/10 |

## Sorun Giderme

### "RAG sistemi kulanilamiyor"

```bash
# Vectorstore'u kontrol et
python scripts/init_vectorstore.py --check

# Gerekirse kopyala
cp -r ../TRANSKRIPSIYON-RAG-VDB/vectorstore .
```

### "GEMINI_API_KEY bulunamadi"

```bash
# .env dosyasina ekle
echo "GEMINI_API_KEY=your-key-here" >> .env

# veya export
export GEMINI_API_KEY=your-key-here
```

### Yavas Ceviri Suresi

- Ilk calistirmada embedding modeli indirilir (~400MB)
- Sonraki calistirmalar daha hizli olacaktir
- Network baglantisini kontrol edin

## Ornek Calisma Akisi

```bash
# 1. Ortami hazirlav
cd anlamlandirma-sistemi
pip install -r requirements.txt

# 2. Vectorstore'u kontrol et
python scripts/init_vectorstore.py --check

# 3. Hizli test (2 ornek)
python experiments/run_experiments.py --words 3 --limit 2

# 4. Tam benchmark
python experiments/run_experiments.py --all --output full_results.json --report benchmark_report.md

# 5. Raporu incele
cat benchmark_report.md
```

## Akademik Kullanim

Deney sonuclari makale icin kullanilabilir:

1. JSON ciktisini `full_results.json` olarak kaydedin
2. Markdown raporunu `benchmark_report.md` olarak kaydedin
3. Istatistikleri tablolar halinde sunun
4. Kelime sayisina gore performans grafiklerini olusturun

Detayli akademik dokumantasyon icin plan dosyasindaki "Makale Icin Kullanim Dokumani" bolumune bakiniz.
