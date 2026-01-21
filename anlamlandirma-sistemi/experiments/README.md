# Experiments (Benchmark) Rehberi

Bu klasör, `anlamlandirma-sistemi` içindeki TİD transkripsiyon → Türkçe çeviri akışını **deneysel olarak ölçmek** için hazırlanmış deney/benchmark altyapısını içerir.

Amaçlar:
- Farklı transkripsiyon uzunluklarında (3/4/5 kelime) başarı ve kalite ölçümü
- RAG açık/kapalı karşılaştırması
- Farklı LLM sağlayıcıları (gemini/openai/claude) karşılaştırması
- Sonuçları JSON + Markdown rapor olarak dışa aktarmak

---

## Klasör Yapısı

```
experiments/
├── test_sets/
│   ├── 3_word_glosses.json
│   ├── 4_word_glosses.json
│   └── 5_word_glosses.json
├── experiment_runner.py      # Batch koşucu (provider + rag seçimi)
├── report_generator.py       # Markdown rapor üretimi
├── run_experiments.py        # CLI giriş noktası
└── verify_test_isolation.py  # Split/izolasyon kontrolleri (opsiyonel)
```

---

## Test Seti Formatı

Her dosya bir liste içerir:

```json
[
  { "gloss": "BEN OKUL GITMEK", "reference": "Okula gidiyorum." },
  { "gloss": "SEN NEREYE GITMEK", "reference": "Nereye gidiyorsun?" }
]
```

Notlar:
- `gloss`: TİD transkripsiyon (space-separated)
- `reference`: Altın referans Türkçe cümle (tek referans)

---

## CLI ile Çalıştırma

Temel:

```bash
cd anlamlandirma-sistemi
python experiments/run_experiments.py --all
```

Sadece belirli uzunluk:

```bash
python experiments/run_experiments.py --words 3
python experiments/run_experiments.py --words 3 4 5
```

Limitli (kota/rate limit için önerilir):

```bash
python experiments/run_experiments.py --all --limit 5
```

RAG kapalı (baseline benzeri):

```bash
python experiments/run_experiments.py --all --no-rag
```

Çıktı kaydetme:

```bash
python experiments/run_experiments.py --all --output results.json --report benchmark_report.md
```

Provider seçimi:

```bash
python experiments/run_experiments.py --all --provider gemini
python experiments/run_experiments.py --all --provider openai
python experiments/run_experiments.py --all --provider claude
```

---

## Web Üzerinden Çalıştırma (Flask)

Uygulama çalışırken:
- Deney ekranı: `GET /experiments`
- Tek batch: `POST /api/run_experiment`
- Hepsi: `POST /api/run_all_experiments`

Örnek istek:

```bash
curl -X POST http://localhost:5005/api/run_experiment \
  -H "Content-Type: application/json" \
  -d '{ "word_count": 3, "provider": "gemini", "use_rag": true, "limit": 5 }'
```

---

## Üretilen Metrikler

Bu deney altyapısında “başarı” ve “özgüven” odaklı metrikler öne çıkar:
- **successful / failed**: LLM yanıtı üretildi mi?
- **avg_confidence**: LLM’in 1–10 özdeğerlendirmesi (parse edilerek)
- **avg_latency_ms**: Çağrı gecikmesi (API ile)

Not: BLEU/BERTScore gibi metrikler `TRANSKRIPSIYON-RAG-VDB/evaluation` tarafında daha ayrıntılı ele alınır.

---

## Kota / Rate-Limit ile Çalışma

Gemini/OpenAI/Claude kota sorunlarında iki yaklaşım:

1) `--limit` ile küçük örneklem
2) Manuel workflow:
   - Prompt üret: `POST /api/get_prompt`
   - LLM yanıtını manuel gir: `POST /api/process_manual_response`

Bu yöntem, makalede “API kısıtı” altında deneylerin sürdürülebilirliğini sağlar.

---

## İlgili Dokümanlar

- Üst seviye kılavuz: `anlamlandirma-sistemi/DENEY_KILAVUZU.md`
- RAG entegrasyonu: `anlamlandirma-sistemi/RAG_ENTEGRASYON.md`

---

**Son Güncelleme:** 2026-01-21

