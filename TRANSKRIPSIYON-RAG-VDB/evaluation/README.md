# Evaluation (Benchmark) Rehberi

Bu klasör, `TRANSKRIPSIYON-RAG-VDB` TID→TR çeviri pipeline’ı için **akademik değerlendirme** (benchmark) çıktıları ve metrik hesaplama kodlarını içerir.

Burada iki amaç vardır:
1. **Metrik hesaplama**: BLEU ve BERTScore ile otomatik ölçüm
2. **Makaleye uygun raporlama**: Sonuçların Türkçe/İngilizce rapor dosyaları

---

## İçerik

| Dosya | Amaç |
|------|------|
| `benchmark.py` | BLEU + BERTScore hesaplayıcı (TranslationBenchmark) |
| `baseline.py` | RAG’siz zero-shot baseline çevirici (karşılaştırma için) |
| `test_sets/test_glosses.json` | Test seti (gloss + reference) |
| `benchmark_results.json` | Mini benchmark sonucu (örnek) |
| `full_benchmark_results.json` | Tam benchmark denemesi (rate limit nedeniyle başarısızlar da var) |
| `benchmark_results_corrected.json` | Sadece **başarılı** örnekler üzerinden düzeltilmiş metrikler |
| `BENCHMARK_RAPORU.md` | Makale için Türkçe rapor |
| `BENCHMARK_REPORT_EN.md` | Makale için İngilizce rapor |

---

## Test Seti Formatı

`test_sets/test_glosses.json` formatı:

```json
[
  { "gloss": "BEN OKUL GITMEK", "reference": "Okula gidiyorum." },
  { "gloss": "SEN NEREYE GITMEK", "reference": "Nereye gidiyorsun?" }
]
```

Öneri:
- Aynı gloss için birden fazla referans varsa (multi-reference), ileride ayrı alan olarak eklenebilir.

---

## Metrikler

### 1) BLEU (sacrebleu)
- N-gram overlap
- MT literatüründe standart metrik
- Kısa cümlelerde katı olabilir (küçük kelime farklarını fazla cezalandırır)

### 2) BERTScore (bert-score)
- Semantik benzerlik odaklı
- TİD→TR gibi yeniden biçimlendirme/morfoloji ekleme gerektiren işlerde daha anlamlı olabilir
- Bu projede `lang=\"tr\"` ile hesaplanır

---

## Nasıl Çalıştırılır?

### A) Kendi benchmark runner’ınız

Repo içinde iki yol var:

1) `scripts/run_benchmark.py` (örnek runner)  
2) Pipeline’ı doğrudan çağırıp sonuçları JSON’a yazmak (bizim yaptığımız mini benchmark yaklaşımı)

> Not: LLM kota / rate-limit yüzünden “tam test seti” koşuları yarıda kalabilir. Bu yüzden `benchmark_results_corrected.json` yaklaşımı (başarılı örnekler üzerinden metrik) eklendi.

### B) Baseline karşılaştırması

`baseline.py` içindeki `BaselineTranslator`, RAG olmadan “simple prompt” ile LLM çağırır. RAG ile kıyaslamak için:

- RAG: `pipeline/translation_pipeline.py`
- Baseline: `BaselineTranslator.translate()`

---

## Çıktılar Nasıl Yorumlanmalı?

Bu repo gerçek LLM koşularında API kotasına takılabildiği için sonuçlar iki şekilde raporlanır:

1) **Tüm test seti** (başarısız/boş cevaplar dahil):  
   - `full_benchmark_results.json`
   - Bu, altyapının “uçtan uca koşabildiği”ni gösterir; ancak metrikler başarısızlar nedeniyle düşer.

2) **Sadece başarılı örnekler** üzerinden metrik:  
   - `benchmark_results_corrected.json`
   - Makalede “rate-limit kısıtı” açıkça belirtilerek raporlanmalıdır.

Önerilen makale cümlesi:
> “API rate limit nedeniyle tüm örneklerde çıkarım alınamamış; metrikler, yalnızca başarıyla üretilen örnekler üzerinden ayrıca raporlanmıştır.”

---

## Makale İçin

- Türkçe rapor: `BENCHMARK_RAPORU.md`
- İngilizce rapor: `BENCHMARK_REPORT_EN.md`

Bu iki rapor:
- Method (evaluation protocol)
- Results (BLEU/BERTScore tabloları)
- Discussion (kısıtlar, hata analizi, RAG katkısı)

bölümlerine doğrudan taşınabilecek şekilde hazırlanmıştır.

---

**Son Güncelleme:** 2026-01-20

