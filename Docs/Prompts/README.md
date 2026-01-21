# Prompts Dokümantasyon Rehberi

Bu klasör, TID transkripsiyon → Türkçe çeviri sistemi için kullanılan prompt tasarımları, deneme notları ve makale yazımında referans olacak metinleri içerir.

Ana amaçlar:
- Prompt evrimini (v0 → vN) kayıt altında tutmak
- Deneylerde hangi prompt sürümünün kullanıldığını açıkça belgelemek
- Makalede **Method / Prompt Engineering** bölümünü yazarken doğrudan alıntı yapabilmek

---

## Dosyalar

| Dosya | İçerik | Kullanım |
|-------|--------|----------|
| `prompt.md` | İlk kapsamlı problem tanımı ve deneysel hedefler | Makale giriş / problem tanımı, deney motivasyonu |
| (ileride) `tid_prompt_v1.md` | Temel TID → TR prompt taslağı | İlk LLM denemeleri |
| (ileride) `tid_prompt_v2_rag.md` | RAG + 3 alternatifli çıktı formatı içeren sürüm | RAG tabanlı deneyler |

> Not: Şu an yalnızca `prompt.md` aktif; ileride her önemli değişiklik için yeni sürüm dosyası açmanız önerilir (üzerine yazmak yerine).

---

## `prompt.md` — İçerik Özeti

Bu dosya:
- İşaret dili **tanıma modeli** (226 sınıf Transformer) ve **transkripsiyon sistemi** arasındaki ilişkiyi tanımlar.
- Örnek TID transkripsiyonları ve hedef Türkçe cümleleri verir.
- Eksik kelime senaryolarını (örn. `TAMIR` kelimesinin atlanması) ve sistemin bu eksikleri hem tespit edip hem de doğrudan doğal Türkçe’ye çevirmesini hedefler.
- Değerlendirme için:
  - Farklı uzunluklardaki transkripsiyonlar (3, 5, 7 kelime),
  - Farklı eksik kelime oranları,
  - Hem sözdizimsel hem anlamsal metriklerin (BLEU + BERTScore) kullanılmasını önerir.

Makale açısından:
- **Giriş / Motivation** bölümünde alıntılanabilecek senaryolar içerir (`ESKI TELEVIZYON ATMAK ...` gibi).
- **Experimental Setup** bölümünde bahsedilecek deney varyasyonlarını tanımlar (kelime uzunluğu, eksik oranı, LLM tipi gibi).

---

## Sürümleme Önerisi

Yeni prompt sürümleri için aşağıdaki şemayı öneriyoruz:

- `prompt_v1_baseline.md` — Basit TID → TR prompt (RAG yok)
- `prompt_v2_rag_3alt.md` — RAG + 3 alternatif + güven puanı
- `prompt_v3_tid_grammar.md` — TID dilbilgisi ve Türkçe morfoloji kuralları eklenmiş sürüm
- `prompt_v4_eval_ready.md` — Benchmark deneylerinde kullanılan **donmuş** sürüm

Her sürüm dosyasının en başında şu meta bilgiyi tutun:

```text
Sürüm: v3
Tarih: 2026-01-20
Kullanıldığı deneyler: 3/4/5 kelimelik RAG + Gemini, benchmark_v3
İlgili kod: TRANSKRIPSIYON-RAG-VDB/prompt_builder/*
```

---

## Makale Yazımı İçin Nasıl Kullanılır?

### 1. Problem Tanımı

`prompt.md` içindeki örnekleri:
- **Problem Statement** kısmında “Eksik kelimeli TID transkripsiyonlarından doğal Türkçe üretme” görevini açıklamak için kullanın.
- Özellikle eksik kelime senaryolarını (TAMIR, YENI vb.) açıklayıcı şekiller / kutucuklar olarak verebilirsiniz.

### 2. Prompt Engineering Bölümü

Güncel prompt şablonunuz (RAG + 3 alternatif formatı vb.) `TRANSKRIPSIYON-RAG-VDB/prompt_builder/system_instructions.py` ve `templates.py` içinde kod olarak tutuluyor.  
Bu klasörde ise:
- Metinsel açıklamalar,
- Tasarım kararlarının gerekçeleri,
- Alternatif fikirler ve iptaller saklanabilir.

Makalede:
- Kısa bir **pseudo-prompt** verip, ayrıntılı tam sürümü ek materyal (appendix / supplementary) olarak bu klasöre referans gösterebilirsiniz.

### 3. Deney Tasarımı

`prompt.md` içindeki maddeler, deney tasarımı için doğrudan checkliste dönüştürülebilir:
- [ ] 3 / 5 / 7 kelimelik transkripsiyon setleri hazır
- [ ] Eksik kelime oranları (%0, %10, %20) tanımlı
- [ ] Hedef metrikler: BLEU + BERTScore + insan değerlendirmesi (opsiyonel)

Bu checklist’i `data-statistics-AULTS` ve `TRANSKRIPSIYON-RAG-VDB/evaluation` çıktıları ile birlikte kullanarak, **Methods → Evaluation Setup** kısmını doldurabilirsiniz.

---

## İleride Eklenecekler

- Farklı LLM’ler için (Gemini, Claude, GPT) optimize edilmiş prompt varyantları
- İnsan değerlendirmesi için kullanılacak talimat metinleri (annotator guideline)
- Makaleye eklenecek tam prompt sürümlerinin referans ID’leri (ör: “Appendix A: Prompt v4.1”)

---

**Son Güncelleme:** 2026-01-20
