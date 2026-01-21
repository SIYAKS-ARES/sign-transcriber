# **🔹 KISA ÖZET — Adım Adım Plan**

## **1) Veri Hazırlama**

* TID `data.json` → **transkripsiyon** alanlarından gloss-benzeri token dizileri çıkar.
* Normalizasyon: küçük harf, noktalama temizleme, whitespace tokenizasyonu.
* Sliding-window ile **L = 3, 5, 7** uzunluklu sekanslar üret.
* Her sekansın **gold çevirisini** kaydet.
* Kontrollü veri bozma (corruption):

  * Missing rate **R ∈ {0,10,20,30,50}%**
  * Missing type **{content, function, random, contiguous}**
  * Her kombinasyondan **N = 150** örnek.

## **2) Test Seti Oluşturma**

* Her örnek için kayıt:

  * `id`, `source_gloss_gold`, `source_gloss_corrupted`, `missing_positions`, `gold_translation`, `L`, `R`, `missing_type`, `seed`.
* Yaklaşık toplam: **9,000** örnek.

## **3) Modeller & Baselines**

* **Pass-through** baseline (kural tabanlı).
* **Heuristic n-gram LM** (unigram/bigram).
* **Prompt-based LLM** (top-3 üretim).
* Hafif **seq2seq fine-tune** (mt5-small / T5-small, LoRA/adapter).
* **Two-stage pipeline:**

  * (A) Eksik token tahmini
  * (B) Türkçe üretim

## **4) Değerlendirme Metrikleri**

* BLEU
* METEOR (opsiyonel)
* WER
* BERTScore
* Top-k oracle (top-1 / top-3 doğruluk)
* İnsan değerlendirmesi için anket şablonu (sonradan kullanılacak).

## **5) Analiz**

* Performans vs missing rate grafikleri.
* Missing type kırılganlık analizi.
* L uzunluğu ile performans ilişkisi.
* İstatistiksel testler: paired bootstrap / Wilcoxon.

## **6) Artefaktlar (Hazır Üretilebilir)**

* `experiment_matrix.csv`
* `synthetic_generator.ipynb` (veri üretimi)
* `evaluation_pipeline.py`
* `baseline_prompt_examples.txt`
* `human_annotation_template.csv`

## **7) Öncelikli Yapılacaklar**

1. Veri üretim notebook’u (`synthetic_generator.ipynb`)
2. Deney matrisi (`experiment_matrix.csv`)
3. Değerlendirme script’i
4. Baseline prompt dosyası
5. (Opsiyonel) ufak bir mt5-small fine-tune denemesi

---

