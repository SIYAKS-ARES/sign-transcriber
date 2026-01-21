# Sign Transcriber: Proje Geliştirme, Literatür ve Deneysel Tasarım Dokümanı

Bu doküman, "Sign Transcriber" bitirme projesi kapsamında yapay zeka asistanı ile gerçekleştirilen teknik görüşmelerin, literatür tarama stratejilerinin ve deneysel kurgu planlarının konsolide edilmiş halidir.

## BÖLÜM 1: PROJE BAĞLAMI VE TEKNİK DURUM

### 1.1. Proje Tanımı

**Hedef:** Kamera veya video üzerinden alınan görüntüleri işleyerek Türk İşaret Dili'ni (TİD) sürekli (continuous) olarak tanıyan ve bu işaret dizilerini (gloss) anlamlı, doğal Türkçe yazı ve sese çeviren bir sistem geliştirmek.

### 1.2. Mevcut Teknik Durum

* **Önceki Mimari:** LSTM + CNN + MediaPipe (Yetersiz verim).
* **Mevcut Mimari:** Transformers + MediaPipe.
  * *Durum:* Başarım arttı ancak eğitim verisi küçük olduğu için **overfitting (aşırı öğrenme)** problemi yaşanıyor.
  * *Kapasite:* Model şu an 226 kelime sınıfı üzerinden eğitildi.
* **Mevcut Sorun:** Modelin ürettiği çıktılarda (Gloss dizileri) kelime atlamaları (missing words) ve bağlam kopuklukları oluşuyor.
* **Yeni Hedef:** İşaret dili modelinden gelen eksik/hatalı "Gloss" dizilerini alıp, bağlamsal olarak tamamlayan ve doğal Türkçe cümlelere dönüştüren bir **Transkripsiyon/Çeviri Modülü** tasarlamak.

## BÖLÜM 2: LİTERATÜR TARAMASI STRATEJİSİ

Akademik platformlarda (Web of Science, Google Scholar) "SLR" araması yapıldığında çıkan gürültüyü (Single Lens Reflex kamera, biyolojik terimler vb.) engellemek ve Transformer/Overfitting odaklı doğru kaynaklara ulaşmak için geliştirilen stratejilerdir.

### 2.1. Temel Arama Mantığı ve Operatörler

* **NOT Operatörü:** Gürültüyü temizlemek için hayati önem taşır. (Örn: `NOT "Single Lens Reflex"`)
* **" " (Tırnak İşareti):** Kesin eşleşme sağlar. `Sign Language` şeklinde ayrı kelimeler yerine `"Sign Language"` kalıbını arar.
* **AND / OR:** Konuyu daraltmak (AND) veya varyasyonları kapsamak (OR) için kullanılır.

### 2.2. Hazır Arama Sorguları (Copy-Paste için)

Aşağıdaki sorgular Web of Science ve Google Scholar için optimize edilmiştir.

#### A. Genel Mimari ve CSLR Taraması (En Kapsayıcı)

Bu sorgu; ASL/CSL/DGS fark etmeksizin Transformer tabanlı, Pose/Skeleton kullanan modern çalışmaları getirir ve kamera/biyoloji gürültüsünü temizler.

```
("continuous sign language" OR "sign language recognition" OR "sign language translation" OR CSLR)
AND (transformer* OR "self-attention" OR "sequence to sequence" OR "vision transformer")
AND (pose OR landmark OR "spatio-temporal" OR "skeleton-based" OR "MediaPipe")
NOT ("single lens reflex" OR "SLR camera" OR "SCAR" OR "biology" OR "protein" OR "optics")

```

#### B. Overfitting ve Regularization Odaklı Tarama

Mevcut Transformer modelindeki aşırı öğrenme sorununu çözmek için kullanılan teknikleri (Data Augmentation, DropPath vb.) bulmak için:

```
("sign language recognition" OR "continuous sign language")
AND (transformer* OR "temporal modeling")
AND ("data augmentation" OR "regularization" OR "dropout" OR "label smoothing" OR "stochastic depth" OR "contrastive learning" OR "self-supervised")
NOT ("single lens reflex" OR "camera" OR "biology")

```

#### C. Veri Seti Odaklı Taramalar (SOTA Karşılaştırması)

Literatürdeki kıyaslamaları görmek için önemli veri setleri ile arama:

* `"RWTH-PHOENIX-Weather 2014" AND (CSLR OR transformer)`
* `"AUTSL" AND "sign language" AND "deep learning"`

### 2.3. Önerilen Anahtar Kelimeler (Keywords)

* **Temel:** "Continuous Sign Language Recognition (CSLR)", "Sign Language Translation", "Gloss Prediction".
* **Model:** "Transformer", "Vision Transformer (ViT)", "Seq2Seq", "Spatio-temporal".
* **Veri/Yöntem:** "Skeleton-based", "Pose-based", "Keypoint", "MediaPipe".

## BÖLÜM 3: TRANSKRİPSİYON SİSTEMİ DENEYSEL TASARIMI

Bu bölüm, işaret dilinden gelen eksik/hatalı kelime dizilerini (Gloss) doğal Türkçeye çeviren modülün akademik makalesi için kurulan deney düzeneğidir.

### 3.1. Deneyin Amacı

Gelen sıralı işaret dili transkripti (Gloss) içindeki  **eksik/atlanan kelimeleri tespit edip** , bağlamla uyumlu aday kelimeler öneren ve bu düzeltmelerle **doğal, akıcı Türkçe cümle** üreten bir sistemin başarımını ölçmek.

### 3.2. Veri Hazırlığı ve Kaynaklar

1. **Kaynak:** `tidsozluk.aile.gov.tr` adresinden scrape edilen ~2000 kelimelik veri seti (Kelime, Transkripsiyon, Çeviri içerir).
2. **Mevcut Model Çıktısı:** 226 kelimelik modelin çıktıları (Sentetik olarak simüle edilecek).
3. **Gold Standart:** Tam Gloss dizisi ve karşılık gelen doğru Türkçe çeviri.

### 3.3. Sentetik Veri Üretimi ve Hata Simülasyonu

Gerçek model hatalarını taklit etmek için "Gold" veriler üzerinde kontrollü bozulmalar (corruption) yapılacaktır.

* **Varyasyon 1: Dizi Uzunluğu (L)**
  * 3 Kelimelik Diziler
  * 5 Kelimelik Diziler
  * 7 Kelimelik Diziler
* **Varyasyon 2: Eksik Kelime Oranı (Missing Rate - R)**
  * %0 (Kontrol Grubu)
  * %10, %20, %30, %50 (Deneysel Kademeler)
* **Varyasyon 3: Eksiklik Türü**
  * *Random:* Rastgele kelime silme.
  * *Content:* Sadece isim/fiil gibi içerik kelimelerini silme.
  * *Function:* Sadece bağlaç/ek gibi işlevsel kelimeleri silme.
  * *Contiguous:* Ardışık blok halinde silme (bağlamı zorlamak için).

Örnek Sayısı (Power Analysis):

Her deney hücresi (Uzunluk × Oran × Tip) için N = 150 (Pilot) veya N = 300 (Full) örnek önerilmektedir.

* Toplam Tahmini Örnek: ~7,200 (Full scale) veya ~3,600 (Pilot).

### 3.4. Önerilen Yöntemler ve Karşılaştırma (Baselines)

Makalede kıyaslanacak 5 farklı yaklaşım belirlenmiştir:

1. **No-correction (Baseline):** Eksik gloss dizisini olduğu gibi çeviriye sok.
2. **Rule-based / Heuristic N-gram:** TİD korpusundan bigram olasılıklarına göre boşluk doldur.
3. **Prompt-based LLM:** GPT/Claude/Gemini gibi modellere "Bu gloss dizisindeki eksikleri tamamla ve çevir" promptu ver.
4. **Seq2Seq Fine-tune:** Küçük bir modeli (örn. mT5-small) bu görev için fine-tune et.
5. **Two-Stage (Önerilen Ana Yöntem):**
   * *Aşama 1 (Detection & Candidate):* Eksik kelime yerini tespit et + Aday kelime üret (Masked LM veya Bigram ile).
   * *Aşama 2 (Generation):* En iyi adayı seç ve doğal cümleyi üret.

## BÖLÜM 4: DEĞERLENDİRME METRİKLERİ

Sistemin başarısı hem otomatik metrikler hem de insan değerlendirmesi ile ölçülecektir.

### 4.1. Otomatik Metrikler

* **Çeviri Kalitesi:** BLEU, METEOR, chrF (Türkçe morfolojisi için önemli).
* **Anlamsal Benzerlik:** BERTScore (Kelime hatası olsa bile anlamın korunup korunmadığını ölçer).
* **Hata Oranları:** WER (Word Error Rate).
* **Detection Başarısı:** Precision/Recall (Eksik kelime yerini doğru buldu mu?), Top-k Accuracy (Doğru kelime önerilen ilk k kelime içinde mi?).

### 4.2. İnsan Değerlendirmesi (Human Eval)

En az 3 değerlendirici ile aşağıdaki kriterler 1-5 Likert ölçeğinde puanlanacaktır:

1. **Fidelity (Sadakat):** Çeviri, orijinal işaret dizisinin anlamını taşıyor mu?
2. **Fluency (Akıcılık):** Çıktı doğal, gramer kurallarına uygun bir Türkçe mi?
3. **Missing-Word Correctness:** Sistem eksik yeri doğru tahmin etti mi?

## BÖLÜM 5: UYGULAMA YOL HARİTASI (M3 Macbook Pro için)

Donanım kısıtları (M3 Macbook) göz önüne alınarak "Lightweight" ve "Modular" bir yaklaşım benimsenmiştir.

### 5.1. Adım Adım İş Planı

1. **Veri Hazırlama Scripti (`synthetic_generator.py`):**
   * TİD JSON dosyasını okur.
   * Sliding window ile 3/5/7 uzunluğunda parçalar.
   * Belirlenen oranlarda (R) rastgele kelimeleri siler (Dropout).
   * Çıktıyı `experiment_matrix.csv` olarak kaydeder.
2. **Model Çalıştırma:**
   * Önce **Prompt-based** ve **Heuristic** yöntemleri çalıştır (GPU gerektirmez).
   * Zorlu senaryolar için küçük bir **mT5-small** modelini LoRA veya Adapter ile fine-tune etmeyi dene.
3. **Değerlendirme Scripti (`evaluation_pipeline.py`):**
   * CSV'deki `Prediction` ve `Gold Translation` sütunlarını karşılaştırır.
   * BLEU, BERTScore ve WER hesaplar.

### 5.2. Teknik Kısıtlar ve Çözümler

* **Türkçe Dil Modeli Eksikliği:** Türkçe'ye özel eğitilmiş güçlü bir işaret dili modeli yok. Bu durum makalede bir "Limitation" olarak belirtilecek ve Multilingual modellerin (mBERT, mT5) performansı analiz edilecek.
* **Donanım:** Ağır eğitimler yerine "Inference" ağırlıklı (Prompting) ve "Light Fine-tuning" yöntemleri kullanılacak.

### 5.3. Makale İçin Beklenen Hipotezler

* Eksik kelime oranı arttıkça BLEU (yüzeysel) puanı hızla düşecek, ancak BERTScore (anlamsal) daha dirençli kalacaktır.
* 3 kelimelik kısa dizilerde bağlam az olduğu için tamamlama başarısı, 5-7 kelimelik dizilere göre daha düşük olacaktır.
* Two-stage (Tespit et -> Doldur) yaklaşımı, doğrudan çeviriye göre daha kontrollü ve açıklanabilir sonuçlar verecektir.

**Not:** Bu doküman, projenin "Metodoloji", "Literatür Taraması" ve "Deneysel Sonuçlar" bölümlerinin iskeletini oluşturmaktadır. Tüm Python script taslakları ve CSV formatları önceki sohbet kayıtlarında (dosya 2 ve 6) mevcuttur.
