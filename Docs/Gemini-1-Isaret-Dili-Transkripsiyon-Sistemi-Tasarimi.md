

# **Düşük Kaynaklı Türk İşaret Dili Tanıma Sistemlerinde Hata Toleranslı Transkripsiyon: Gürültülü Gloss Akışlarının Anlamsal ve Yapısal Onarımı İçin Derin Öğrenme Tabanlı Bir Çerçeve**

## **1\. Giriş ve Problem Tanımı**

### **1.1. İşaret Dili Çevirisinde Mevcut Paradigma ve Zorluklar**

İşaret Dili Çevirisi (Sign Language Translation \- SLT), bilgisayarlı görü ve doğal dil işleme (NLP) disiplinlerinin kesişim noktasında yer alan, görsel-uzamsal bir modaliteden (işaret dili) işitsel-doğrusal bir modaliteye (yazılı/konuşma dili) geçişi hedefleyen karmaşık bir yapay zeka problemidir. Türk İşaret Dili (TİD) gibi düşük kaynaklı dillerde, bu problem veri yetersizliği nedeniyle daha da derinleşmektedir. Mevcut literatür incelendiğinde, SLT sistemlerinin genellikle iki ana mimari üzerine kurulduğu görülmektedir: uçtan uca (end-to-end) sistemler ve kademeli (cascaded) sistemler.1 Büyük veri setlerine (örneğin RWTH-PHOENIX-Weather 2014T gibi) sahip dillerde uçtan uca modeller doğrudan videodan metne çeviri yapabilirken, TİD gibi annotasyonu yapılmış paralel video-metin verisinin kısıtlı olduğu durumlarda kademeli yaklaşım zorunlu bir standart haline gelmiştir.2

Bu raporda ele alınan senaryo, kademeli mimarinin tipik bir örneğini teşkil etmektedir. Kullanıcı tarafından geliştirilen birinci aşama (Sign Language Recognition \- SLR), video akışını "gloss" (işaret etiketi) dizilerine dönüştürmektedir. Ancak, düşük eğitim verisi nedeniyle bu modelin çıktısı gürültülüdür; kelime atlamaları (deletion errors), yanlış tanımalar (substitution errors) ve zamanlama hataları içermektedir. Bu durum, problemin doğasını değiştirmektedir: Artık elimizdeki görev sadece bir "çeviri" (translation) problemi değil, aynı zamanda bir "gürültü giderme ve yapısal onarım" (denoising and reconstruction) problemidir.

Geleneksel Makine Çevirisi (MT) sistemleri, kaynak dildeki cümlenin tam ve dilbilgisel olarak doğru olduğunu varsayar. Ancak kullanıcının senaryosunda, girdi akışı (örneğin: "OKUL... GİTMEK...") eksiktir ve TİD'in zengin morfolojik yapısını (yüz ifadeleri, kafa hareketleri gibi manuel olmayan işaretler) kaybetmiştir.4 Bu nedenle, tasarlanacak transkripsiyon sistemi, eksik olan bu bilgileri bağlamsal ipuçlarından ve hedef dilin (Türkçe) istatistiksel olasılıklarından yola çıkarak "halüsinasyon" yöntemiyle yeniden inşa etmek zorundadır. Bu rapor, eksik veri ve gürültülü girdi kısıtlamaları altında çalışan, Transformer tabanlı, hataya dayanıklı bir Gloss-to-Text (G2T) mimarisini ve bu sistemin performansını ölçecek titiz bir deneysel kurguyu detaylandırmaktadır.

### **1.2. Türk İşaret Dili ve Türkçe Arasındaki Yapısal Uçurum**

Tasarım kararlarını gerekçelendirmek için TİD ve Türkçe arasındaki derin tipolojik farkların anlaşılması elzemdir. Türkçe, sondan eklemeli (agglutinative) yapısı ile bilinirken, TİD eşzamanlı (simultaneous) ve uzamsal (spatial) bir yapıya sahiptir.6 Bir TİD cümlesinde zaman, özne ve nesne ilişkileri kelimelerin sıralanışından ziyade, işaretin yapıldığı yön, tekrar sayısı ve eşlik eden mimiklerle belirlenir.

| Özellik | Türkçe (Hedef Dil) | Türk İşaret Dili (Kaynak Gloss Akışı) | Transkripsiyon Zorluğu |
| :---- | :---- | :---- | :---- |
| **Zaman (Tense)** | Fiil sonuna eklenen ekler (örn. *\-di*, *\-ecek*) | Zaman zarfları (DÜN, YARIN) veya tamamlama belirteçleri (BİTTİ) 8 | Tanıma modeli "DÜN" işaretini kaçırırsa, sistem zamanı bağlamdan tahmin etmelidir. |
| **Olumsuzluk** | Ek (*\-me/-ma*) veya kelime (*değil*) | Kafa sallama (headshake), geriye yaslanma (backward tilt) veya manuel işaret (DEĞİL) 5 | Video modelinin kafa hareketini algılayamaması durumunda cümle olumluya dönebilir. |
| **Sözdizimi** | Genellikle SOV (Özne-Nesne-Yüklem), serbest dolaşım | Konu-Yorum (Topic-Comment), OSV veya SOV 10 | Gloss akışındaki devrik yapı, Türkçe kurallı cümleye dönüştürülmelidir. |
| **Kişi Zamirleri** | Fiil çekim ekleri (*gel-di-m*) | Uzamsal indeksleme (spatial indexing/pointing) 12 | Glosslarda işaret zamirleri (INDEX) eksikse, sistem özneyi "gizli özne" olarak türetmelidir. |

Bu tablo, problemin sadece kelime çevirisi olmadığını, aynı zamanda "kayıp morfolojinin yeniden üretimi" olduğunu göstermektedir. Özellikle TİD'de olumsuzluk ve soru yapılarının genellikle manuel olmayan işaretlerle (non-manual markers) yapılması 9, ve mevcut video tanıma modelinin muhtemelen bu ince detayları kaçırıyor olması, transkripsiyon sisteminin en büyük risk faktörüdür. Tasarlanacak sistem, bu belirsizliği yönetebilecek olasılıksal bir derinliğe sahip olmalıdır.

---

## **2\. Literatür İncelemesi ve Mevcut Yaklaşımların Sınırları**

Akademik bağlamda, Gloss-to-Text (G2T) çevirisi genellikle düşük kaynaklı bir Nöral Makine Çevirisi (NMT) problemi olarak ele alınır.13 Ancak TİD özelinde yapılan çalışmalar, dilin kendine has zorluklarını ortaya koymaktadır.

### **2.1. Kural Tabanlı ve Hibrit Sistemler**

Erken dönem çalışmalar ve bazı güncel hibrit yaklaşımlar, TİD'den Türkçeye çeviri için kural tabanlı yöntemler önermiştir. Örneğin, Kayahan ve Güngör 15 tarafından önerilen hibrit sistemler, TİD glosslarını önce morfolojik analizden geçirip, ardından önceden tanımlanmış gramer kuralları ile Türkçeye dönüştürmeyi denemiştir. Bu sistemlerde, "BEN GİTMEK" gloss dizisi, bir kural motoru tarafından "Ben gidiyorum" veya "Ben giderim" şeklinde çekimlenir. Ancak bu yöntemlerin temel kısıtı, kural setlerinin kırılganlığıdır. Kullanıcının belirttiği gibi tanıma modelinden gelen akışta "BEN" kelimesi eksikse (yani girdi sadece "GİTMEK" ise), kural tabanlı sistem çöker veya yanlış bir varsayımla (örneğin 3\. tekil şahıs) çıktı üretir. Kural tabanlı sistemler, "missing word" (eksik kelime) senaryolarında gereken esnekliğe ve genelleme yeteneğine sahip değildir.17

### **2.2. İstatistiksel ve Nöral Yaklaşımlar**

Daha modern yaklaşımlar, Transformer mimarilerini kullanarak G2T problemini ele almaktadır.3 Camgoz ve ark. 21 ile Yin ve Read 22, gloss dizilerini kaynak dil, konuşma dilini hedef dil olarak kabul eden Seq2Seq modelleri eğitmişlerdir. Ancak bu çalışmaların çoğu, PHOENIX-14T gibi "temiz" veya "annotate edilmiş" gloss verileri üzerinde çalışmaktadır. Kullanıcının senaryosundaki gibi, bir video tanıma modelinin ürettiği "gürültülü ve eksik" gloss akışları üzerinde çalışan sistemler için, **Denoising Autoencoder** (Gürültü Giderici Otokodlayıcı) mantığının çeviri sistemine entegre edilmesi gerekmektedir.2 Literatürdeki bu boşluk, kullanıcının yazacağı makalenin temel katkısı (contribution) olacaktır.

---

## **3\. Metodoloji: Gürültüye Dayanıklı Transkripsiyon Mimarisi**

Kullanıcının "veri azlığı" ve "kelime kaçırma" sorunlarını çözmek için önerilen mimari, **Sentetik Veri Artırma (Data Augmentation)** ile güçlendirilmiş, ön eğitimli bir **Çok Dilli Transformer (mBART/mT5)** modeline dayanmaktadır.

### **3.1. Veri Sorununun Çözümü: Kural Tabanlı Gürültü Enjeksiyonu (Back-Translation with Noise)**

TİD için yeterli miktarda (Video \-\> Gloss \-\> Türkçe) üçlüsü bulunmadığından, elimizdeki bol miktardaki Türkçe metin verisini kullanarak "sanki video modelinden gelmiş gibi" bozuk gloss dizileri üretmemiz gerekmektedir. Bu süreç, literatürde "Back-Translation" (Geri Çeviri) ve "Synthetic Noise Injection" (Sentetik Gürültü Enjeksiyonu) olarak bilinir.2

Önerilen veri üretim algoritması şu adımları izlemelidir:

1. **Kaynak Veri Seçimi:** Türkçe Wikipedia, TR-News veya OpenSubtitles gibi genel amaçlı ve konuşma diline yakın metin derlemeleri seçilir.  
2. **Sözde-Gloss (Pseudo-Gloss) Üretimi:** Türkçe cümleler, TİD gloss formatına dönüştürülür. Bu aşamada **Zemberek** 25 veya **Zeyrek** 27 gibi NLP kütüphaneleri kritik rol oynar.  
   * *Lemmatizasyon:* Türkçe kelimelerin kökleri bulunur. Örneğin "yapabileceğim" kelimesi, Zemberek ile analiz edilerek YAP, ABIL (yeterlilik), ECEK (gelecek), IM (1. tekil) parçalarına ayrılır. Ancak TİD gloss yapısında bu genellikle sadece YAPMAK kökü ve belki bir zaman zarfı ile temsil edilir.  
   * *Durdurma Kelimelerinin (Stop-Words) Temizlenmesi:* TİD'de karşılığı olmayan fonksiyonel kelimeler (örneğin "bir", "mi" soru eki, bağlaçlar) cümleden atılır.28  
3. **TİD Sözdizimi Simülasyonu:** Oluşan kök dizisi, TİD'in tipik sözdizimine (örn. Yüklem sonda, Olumsuzluk en sonda) göre yeniden sıralanır. Örneğin "Okula gitmiyorum" \-\> OKUL GİTMEK DEĞİL.  
4. **Tanıma Hatalarının Simülasyonu (Gürültü Enjeksiyonu):** Kullanıcının "kelime kaçırma" sorununu modele öğretmek için, bu temiz gloss dizisi stokastik olarak bozulur.29  
   * **Silme (Deletion \- $P\_{del}$):** Kelimeler rastgele %20-%40 oranında silinir. Bu, kameranın işareti yakalayamamasını simüle eder.  
   * **Yer Değiştirme (Permutation \- $P\_{perm}$):** Yan yana duran glossların yeri değiştirilir.  
   * **İkame (Substitution \- $P\_{sub}$):** Benzer el şekline sahip başka bir gloss ile değiştirilir (eğer el şekli karışıklık matrisi varsa).

Bu yöntemle, milyonlarca satırlık (Bozuk Gloss \-\> Temiz Türkçe) eğitim verisi oluşturulur. Model, bu verilerle eğitildiğinde, eksik kelimeleri tamamlamayı ve bozuk yapıyı düzeltmeyi "öğrenir".

### **3.2. Model Mimarisi: Ön Eğitimli Dil Modellerinin (PLM) Gücü**

Kullanıcı "Türkçe dil modeli eksikliğinden kaynaklanan context sorunlarını" nasıl ele alacağını sormuştur. Bunun en güçlü cevabı, **Transfer Öğrenmesi** kullanmaktır. Sıfırdan bir Transformer eğitmek yerine, Türkçe dilbilgisine halihazırda hakim olan **mBART-50** veya **mT5** modelleri kullanılmalıdır.2

Bu modeller, terabaytlarca Türkçe metin üzerinde eğitildikleri için, kelimelerin anlamsal ilişkilerini (context) ve morfolojik kurallarını zaten bilmektedir. Örneğin, model ÇAY ve İÇMEK glosslarını gördüğünde, aradaki ilişkinin "Çay içiyorum" veya "Çay içti" olabileceğini, ancak "Çay uçtu" olamayacağını, sahip olduğu dil modeli bilgisi sayesinde bilir. Bizim yapacağımız ince ayar (fine-tuning), modelin bu bilgisini TİD glosslarından gelen sinyallerle tetiklemesini sağlamaktır.

Modelin kodlayıcı (Encoder) kısmına gürültülü glosslar verilir, kod çözücü (Decoder) kısmı ise temiz Türkçe cümleyi üretir. Decoder, doğası gereği bir Dil Modeli (LM) olduğu için, eksik kalan kelimeleri (örneğin özne veya zaman eki) en yüksek olasılıklı Türkçe yapıya göre tamamlar.

---

## **4\. Deneysel Kurgu ve Test Metodolojisi**

Makalenin bilimsel değerini artıracak en önemli bölüm, sistemin performansını objektif olarak ölçen deneysel tasarımdır. Kullanıcının önerdiği "3 kelimelik, 5 kelimelik girdilerle test etme" fikri, literatürde **"Uzunluğa Bağlı Performans Analizi" (Length-Based Bucketing)** olarak bilinir ve sistemin sınırlarını belirlemek için mükemmel bir yöntemdir.32

### **4.1. Kelime Havuzu ve Test Setlerinin Oluşturulması (Bucketing Strategy)**

Test aşaması için sentetik veriden ayrılmış veya elle oluşturulmuş "Golden Set" (Altın Standart) verisi kullanılmalıdır. Bu veri seti, zorluk derecesine göre kategorize edilmelidir (Bucketing).

**Tablo 1: Önerilen Deneysel Test Kategorileri (Buckets)**

| Kategori | Girdi Uzunluğu | Dilbilgisel Özellik | Beklenen Zorluk | Test Amacı |
| :---- | :---- | :---- | :---- | :---- |
| **Kısa Menzil (Short-Horizon)** | 1-3 Gloss | Emir kipleri, basit selamlaşmalar, eksiltili cümleler. | **Yüksek Belirsizlik:** Bağlam az olduğu için modelin halüsinasyon görme riski yüksektir. | Modelin temel kelime dağarcığını ve "default" (varsayılan) tamamlama yeteneğini ölçmek. |
| **Orta Menzil (Medium-Horizon)** | 4-7 Gloss | Standart SOV cümleleri, Özne-Nesne ilişkileri. | **Yapısal Bütünlük:** Özne ve yüklem arasındaki uyumun (agreement) korunması. | Sentaktik yeniden sıralama (reordering) başarısını ölçmek. |
| **Uzun Menzil (Long-Horizon)** | \>8 Gloss | Bağlaçlı cümleler, yan cümlecikler. | **Hata Yayılımı:** Tanıma hatalarının cümlenin geri kalanını bozma riski. | Uzun mesafeli bağımlılıkların (long-term dependency) korunup korunmadığını test etmek. |

### **4.2. Değişken Gürültü Oranları ile Dayanıklılık Testi (Robustness Test)**

Sistemin sadece uzunluğa göre değil, gürültü seviyesine göre de test edilmesi gerekir. Deneysel kurguda şu senaryolar karşılaştırılmalıdır 29:

* **Senaryo A (İdeal):** %0 Silme oranı (Mükemmel tanıma varsayımı).  
* **Senaryo B (Gerçekçi):** %20 Silme oranı (Mevcut sistemin tahmini performansı).  
* **Senaryo C (Zorlu):** %45-%50 Silme oranı (Kötü ışık, hızlı işaretleme).

Bu kurgu, makalede "Gürültü Seviyesine Göre BLEU Skorunun Değişimi" grafiği ile sunulabilir ve sistemin hata toleransı (fault tolerance) kanıtlanabilir.

### **4.3. Değerlendirme Metrikleri: BLEU'nun Ötesine Geçmek**

Sondan eklemeli diller (Türkçe) için standart BLEU metriği yanıltıcı olabilir. Çünkü BLEU, n-gram eşleşmesine bakar. Eğer model "Gidiyorum" yerine "Gideceğim" üretirse, kök aynı olsa da BLEU puanı çok düşebilir, oysa anlam yakındır.3 Bu nedenle makalede çoklu metrik raporlanmalıdır:

1. **BLEU-4:** Klasik n-gram hassasiyeti için.36  
2. **ROUGE-L:** Cümle yapısının ve kelime sırasının doğruluğu için.  
3. **METEOR:** Kelime köklerini ve eşanlamlıları dikkate aldığı için Türkçe morfolojisine daha uygundur.  
4. **BERTScore / COMET:** Anlamsal benzerliği ölçer.37 Transkripsiyon sistemi tam kelimeyi tutturamasa bile anlamı doğru veriyorsa (örneğin "Mutluyum" yerine "Sevinçliyim"), BERTScore bunu ödüllendirir. Bu, işaret dili çevirisinde hayati önem taşır çünkü işaretler kavramsal düzeydedir.  
5. **chrF (Character F-score):** Kelime düzeyinde değil karakter düzeyinde eşleşmeye baktığı için Türkçe eklerin (suffix) doğruluğunu ölçmede çok etkilidir.39

---

## **5\. Bağlam Sorunu ve Halüsinasyon Yönetimi**

Kullanıcının özellikle vurguladığı "Türkçe dil modeli eksikliğinden kaynaklanan context sorunları", sistemin en kritik zayıf noktasıdır. Eğer girdi çok eksikse (örneğin sadece "GİTMEK" gloss'u geldiyse), model bağlamı tamamen uydurabilir (hallucination).

### **5.1. Bağlam-Duyarlı (Context-Aware) Modelleme**

Bu sorunu makalede ele almanın yolu, modelin sadece o anki cümleye değil, önceki cümlelere de bakmasını sağlamaktır.

* **Kayan Pencere (Sliding Window) Yöntemi:** Modele girdi olarak sadece anlık gloss dizisi değil, önceki 1-2 cümlenin üretilmiş metni de verilir.40  
  * *Örnek Girdi:* \<prev\>Merhaba nasılsın?\</prev\> \<current\>İYİ\</current\>  
  * Çıktı: "İyiyim." (Önceki soru 2\. tekil şahıs olduğu için, cevap 1\. tekil şahıs üretilir).  
    Bu yöntem, diyalog bazlı işaret dili verisetlerinde (örneğin sağlık veya bankacılık senaryoları) bağlam sorununu büyük ölçüde çözer.

### **5.2. Kısıtlı Kod Çözme (Constrained Decoding)**

Modelin aşırı halüsinasyon görmesini engellemek için "Lexically Constrained Decoding" 42 yöntemleri kullanılabilir. Bu yöntemde, tanıma modelinden gelen ve güven skoru yüksek olan glossların (örneğin HASTANE), üretilen Türkçe cümlede mutlaka geçmesi (kök olarak) zorunlu kılınır. Bu, modelin akıcılığı sağlarken konudan sapmasını engeller.

---

## **6\. Tartışma ve Sonuç**

Bu çalışma, TİD gibi düşük kaynaklı ve morfolojik açıdan zengin işaret dilleri için, sadece çeviri yapan değil, aynı zamanda eksik bilgiyi tamamlayan gürültüye dayanıklı bir mimari önermektedir. Önerilen yöntem, veri azlığı sorununu sentetik veri üretimi ve transfer öğrenmesi (mBART) ile aşmakta; deneysel kurgu ise sistemin farklı zorluk seviyelerindeki (uzunluk ve gürültü) başarısını çok boyutlu metriklerle (BERTScore, chrF) kanıtlamaktadır.

Yazılacak makalenin özgün değeri, **tanıma hatasını bir "veri özelliği" olarak kabul edip**, modeli bu hatayı düzeltecek şekilde (denoising objective) eğitmekte yatmaktadır. Gelecek çalışmalarda, bu metin tabanlı bağlamın, video işleme modülüne geri beslenerek (feedback loop), tanıma doğruluğunu artırması da önerilebilir.

**Tablo 2: Önerilen Sistem Bileşenleri ve Araçlar**

| Bileşen | Görev | Önerilen Teknoloji / Kütüphane | Kaynak |
| :---- | :---- | :---- | :---- |
| **Morfolojik Analiz** | Türkçe metni gloss köklerine ayırma | **Zemberek** (Java) veya **Zeyrek** (Python) | 25 |
| **Transkripsiyon Modeli** | Gürültülü Glosstan Türkçeye Çeviri | **mBART-50** veya **mT5** (HuggingFace) | 2 |
| **Değerlendirme** | Anlamsal ve yapısal başarı ölçümü | **SacreBLEU**, **BERTScore**, **Unbabel-COMET** | 3 |
| **Veri Üretimi** | Gürültü simülasyonu | Özel Python Scriptleri (Random Deletion/Swap) | 2 |

Bu rapor, kullanıcının akademik makalesini yazması için gereken teorik altyapıyı, metodolojik detayları ve deneysel yol haritasını eksiksiz bir şekilde sunmaktadır.

#### **Works cited**

1. Sign Language Translation: A Survey of Approaches and Techniques \- MDPI, accessed November 20, 2025, [https://www.mdpi.com/2079-9292/12/12/2678](https://www.mdpi.com/2079-9292/12/12/2678)  
2. Improving Sign Language Gloss Translation with Low-Resource Machine Translation Techniques \- Johns Hopkins Computer Science, accessed November 20, 2025, [https://www.cs.jhu.edu/\~xzhan138/papers/SLMT\_Book\_G2T.pdf](https://www.cs.jhu.edu/~xzhan138/papers/SLMT_Book_G2T.pdf)  
3. Text2Gloss: Translation into Sign Language Gloss with Transformers \- Stanford University, accessed November 20, 2025, [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/JennaSaraMansuetoLukeCBabbitt.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/JennaSaraMansuetoLukeCBabbitt.pdf)  
4. (PDF) Turkish Sign Language Grammar \- ResearchGate, accessed November 20, 2025, [https://www.researchgate.net/publication/314041221\_Turkish\_Sign\_Language\_Grammar](https://www.researchgate.net/publication/314041221_Turkish_Sign_Language_Grammar)  
5. TOPICS IN TURKISH SIGN LANGUAGE (TÜRK ĠġARET DĠLĠ – TĠD) SYNTAX: VERB MOVEMENT, NEGATION AND CLAUSAL ARCHITECTURE KADĠ \- Digital Archive, accessed November 20, 2025, [https://digitalarchive.library.bogazici.edu.tr/bitstreams/4263702d-80bc-4c6e-b0a6-73e324c771fe/download](https://digitalarchive.library.bogazici.edu.tr/bitstreams/4263702d-80bc-4c6e-b0a6-73e324c771fe/download)  
6. An Open, Extendible, and Fast Turkish Morphological Analyzer \- ACL Anthology, accessed November 20, 2025, [https://aclanthology.org/R19-1156.pdf](https://aclanthology.org/R19-1156.pdf)  
7. Sign Languages and Aspects of Turkish Sign Language (TİD) \- EFD / JFL, accessed November 20, 2025, [https://arastirmax.com/en/system/files/dergiler/263/makaleler/30/1/arastirmax-sign-languages-and-aspects-turkish-sign-language-tid.pdf](https://arastirmax.com/en/system/files/dergiler/263/makaleler/30/1/arastirmax-sign-languages-and-aspects-turkish-sign-language-tid.pdf)  
8. Karabüklü & Wilbur: Marking various aspects in Turkish Sign Language \- John Benjamins, accessed November 20, 2025, [https://www.jbe-platform.com/docserver/fulltext/sll.20006.kar.pdf?expires=1632918303\&id=id\&accname=id22648986\&checksum=A8D1AE032202EFE594AC3C95A4516AC9](https://www.jbe-platform.com/docserver/fulltext/sll.20006.kar.pdf?expires=1632918303&id=id&accname=id22648986&checksum=A8D1AE032202EFE594AC3C95A4516AC9)  
9. Aspects of Türk I˙saret Dili \- (Turkish Sign Language) \- The Swiss Bay, accessed November 20, 2025, [https://theswissbay.ch/pdf/Books/Linguistics/Mega%20linguistics%20pack/Sign%20Languages/Turkish%20Sign%20Language%3B%20Aspects%20of%20T%C3%BCrk%20%C4%B0%C5%9Faret%20Dili%20%28Zeshan%29.pdf](https://theswissbay.ch/pdf/Books/Linguistics/Mega%20linguistics%20pack/Sign%20Languages/Turkish%20Sign%20Language%3B%20Aspects%20of%20T%C3%BCrk%20%C4%B0%C5%9Faret%20Dili%20%28Zeshan%29.pdf)  
10. Order of the major constituents in sign languages: implications for all language \- PMC \- NIH, accessed November 20, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4026690/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4026690/)  
11. (PDF) An Experimental Approach to Word Order in Turkish Sign Language \* \- ResearchGate, accessed November 20, 2025, [https://www.researchgate.net/publication/343125490\_An\_Experimental\_Approach\_to\_Word\_Order\_in\_Turkish\_Sign\_Language](https://www.researchgate.net/publication/343125490_An_Experimental_Approach_to_Word_Order_in_Turkish_Sign_Language)  
12. AN ANALYSIS OF TURKISH SIGN LANGUAGE (TİD) PHONOLOGY AND, accessed November 20, 2025, [https://etd.lib.metu.edu.tr/upload/12609654/index.pdf](https://etd.lib.metu.edu.tr/upload/12609654/index.pdf)  
13. Approaching Sign Language Gloss Translation as a Low-Resource Machine Translation Task \- ACL Anthology, accessed November 20, 2025, [https://aclanthology.org/2021.mtsummit-at4ssl.7/](https://aclanthology.org/2021.mtsummit-at4ssl.7/)  
14. \[2105.07476\] Data Augmentation for Sign Language Gloss Translation \- arXiv, accessed November 20, 2025, [https://arxiv.org/abs/2105.07476](https://arxiv.org/abs/2105.07476)  
15. \[PDF\] A Hybrid Translation System from Turkish Spoken Language to Turkish Sign Language | Semantic Scholar, accessed November 20, 2025, [https://www.semanticscholar.org/paper/A-Hybrid-Translation-System-from-Turkish-Spoken-to-Kayahan-G%C3%BCng%C3%B6r/c74b53f98c51b6588201ca71123a59bb1e893d23](https://www.semanticscholar.org/paper/A-Hybrid-Translation-System-from-Turkish-Spoken-to-Kayahan-G%C3%BCng%C3%B6r/c74b53f98c51b6588201ca71123a59bb1e893d23)  
16. A Hybrid Translation System From Turkish SpokenLanguage to Turkish Sign Language | PDF | Verb \- Scribd, accessed November 20, 2025, [https://www.scribd.com/document/948654845/A-Hybrid-Translation-System-From-Turkish-SpokenLanguage-to-Turkish-Sign-Language](https://www.scribd.com/document/948654845/A-Hybrid-Translation-System-From-Turkish-SpokenLanguage-to-Turkish-Sign-Language)  
17. Translation of Sign Language Glosses to Text Using Sequence-to-Sequence Attention Models \- ResearchGate, accessed November 20, 2025, [https://www.researchgate.net/publication/340689060\_Translation\_of\_Sign\_Language\_Glosses\_to\_Text\_Using\_Sequence-to-Sequence\_Attention\_Models](https://www.researchgate.net/publication/340689060_Translation_of_Sign_Language_Glosses_to_Text_Using_Sequence-to-Sequence_Attention_Models)  
18. Grammar Error Correction using Seq2Seq \- ijarcce, accessed November 20, 2025, [https://ijarcce.com/wp-content/uploads/2020/08/IJARCCE.2020.9652.pdf](https://ijarcce.com/wp-content/uploads/2020/08/IJARCCE.2020.9652.pdf)  
19. E-TSL: A Continuous Educational Turkish Sign Language Dataset with Baseline Methods, accessed November 20, 2025, [https://arxiv.org/html/2405.02984v1](https://arxiv.org/html/2405.02984v1)  
20. TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2505.07890v4](https://arxiv.org/html/2505.07890v4)  
21. Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation, accessed November 20, 2025, [https://arxiv.org/html/2405.04164v1](https://arxiv.org/html/2405.04164v1)  
22. Gloss2Text: Sign Language Gloss translation using LLM's and Semantically Aware Label Smoothing \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2407.01394v1](https://arxiv.org/html/2407.01394v1)  
23. Combining Denoising Autoencoders with Contrastive Learning to fine-tune Transformer Models \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2405.14437v1](https://arxiv.org/html/2405.14437v1)  
24. FRUSTRATINGLY EASY DATA AUGMENTATION FOR LOW-RESOURCE ASR \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2509.15373v1](https://arxiv.org/html/2509.15373v1)  
25. Zemberek \- Google Code, accessed November 20, 2025, [https://code.google.com/archive/p/zemberek](https://code.google.com/archive/p/zemberek)  
26. ahmetaa/zemberek-nlp: NLP tools for Turkish. \- GitHub, accessed November 20, 2025, [https://github.com/ahmetaa/zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp)  
27. Zeyrek — Zeyrek 0.1.0 documentation, accessed November 20, 2025, [https://zeyrek.readthedocs.io/en/latest/](https://zeyrek.readthedocs.io/en/latest/)  
28. Preprocessing Turkish Texts Using Zemberek with Python | by Sema Şahin \- Medium, accessed November 20, 2025, [https://medium.com/@semasahin934/preprocessing-turkish-texts-using-zemberek-with-python-6f8e47ff8f8c](https://medium.com/@semasahin934/preprocessing-turkish-texts-using-zemberek-with-python-6f8e47ff8f8c)  
29. Generation of Synthetic Sign Language Sentences \- ISCA Archive, accessed November 20, 2025, [https://www.isca-archive.org/iberspeech\_2021/villaplana21\_iberspeech.pdf](https://www.isca-archive.org/iberspeech_2021/villaplana21_iberspeech.pdf)  
30. Towards the Development of Balanced Synthetic Data for Correcting Grammatical Errors in Arabic \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2502.05312v1](https://arxiv.org/html/2502.05312v1)  
31. Synthetic Data Generation for Grammatical Error Correction with Tagged Corruption Models \- ACL Anthology, accessed November 20, 2025, [https://aclanthology.org/2021.bea-1.4.pdf](https://aclanthology.org/2021.bea-1.4.pdf)  
32. Impact of sentence length on translation quality \- ResearchGate, accessed November 20, 2025, [https://www.researchgate.net/figure/mpact-of-sentence-length-on-translation-quality\_fig2\_388994863](https://www.researchgate.net/figure/mpact-of-sentence-length-on-translation-quality_fig2_388994863)  
33. Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights \- arXiv, accessed November 20, 2025, [https://arxiv.org/pdf/2507.01532?](https://arxiv.org/pdf/2507.01532)  
34. Text-Driven Diffusion Model for Sign Language Production \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2503.15914v1](https://arxiv.org/html/2503.15914v1)  
35. Beyond BLEU: Training Neural Machine Translation with Semantic Similarity, accessed November 20, 2025, [https://aclanthology.org/P19-1427/](https://aclanthology.org/P19-1427/)  
36. Translation of Sign Language Glosses to Text Using Sequence-to-Sequence Attention Models \- Signal Processing and Communications Lab, accessed November 20, 2025, [http://xanthippi.ceid.upatras.gr/HealthSign/resources/Publications/sitis\_paper\_25\_10.pdf](http://xanthippi.ceid.upatras.gr/HealthSign/resources/Publications/sitis_paper_25_10.pdf)  
37. BERTScore: Evaluating Text Generation with BERT \- OpenReview, accessed November 20, 2025, [https://openreview.net/pdf?id=SkeHuCVFDr](https://openreview.net/pdf?id=SkeHuCVFDr)  
38. COMET for Low-Resource Machine Translation Evaluation: A Case Study of English-Maltese and Spanish-Basque \- ACL Anthology, accessed November 20, 2025, [https://aclanthology.org/2024.lrec-main.315/](https://aclanthology.org/2024.lrec-main.315/)  
39. Effective Sign Language Evaluation via SignWriting \- arXiv, accessed November 20, 2025, [https://arxiv.org/abs/2410.13668](https://arxiv.org/abs/2410.13668)  
40. SCOPE: Sign Language Contextual Processing with Embedding from LLMs \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2409.01073v1](https://arxiv.org/html/2409.01073v1)  
41. Continuous Sign Language Recognition through a Context-Aware Generative Adversarial Network \- MDPI, accessed November 20, 2025, [https://www.mdpi.com/1424-8220/21/7/2437](https://www.mdpi.com/1424-8220/21/7/2437)  
42. ASR Error Correction using Large Language Models \- arXiv, accessed November 20, 2025, [https://arxiv.org/html/2409.09554v2](https://arxiv.org/html/2409.09554v2)  
43. COMET for Low-Resource Machine Translation Evaluation: A Case Study of English-Maltese and Spanish-Basque \- ACL Anthology, accessed November 20, 2025, [https://aclanthology.org/2024.lrec-main.315.pdf](https://aclanthology.org/2024.lrec-main.315.pdf)