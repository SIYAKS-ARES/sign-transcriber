<Sistem_Talimatları>
    <Rol_ve_Uzmanlık>
        Sen, Doğal Dil İşleme (NLP), Bilgisayarlı Görü (Computer Vision) ve özellikle İşaret Dili İşleme (Sign Language Processing - SLP) alanlarında uzmanlaşmış Kıdemli Bir Akademik Araştırmacısın.
        Aynı zamanda Büyük Dil Modellerinin (LLM) eksik veri tamamlama (imputation) ve nöral makine çevirisi (NMT) konularındaki uygulamalarına derinlemesine hakimsin.
        Görevin, aşağıda belirtilen araştırma önerisi için kapsamlı bir literatür taraması yapmak, veri kaynaklarını tespit etmek ve sağlam bir deneysel metodoloji kurgulamaktır.
    </Rol_ve_Uzmanlık>

    <Araştırma_Görevi>
        Türk İşaret Dili (TİD) gloslarından (işaret dizileri) doğal Türkçeye çeviri yapan, ancak gloslardaki yapısal/anlamsal boşlukları (eksik fiil, bağlaç, nesne) LLM tabanlı yöntemlerle dolduran bir sistem için zemin araştırması yapmalısın.
    </Araştırma_Görevi>

    <Araştırma_Adımları_ve_Kısıtlamalar>
        Araştırmanı aşağıdaki 4 ana başlık altında, "Adım Adım Düşünerek" (Chain of Thought) yapılandır:

        1. VERİ KAYNAKLARI VE DERLEM (CORPUS) ANALİZİ:
           - TİD için video/transkripsiyon/çeviri üçlüsünü (tercihen Glos formatında) içeren tüm akademik veri setlerini, veritabanlarını ve tezleri listele.
           - Erişim kısıtlı olanlar dahil olmak üzere (TİD Corpus, BosphorusSign vb.) kaynakların içeriğini, büyüklüğünü ve erişim yöntemlerini belirt.
           - Eğer doğrudan TİD kaynağı yoksa, dil ailesi veya yapısal özellikleri TİD'e en çok benzeyen diğer İşaret Dili korpuslarını (örn: Alman İşaret Dili - DGS) alternatif olarak öner.
           - Glos notasyon standartlarını (HamNoSys, Stokoe vb. yerine dilbilimsel metin tabanlı gloslar) incele.

        2. EKSİK KELİME TAMAMLAMA (MISSING WORD IMPUTATION) LİTERATÜRÜ:
           - İşaret dillerinde veya düşük kaynaklı dillerde "Missing Word Imputation" (Eksik Kelime Tamamlama) üzerine yapılmış güncel LLM çalışmalarını bul.
           - Özellikle fiil, bağlaç ve nesne eksikliklerinin tamamlanmasında kullanılan mimarileri (BERT, T5, GPT türevleri) ve başarı oranlarını analiz et.
           - "Eksiklik Oranı" (Missing Rate) ve "Eksiklik Türü" (Random vs Structural) üzerine yapılmış deneysel kurguları özetle.

        3. DENEYSEL KURGU VE METODOLOJİ ÖNERİSİ:
           - Aşağıdaki akışı temel alan bir deney tasarımı oluştur:
             (Girdi: Eksik TİD Glos) -> [LLM Tamamlama Modülü] -> (Çıktı: Tam Glos) -> [Çeviri Modülü] -> (Çıktı: Doğal Türkçe).
           - Sentetik veri üretimi (Mock Data Generation) için TİDSözlük veya benzeri kaynaklardan alınan tam cümlelerden kasıtlı hata enjeksiyonu (fiil çıkarma, özne silme) yöntemlerini detaylandır.

        4. PERFORMANS METRİKLERİ VE DEĞERLENDİRME:
           - Bu sistemin başarısını ölçmek için aşağıdaki metriklerin nasıl kullanılacağını açıkla:
             * Yapısal Doğruluk: Perplexity (PPL), Accuracy.
             * Çeviri Kalitesi (Sözcük Bazlı): BLEU, ROUGE.
             * Anlamsal Benzerlik (Derin Öğrenme Bazlı): METEOR, BERTScore.
           - İnsan değerlendirmesi (Human Evaluation) için akıcılık ve doğruluk kriterlerini içeren bir şablon öner.
    </Araştırma_Adımları_ve_Kısıtlamalar>

    <Çıktı_Formatı>
        Yanıtını, akademik bir makalenin "Literatür Taraması" ve "Yöntem" bölümlerine altlık oluşturacak şekilde, kaynak atıfları içeren detaylı bir Markdown raporu olarak sun. Her bölümün sonunda o konuyla ilgili en kritik 3 makaleyi/tezi listele.
    </Çıktı_Formatı>
</Sistem_Talimatları>

<?xml version="1.0" encoding="UTF-8"?>
<ResearchProposal>
    <Feedback>
        <Message>Harika bir çalışma! Türk İşaret Dili (TİD) transkripsiyonu ve çevirisi alanında transformer tabanlı bir modelle öncü bir çalışma yapmanız takdire şayan. Verdiğiniz detaylar, başlangıç promptunuzu hem araştırma hem de deneysel kurgu açısından çok daha güçlü hale getirdi.</Message>
    </Feedback>

    `<ImprovedMainPrompt>`
        `<Description>`İyileştirilmiş prompt, hem ihtiyacınız olan veri kaynaklarını bulmaya odaklanacak hem de makaleniz için teorik zemin oluşturacak çalışmaları araştırmaya yöneltecektir.`</Description>`
        `<PromptText>`
            "Türk İşaret Dili (TİD) video/transkripsiyon-çeviri çiftlerini (glos formatı tercihli) içeren, akademik/dilbilimsel nitelikteki veri setlerini, veri tabanlarını, tezleri ve makaleleri (erişim kısıtlı kaynaklar dahil) derinlemesine araştır. Ayrıca, eksik kelime tamamlama (Missing Word Imputation) ve dilbilimsel olarak farklı diller arası anlamsal çeviri konularında LLM (Büyük Dil Modeli) tabanlı deneysel kurgu ve performans değerlendirme metrikleri (özellikle BLEU, METEOR ve anlamsal benzerlik metrikleri) kullanan güncel çalışmaları bul. Amaç, TİD glos transkripsiyonundaki yapısal/anlamsal boşlukları (fiil, bağlaç, nesne) doldurarak doğal Türkçe diline akıcı çeviri yapan bir sistem için zemin oluşturmaktır."
        `</PromptText>`
    `</ImprovedMainPrompt>`

    `<ResearchPoints>`
        `<Goal>`Netleştirmek ve Araştırmak Gereken Noktalar`</Goal>`

    `<Section id="1" title="Veri ve Kaynak Araştırması">`
            `<Item>`
                `<Need>`TİD Veri Setleri`</Need>`
                `<Purpose>`Modelin eğitimi ve test verisi olarak kullanılacak, özellikle video/transkripsiyon/çeviri üçlüsünü içeren hazır veri setleri.`</Purpose>`
                `<Topic>`TİD Corpus, TİD Veri Seti, Turkish Sign Language Parallel Corpus, ISL Corpus (Özellikle Dil Aileleri Benzer Olanlar).`</Topic>`
            `</Item>`
            `<Item>`
                `<Need>`Glos Formatı`</Need>`
                `<Purpose>`Transkripsiyon formatını standardize etmek.`</Purpose>`
                `<Topic>`İşaret Dilleri için Standart Glos Kuralları, TİD'de Kullanılan Dilbilimsel Kısaltmalar ve Notasyonlar.`</Topic>`
            `</Item>`
            `<Item>`
                `<Need>`Yazılı Kaynaklar`</Need>`
                `<Purpose>`Makalenin literatür taraması ve teorik zeminini oluşturmak.`</Purpose>`
                `<Topic>`Türk İşaret Dili Söz Dizimi (Syntax), İşaret Dillerinde Konu-Yorum (Topic-Comment) Yapısı, İşaret Dilinden Sözlü Dile Nöral Çeviri.`</Topic>`
            `</Item>`
        `</Section>`

    `<Section id="2" title="Deneysel Kurgu ve Metrikler">`
            `<ProblemDefinition>`
                `<ImputationStep>`T_eksik (Model çıktısı glos) -> T_tamam (Tahmini tam glos)`</ImputationStep>`
                `<TranslationStep>`T_tamam (Tahmini tam glos) -> C_doğal (Doğal Türkçe Cümle)`</TranslationStep>`
            `</ProblemDefinition>`

    `<ExperimentMatrix>`
                `<Experiment type="Eksik Tamamlama Performansı">`
                    `<Variations>`Eksiklik Türü: Fiil, Bağlaç, Nesne/Tümleç. Eksiklik Oranı: %20, %40 (rastgele veya yapısal olarak).`</Variations>`
                    `<MetricRecommendations>`Perplexity (PPL), Accuracy (Tamamlanan kelimenin doğru olup olmadığı), Macro F1-Score (Özellikle farklı eksiklik türleri için).`</MetricRecommendations>`
                `</Experiment>`
                `<Experiment type="Çeviri Performansı (Doğallık)">`
                    `<Variations>`Girdi Uzunluğu: 3, 5, 7 kelimelik glos dizileri. LLM Kullanımı: Zero-shot, Few-shot (Örneklerle), Fine-tuning (Eğer veri yeterliyse).`</Variations>`
                    `<MetricRecommendations>`BLEU (Sözcük örtüşmesi), METEOR (Anlamsal/Kök örtüşmesi), BERTScore veya BARTScore (Derin anlamsal benzerlik).`</MetricRecommendations>`
                `</Experiment>`
                `<Experiment type="Nihai Sistem Performansı">`
                    `<Variations>`Eksiklik içeren TİD-Glos -> Nihai Türkçe Çeviri.`</Variations>`
                    `<MetricRecommendations>`Human Evaluation (İnsan Değerlendirmesi): Akıcılık, Doğruluk ve Gramer Ağırlıklı anket. LLM'lerin sunduğu alternatif çevirilerin kalitesini ölçmek için idealdir.`</MetricRecommendations>`
                `</Experiment>`
            `</ExperimentMatrix>`
        `</Section>`
    `</ResearchPoints>`

    `<MethodologyDraft>`
        `<Title>`Deneysel Kurgu Taslağı (Methodology)`</Title>`

    `<Phase id="A" title="Sentetik Veri Seti Oluşturma (Mock Data Generation)">`
            `<Step name="Temel Veri Seti">`TİDSözlük'ten scrape ettiğiniz tam glos/çeviri çiftlerini alın. (Örn: AĞAÇ O UZUN YAŞAMAK OLMAK -> Ağaç uzun yaşar.)`</Step>`
            `<Step name="Hata Enjeksiyonu">`
                `<Description>`Bu tam glos transkripsiyonlardan, belirlediğiniz varyasyonlara göre kasıtlı olarak kelimeler çıkarın (simüle edilmiş model hatası).`</Description>`
                `<Details>`
                    `<Type>`Fiil Çıkarımı: (Örn: UZUN YAŞAMAK yerine YAŞAMAK kelimesini çıkarın.)`</Type>`
                    `<Type>`Bağlaç/Edat Çıkarımı: (TİD'de az olsa da, Türkçe çevirideki işlevi olan kelimeler.)`</Type>`
                    `<Type>`Nesne/Özne Tamamlayıcısı Çıkarımı: (Örn: YENİ kelimesinin çıkarılması.)`</Type>`
                `</Details>`
            `</Step>`
        `</Phase>`

    `<Phase id="B" title="Deney Yürütme">`
            `<ModelType name="Baseline Modeller">`
                `<Strategy>`Doğrudan Çeviri (Kontrol Grubu): Eksik glos'u, tamamlama aşaması olmadan doğrudan LLM'e çevirtin.`</Strategy>`
                `<Strategy>`Basit Dil Modeli: Eksik kelime için sadece N-gram (veya basit bir Markov/RNN modeli) kullanarak boşluğu doldurmaya çalışın.`</Strategy>`
            `</ModelType>`
            `<ModelType name="Ana Deney (LLM ile Tamamlama ve Çeviri)">`
                `<PromptEngineering>`
                    `<Stage1 description="Tamamlama">`"Aşağıdaki TİD Glos transkripsiyonunda eksik olabilecek bir veya daha fazla kelimeyi bağlama uygun olarak tahmin et. Eksik kelime türünü (fiil, nesne vb.) belirt ve yeni, tam glos transkripsiyonu üret."`</Stage1>`
                    `<Stage2 description="Çeviri">`"Verilen tam TİD Glos transkripsiyonunu akıcı ve doğal Türkçe diline çevir. Birden fazla çeviri alternatifi sun."`</Stage2>`
                `</PromptEngineering>`
            `</ModelType>`
        `</Phase>`

    `<Phase id="C" title="Sonuç Analizi">`
            `<Action>`Hata enjekte edilmiş glosların, sisteminizden çıkan nihai çevirileri ile orijinal gerçek Türkçe çevirilerini karşılaştırarak BLEU, METEOR ve BERTScore değerlerini hesaplayın.`</Action>`
            `<Action>`Makalenizde, özellikle fiil ve bağlaç eksikliğinin çeviri performansını nasıl ve ne kadar düşürdüğünü (BLEU puanları ile) gösterin.`</Action>`
        `</Phase>`
    `</MethodologyDraft>`

    `<NextStepProposal>`
        `<Question>`Sizin için hemen ilk aşama olan veri seti, glos kuralları ve LLM tabanlı eksik kelime tamamlama konularında derinlemesine bir araştırma başlatabilirim. Bu iyileştirilmiş prompt ile araştırmaya başlamamı ister misiniz?`</Question>`
    `</NextStepProposal>`
`</ResearchProposal>`