Ben Türkçe işaret dili için transformer tabanlı bir model eğittim. Model gerçek zamanlı olarak video veya kameradan işaret dili hareketlerini tanıyarak kelimeleri veriyor. Modelin kaçırdığı kelimeler oluyor hala çünkü eğitim verisi çok küçüktü. Şimdi de buradan gelecek kelimelerden oluşturulan cümlelerin içerik açısından tamamlayacak ve transcriptionları gerçek çevireye dönüştürecek bir trankription sistem tasarlamak istiyorum. Akademik olarak çalışacağım ve bir adet makale çıkarmayı düşünüyorum. Bu transkription sisteminin performansını test etmek için bir kelime havuzundan nasıl bir deneysel kurgu yapabilirim.

https://github.com/SIYAKS-ARES/sign-transcriber

Bu konuda türkçe dil modeli olmadığı için birçok context eksiği yaşıyacağız ve bunu da makalede dile getireceğiz.

1. Modelin girdi-çıktı tanımı

Modelin girdileri video veya kamera tarafından alınacak model sırasıyla gelen kelimeleri sırasıyla çevirecek. Burada şöyle transkriptionlar olacak (uzunluk değişken ama biz üç beş kelime ile başlayacağız):

Örnek:

TRANSKRİPSİYON:BEN BİR HAFTA ÖNCE ARABA VERMEK BİR HAFTA İKİ HAFTA GEÇMEK TAMİR TAMİR UZUN SÜRMEK

Çeviri:Geçen hafta arabamı tamire verdim. Hala bitmedi, tamiri uzun sürecek.

Modelden gelen çıktılar videodaki gibi sıralı olacak veya öyle kabul edilecek ama bazen bir iki kelime tanınmadan atlanabilir. Mesela "TAMİR" kelimesi model tarafından tanınamamış olup transkription bu şekilde gelmiş olabilir:

TRANSKRİPSİYON:BEN BİR HAFTA ÖNCE ARABA VERMEK BİR HAFTA İKİ HAFTA GEÇMEK UZUN SÜRMEK

Sistemimiz bu çıktı için "TAMİR" kelimesinin eksikliğini tanıyacak veya oraya mantıksal olarak gelebilecek kelimeler ile çeviriler sunacak ama eksik tanınmamış kelimeleri fark edip eklemenin yanında asıl hedef tam gelen transriptiondan doğru bir çeviri üretebilmek.

2. Eksiklik türü

Yukarıda bahsettiğim gibi bu kelimeler rastgele olabilir ama senaryolara bağlaç, fiil, nesne/tümleç eklenebilir.

3. Transkripsiyon sisteminin hedef çıktısı

Hedef çıktımız eksik kelime ve transcription a rağmen doğru doğal Türkçe dilini üretebilmek.

Örneğin:

TRANSKRİPSİYON:ANNE BURDA YOK AVRUPA ORADA ÇALIŞMAK

Çeviri:Annem burada değil, Avrupa'da çalışıyor.

Transkriptionundan bu doğru çeviriyi elde edebilmek. Çıktı birden fazla alternatif üretebilse daha iyi olur şimdilik.

Bunun örnekleri için elimde güzel bir veri var. "https://tidsozluk.aile.gov.tr/" sitesinden tüm kelimeler için scrapping yaptım. Şu şekilde iki bin adet kelime var:

{

    "kelime": "Ağaç",

    "ingilizce_karsiliklar": [],

    "anlamlar": [

    {

    "sira_no": 1,

    "tur": "Ad",

    "aciklama": "Güncel Türk İşaret Dili Sözlüğü\nSözlük Kullanımı\nHakkında\nProje Ekibi\nİletişim\nEN\nAğaç\nTree, Timber , Wood , Post\n1) Ad Bitki Bilimi Meyve verebilen, gövdesi odun veya kereste olmaya elverişli bulunan ve uzun yıllar yaşayabilen bitki\nÖrnek :",

    "ornek": {

    "transkripsiyon": "AĞAÇ O UZUN YAŞAMAK OLMAK\nÇeviri:\nAğaç uzun yaşar.",

    "ceviri": "Ağaç uzun yaşar."

    }

    },

    {

    "sira_no": 1,

    "tur": "Ad",

    "aciklama": "Ağaç\nTree, Timber , Wood , Post\n1) Ad Bitki Bilimi Meyve verebilen, gövdesi odun veya kereste olmaya elverişli bulunan ve uzun yıllar yaşayabilen bitki\nÖrnek :",

    "ornek": {

    "transkripsiyon": "AĞAÇ O UZUN YAŞAMAK OLMAK\nÇeviri:\nAğaç uzun yaşar.",

    "ceviri": "Ağaç uzun yaşar."

    }

    }

    ]

}

4. Doğru cevabı nereden belirleyeceğiz?

Doğru cevabı bulabilme asıl hedefimiz zaten bunun bir sistem ile insan kullanmadan yapılıp yapılamayacağını araştırıyoruz.

5. Değerlendirme metrikleri

Ne kullanılabilir bilmiyorum açıkçası ama hem sözdizimsel hem anlamsal iki aşamalı bir değerlendirme mantıklı olabilir.

6. Deney varyasyonları

Hedefim hem farklı uzunluktaki kelime dizilerinden (3, 5, 7 kelimelik) cümle üretme performansını karşılaştırmak, hem eksik kelime oranını da manipüle etmek, hemde bu eksik çevirilerden ve transcriptionlardan gerçek çeviri elde edip edemeyeceğimi araştırıp denemek sonuçları paylaşmak. Öncül bir çalışma olacak bu alanda pek çalışma yok.

7. Kelime havuzu boyutu

Şuan model 226 kelime üzerinden eğitildi çıktılar oradan alınacak ama biz mock datalar ile de çalışabiliriz scrape ettiğim verileri de kullanabiliriz ama dengeli bir veri setinden bahsetmek mantıklı olmayabilir. Zamanla artacağını söyleyebilirim.

8. Dil modeli tarafında kullanacağın yapı

Açık kaynak modeller ile LLM ler (gemini claude chatgpt) kullanabiliriz açıkçası türkçe olarak eğitilmiş bir model olmadığı için bu konuda zorlanacağız ama elimizdekiler ile en iyi sonucu elde ettiğimiz sürece sorun olmaz. Makalede bu eksiklikler dile getirilecektir zaten.

9. Deneyin amacı

Ütopik olabilir ama asıl amaç; işaret dilinden gelen transkript edilmiş işaret dili cümlelerinin eksik çevirilmiş gözden kaçmış kelimelerini tahmin edip bunlar için öneriler öneren bu öneriler ile gerçek türkçe doğal diline çeviren bir sistem kurmak. Bunu yaparken sonuçları deneyler ile elde edip bu sonuçlardan oluşan bir makale yayınlamak.

Örnek senaryo:

TRANSKRİPSİYON: ESKİ TELEVİZYON ATMAK TELEVİZYON ALMAK

Çeviri: Eski televizyonu atıp yenisini aldım.

Buradaki transkripsiyon içerisine yeni kelimesi model tarafından algılanamayıp atlanmış. Orijinalde gelmiş olması gereken transkripsiyon: "ESKİ TELEVİZYON ATMAK YENİ TELEVİZYON ALMAK" şeklinde olmalıydı.

Bizim sistemimize gelen bu eksik çeviriyi sistem alıp bağlamdan burada eksik bir kelime olduğunu çıkaracak, bu eksik kelime için olası kelimeler önerecek. Çıktıda eksik kelime olabilir, bu eksik kelime "YENİ" kelimesi olabilir, "YENİ" kelimesi eklenmiş çeviri "Eski televizyonu atıp yenisini aldım." şeklinde olabilir. Diye çıktı vermesini hedefliyoruz gibi düşünebilirsin. Bunu elde etmeye çalışırken seninde "Netleştirmek ve Araştırmak Gereken Noktalar" altında bahsettiğin kısımları araştırıp deneyler elde edeceğiz.

Örneğin üç kelimelik, beş kelimelik gelenlerdeki performansının deneysel kurgusunu yapabilir misin?

Öncesinde bu görevi gerçekleştirmek için benim unuttuğum ve netleşemeyen noktaları netleştirebilmemiz için bana sorular sor ve anladığın üzerine araştırma yapacağın noktaları söyle. Ben hem sorularını yanıtlayacağım hemde yanlış anlaşılmış olabilecek noktaları düzelteceğim.
