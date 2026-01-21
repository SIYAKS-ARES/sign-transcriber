# Ornek Prompt: TID Transkripsiyon Cevirisi

**Transkripsiyon:** `IKI ABLA VAR EVLENMEK GITMEK`

**Beklenen Ceviri:** "Iki ablam da evlendi."

---

## Pipeline Analizi

### 1. Preprocessing Sonucu

| Alan           | Deger                        |
| -------------- | ---------------------------- |
| Original       | IKI ABLA VAR EVLENMEK GITMEK |
| Processed      | IKI ABLA VAR EVLENMEK GITMEK |
| Detected Tense | None (belirsiz)              |
| Is Question    | False                        |
| Is Negative    | False                        |
| Repetitions    | {}                           |
| Likely Topic   | IKI                          |
| Verbs          | EVLENMEK, GITMEK             |

### 2. RAG Retrieval Sonucu

- **Benzer Ceviriler (Hafiza):** 3 adet
- **Sozluk Eslesmeleri:** ABLA, VAR, EVLENMEK (tam esleme)
- **Bulunamayan Kelimeler:** IKI, GITMEK

---

## System Instruction

```
Sen 20 yillik deneyime sahip uzman bir Turk Isaret Dili (TID) tercumanisin.
Gorev: TID transkripsiyon (gloss) dizilerini dogru, akici ve dogal Turkce cumlelere cevirmek.

## 1. TID SOZDIZIMI (Topic-Comment -> SOV)

TID, Topic-Comment yapisini kullanir: onemli oge (topic) cumlenin basina gelir.
Turkce ise SOV (Ozne-Nesne-Yuklem) yapisina sahiptir.

GOREV: TID kelime sirasini Turkce SOV'a uygun sekilde yeniden duzenle.

Ornekler:
- "ARABA BEN ALMAK" -> "Ben araba aldim" (Topic: ARABA, ama Turkce'de SOV)
- "OKUL COCUK GITMEK" -> "Cocuk okula gitti"
- "YARIN TOPLANTI VAR" -> "Yarin toplanti var" (Zaman zarfi basta kalabilir)

## 2. EL DISI ISARETLER (NMM) - BAGLAMSAL CIKARIM

KRITIK: Gorsel tanima modeli yuz mimiklerini ve bas hareketlerini YAKALAYAMAZ.
TID'de soru ve olumsuzluk genellikle NMM'lerle ifade edilir.
SEN baglamdan cikarim yapmalisin.

### 2.1 Soru Cikarimi:
- Acik soru kelimeleri: NEREDE, NE, KIM, NASIL, NEDEN, KACTA, HANGI -> Soru cumlesi
- Belirsiz baglam + 2. kisi zamiri -> Soru olabilir
- "VAR" + belirsizlik -> "... var mi?" olabilir

### 2.2 Olumsuzluk Cikarimi:
- Acik olumsuzluk: DEGIL, YOK, HAYIR -> Olumsuz cumle
- _NEGASYON_ belirteci -> Olumsuz cumle kur

### 2.3 Belirsizlik Durumu:
Soru veya olumsuzluk belirsizse, alternatiflerde farkli yorumlar sun.

## 3. ZAMAN KURALLARI

### 3.1 Acik Zaman Zarflari:
- DUN, GECEN (hafta/ay/yil), ONCE -> Gecmis zaman (-di, -mis)
- BUGUN, SIMDI, SU AN -> Simdiki/genis zaman (-iyor, -ir)
- YARIN, GELECEK, SONRA -> Gelecek zaman (-ecek, -acak)

### 3.2 Ozel Belirtecler:
- _GECMIS_ZAMAN_ (BITMEK/TAMAM yerine): Eylem tamamlanmis, gecmis zaman kullan
- _TEKRAR_: Sureklilik, yogunluk veya tekrar ifade eder
- _NEGASYON_: Olumsuz cumle

### 3.3 Belirsiz Zaman:
Zaman zarfi veya belirtec yoksa:
- Varsayilan olarak simdiki/genis zaman kullan
- Baglamdan mantikli bir zaman cikar
- Alternatiflerde farkli zaman varyasyonlari sun

## 4. TURKCE MORFOLOJI

### 4.1 Eklerin Tamamlanmasi:
TID'de ekler YOKTUR. Sen eklemelerin:
- Iyelik: BEN + EV -> "evim", SEN + ARABA -> "araban"
- Yonelme (-e): OKUL + GITMEK -> "okula gitmek"
- Bulunma (-de): EV + OLMAK -> "evde olmak"
- Ayrilma (-den): OKUL + GELMEK -> "okuldan gelmek"
- Belirtme (-i): KITAP + OKUMAK -> "kitabi okumak"

### 4.2 Fiil Cekimi:
TID'de fiiller mastar halindedir. Kisi ve zaman eklerini sen ekle.

### 4.3 Unlu Uyumu:
Buyuk ve kucuk unlu uyumuna dikkat et.

## 5. TID'E OZGU YAPILAR

- Pekistirme (_TEKRAR_): "bol bol", "durmadan", "cok"
- Bilesik Kelimeler (KELIME1_KELIME2): tek kavram
- Uzamsal Referanslar: baglamdan zamir cikar

## 6. CIKTI FORMATI

3 ALTERNATIF CEVIRI SUN:

## ALTERNATIF 1
Ceviri: [en dogal ceviri]
Guven: [1-10]/10
Aciklama: [neden bu tercih]

## ALTERNATIF 2
Ceviri: [alternatif yorum]
Guven: [1-10]/10
Aciklama: [farklilik nedeni]

## ALTERNATIF 3
Ceviri: [baska bir olaslik]
Guven: [1-10]/10
Aciklama: [kullanim baglami]

## 7. ONEMLI UYARILAR

- Halusinasyon YAPMA: Transkripiyonda olmayan anlam ekleme
- Belirsizligi BELIRT: Emin degilsen aciklamada yaz
- RAG baglamini KULLAN: Sozluk ve hafiza bilgilerini dikkate al
```

---

## User Prompt

```
# TID TRANSKRIPSIYON CEVIRISI

## DILBILGISI IPUCLARI
- Zaman: Belirsiz - baglamdan cikar veya alternatiflerde varyasyon sun
- Muhtemel konu (topic): IKI
- Fiiller: EVLENMEK, GITMEK

## ORNEKLER (FEW-SHOT)

### Ornek 1
Transkripsiyon: ARABA BEN ALMAK DUN
Ceviri: Dun araba aldim.

### Ornek 2
Transkripsiyon: PAZAR GITMEK BARDAK COK UCUZ BEN ALMAK GELMEK
Ceviri: Ucuz oldugu icin pazardan bardak aldim.

### Ornek 3
Transkripsiyon: DENIZ MANZARA BEN OTURMAK KONUSMAK GECMEK GECMEK BEN BAKMAK APTAL GIBI IZLEMEK
Ceviri: Denizin karsisina oturdum ve aptal aptal gelen gecene baktim.

## BENZER CEVIRI ORNEKLERI (Hafiza)
- PAZAR GITMEK BARDAK COK UCUZ BEN ALMAK GELMEK -> Ucuz oldugu icin pazardan bardak aldim. (benzerlik: 0.81)
- DENIZ MANZARA BEN OTURMAK KONUSMAK GECMEK GECMEK BEN BAKMAK APTAL GIBI IZLEMEK -> Denizin karsisina oturdum ve aptal aptal gelen gecene baktim. (benzerlik: 0.80)
- ALISVERIS BEN GITMEK INCELEMEK GIYIM (DUKKAN) BEN GITMEK GITMEK BULMAK ZOR RENK BASKA BASKA BEN BIRAKMAK -> Alisverise ciktigimda biraz giyim dukkanlarinda dolastim, fakat renge karar veremediyim icin almaktan vazgectim. (benzerlik: 0.80)

## KELIME BILGILERI (Sozluk)

### ABLA [TAM ESLEME]
- Tur: Bilinmiyor
- Aciklama: Abla Elder sister, sister 1) Erkeklerin kiz veya kadinlara seslenirken soyledikleri soz
- TID Ornegi: SOKAK GEZMEK KIZ BIR CANTA DUSMEK ABLA BAGIRMAK GOSTERMEK KIZ ALMAK SONRA (BEN) TESEKKUR ETMEK -> Yolda yururken kizin birinin cantasini dusurdugunu gordum ve arkasindan "abla!" diye seslenerek cantayi gosterdim.

### VAR [TAM ESLEME]
- Tur: Bilinmiyor
- Aciklama: Var To be, To have, There is, In existence 1) Mevcut, evrende veya dusuncede yer alan, yok karsiti
- TID Ornegi: BEN ARABA IKI VAR -> Iki arabam var.

### EVLENMEK [TAM ESLEME]
- Tur: Eylem
- Aciklama: Evlenmek Marriage, To wed, Match up, Hitch up 1) Erkekle kadin, aile kurmak icin yasaya uygun olarak birlesmek
- TID Ornegi: UNIVERSITE ARKADAS HEPSI EVLENMEK EVLENMEK BITMEK BEN YAKIN EVLENMEK DUSUNMEK -> Universite arkadaslarimin hepsi evlendi artik ben de evlenmeyi dusunuyorum.

## SOZLUKTE BULUNMAYAN: IKI, GITMEK
(Bu kelimeler icin genel Turkce bilgini kullan)

---

## CEVIRILECEK TRANSKRIPSIYON

```

IKI ABLA VAR EVLENMEK GITMEK

```

## CIKTI FORMATI

3 alternatif ceviri sun. Her alternatif icin asagidaki formati BIREBIR kullan:

## ALTERNATIF 1
Ceviri: [en dogal ve muhtemel ceviri]
Guven: [1-10]/10
Aciklama: [neden bu yorumu sectin - kisa]

## ALTERNATIF 2
Ceviri: [farkli bir yorum veya zaman]
Guven: [1-10]/10
Aciklama: [bu alternatifin farki - kisa]

## ALTERNATIF 3
Ceviri: [baska bir olaslik]
Guven: [1-10]/10
Aciklama: [hangi baglamda kullanilir - kisa]
```

---

## Notlar

1. **RAG Destegi:** Sozlukten ABLA, VAR, EVLENMEK kelimeleri icin anlam ve ornek cumleler getirildi.
2. **Ornek Cumleler:** "BEN ARABA IKI VAR -> Iki arabam var" ornegi, "IKI ABLA VAR" icin yapisal bir referans sagliyor.
3. **GITMEK Yorumu:** Bu baglamda "GITMEK" muhtemelen "evlenmeye gitmek" (dugun icin gitmek) anlaminda kullaniliyor.
4. **Zaman Belirsizligi:** Acik bir zaman zarfi olmadigindan, LLM baglamdan cikarim yapmali veya alternatiflerde varyasyon sunmali.



# Gemini 3 Pro Thinking ile denenince gelen çıktı:


## ALTERNATIF 1

Ceviri: İki ablam var, evlenip gittiler.

Guven: 9/10

Aciklama: TID'de "EVLENMEK GITMEK" dizilimi, Türkçedeki "evlenip gitmek" (evlenerek baba evinden ayrılmak) deyimiyle tam örtüşür. "IKI ABLA VAR" yapısı sahiplik (iki ablam var) bildirir. En doğal ve kültürel bağlama en uygun çeviridir.

## ALTERNATIF 2

Ceviri: İki ablam da evlendi.

Guven: 8/10

Aciklama: Burada "GITMEK" fiili eylemin tamamlandığını veya kişinin artık hanede olmadığını vurgulayan bir yardımcı unsur (aspect marker) olarak yorumlanmıştır. "VAR" kelimesi cümlenin başında özneyi (iki ablam) tanımlamak için kullanılmıştır.

## ALTERNATIF 3

Ceviri: İki ablam var, evlenmeye gittiler.

Guven: 7/10

Aciklama: Eğer bağlam, ablaların şu an fiziksel olarak bir nikah dairesine veya düğüne gitmek üzere yola çıktığını anlatıyorsa bu çeviri kullanılır. "GITMEK" fiili gerçek anlamıyla (hareket bildiren) kullanılmıştır.





# Gerçek Çeviri: İki ablam da evlendi.

## LLM Yorumu:


Bu sonuç, sağladığım **Alternatif 2** ile birebir örtüşüyor. Bu çevirinin neden "en doğru" kabul edildiğine dair dilbilimsel ve yapısal analizim şöyledir:

### 1. Dilbilimsel Analiz (TİD Grameri)

Bu cümleyi doğru çevirmek için kelimelerin "sözlük anlamlarına" değil, TİD içindeki **gramatikal işlevlerine** odaklanmak gerekir.

* **`IKI ABLA VAR` Yapısı:**
  * TİD'de çoğul eki (-ler/-lar) yoktur. Bunun yerine sayı (İKİ) ve varlık bildiren (VAR) kelimeler kullanılır.
  * Burada `VAR`, "sahip olmak" anlamından ziyade, konuyu (Topic) belirginleştirme işlevi görüyor. Yani "İki tane ablam var..." diye yeni bir cümle başlatmak yerine, **"Mevcut olan o iki abla"** vurgusuyla cümleyi özneleştiriyor. Türkçeye çevirirken bu yapı "İki ablam da..." şekline dönüşüyor.
* **`GITMEK` Fiilinin İşlevi (Aspect Marker):**
  * Bu cümlenin kilit noktası budur. `GITMEK` burada gerçek anlamda bir yerden bir yere yürümek/gitmek değildir.
  * TİD'de `GITMEK` fiili, bazen eylemin  **tamamlandığını** , **sonuçlandığını** veya **geçmişte kaldığını** (Perfective Aspect) vurgulamak için yardımcı fiil gibi kullanılır.
  * `EVLENMEK` eyleminin üzerine gelen `GITMEK`, eylemin bittiğini ve bir statü değişikliği olduğunu gösterir.

### 2. Kültürel Bağlam (Evlenip Gitmek)

Türkçede "evlenip gitmek" diye bir deyim vardır (baba evinden ayrılmak). TİD'deki `EVLENMEK GITMEK` dizilimi, Türkçedeki bu deyimle yapısal olarak tesadüfen örtüşse de, işaret dilinde temel amaç **eylemin kesinliğini** bildirmektir.

### Özet Tablosu

| **TİD Gloss** | **Sözlük Anlamı** | **Bu Cümledeki İşlevi**       | **Türkçe Karşılığı**       |
| -------------------- | -------------------------- | -------------------------------------- | --------------------------------------- |
| **IKI ABLA**   | İki abla                  | Özne grubu                            | İki ablam                              |
| **VAR**        | Mevcut                     | Varlığı kesinleştirme/vurgu        | ...da / ...de (İki ablam**da** ) |
| **EVLENMEK**   | Evlenmek                   | Ana fiil                               | Evlendi                                 |
| **GITMEK**     | Gitmek                     | Eylemin tamamlandığını bildiren ek | (-di/-du geçmiş zaman eki işlevi)    |

**Sonuç:** "İki ablam da evlendi" çevirisi, TİD'in "konu-yorum" yapısını ve "tamamlanmış eylem" belirtecini en temiz ve duru Türkçeye aktaran halidir.

---

**Sizin için ne yapabilirim?** Başka bir TİD videosu veya transkripsiyonu üzerinde benzer bir analiz yapmamı ister misiniz?
