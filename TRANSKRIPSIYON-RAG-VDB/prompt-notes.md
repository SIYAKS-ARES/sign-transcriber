# Türk İşaret Dili (TİD) Transkripsiyonu Doğal Dil İşleme Rehberi

Bu doküman, el işaretlerinden elde edilen kelime dizilerini (glos), Gemini API kullanarak anlamlı, kurallı ve doğal Türkiye Türkçesine dönüştürme yöntemini kapsar.

## 1. Temel Sorun ve Çözüm Yaklaşımı

Sorun: İşaret dili transkriptleri ek içermez, mastar halindedir ve devrik bir yapıya sahiptir (Örn: "DÜN OKUL GİTMEK"). Doğrudan çeviri yapıldığında anlamsız sonuçlar doğar.

Çözüm: Gemini API'yi "Sistem Talimatı (System Instruction)" ve "Örneklemeli Öğrenme (Few-Shot Prompting)" yöntemleriyle bir "Linguistik Editör" olarak yapılandırmak.

---

## 2. Prompt Mühendisliği ve Kural Çerçevesi

Modelin kararlı çalışması için tanımlanan kurallar bütünü şunlardır:

* **Zaman Çekimi:** Cümledeki zaman zarfları (bugün, dün, sonra) tespit edilip tüm fiiller bu zamana göre çekimlenir.
* **Ek Yönetimi:** TİD'de bulunmayan iyelik (-im), yönelme (-e), bulunma (-de) ve vasıta (-ile) ekleri anlam akışına göre otomatik eklenir.
* **"BİTMEK" Belirteci:** TİD'de "BİTMEK" veya "TAMAM", genellikle eylemin geçmişte tamamlandığını belirtir. Model bunu bir kelime olarak değil, geçmiş zaman eki (-di) olarak yorumlar.
* **Pekiştirme:** Tekrar eden kelimeler (GEZMEK GEZMEK), Türkçedeki zarf fiillere veya pekiştirmeli ifadelere (bol bol gezmek, gezip durmak) dönüştürülür.
* **Sözdizimi (Syntax):** Girdi ne kadar devrik olursa olsun, çıktı her zaman "Özne + Nesne + Yüklem" sırasına uygun kurallı cümle olmalıdır.

---

## 3. Uygulama Kodu (Python)

Aşağıdaki yapı, `gemini-1.5-flash` modelini en verimli şekilde kullanmak üzere tasarlanmış, projeye hazır bir sınıftır.

**Python**

```
import google.generativeai as genai

class TIDTranslator:
    def __init__(self, api_key):
        # API Yapılandırması
        genai.configure(api_key=api_key)
      
        # SİSTEM TALİMATI: Modelin 'beyni' ve uyması gereken katı kurallar
        self.system_instruction = """
        Sen uzman bir Türk İşaret Dili (TİD) tercümanısın. Görevin, büyük harfli glosları 
        doğal Türkçeye çevirmektir.
      
        KURALLAR:
        1. Mastarları (GİTMEK, YAPMAK) zamana ve kişiye göre çekimle.
        2. Hal eklerini (-e, -de, -den) ve iyelik eklerini (-im, -in) cümlenin gelişine göre ekle.
        3. 'BİTMEK' kelimesini geçmiş zaman eki (-dı, -di) veya 'tamamladım' anlamında işle.
        4. Tekrarları (KOŞMAK KOŞMAK) süreklilik veya yoğunluk olarak çevir.
        5. Sadece çeviriyi döndür, ek açıklama yapma.
        """
      
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.system_instruction
        )
      
        # FEW-SHOT ÖRNEKLERİ: Modelin stili anlaması için hafıza kaydı
        self.history = [
            {"role": "user", "parts": "DÜN ARKADAŞ BULUŞMAK KAHVE İÇMEK"},
            {"role": "model", "parts": "Dün arkadaşımla buluşup kahve içtik."},
            {"role": "user", "parts": "EV YEMEK YOK ÇARŞI GİTMEK LAZIM"},
            {"role": "model", "parts": "Evde yemek yok, çarşıya gitmemiz gerekiyor."},
            {"role": "user", "parts": "BEN OKUL GİTMEK SINAV OLMAK BİTMEK"},
            {"role": "model", "parts": "Okula gidip sınavımı tamamladım."}
        ]

    def translate(self, glos_input):
        """Transkripti alır ve doğal Türkçeye çevirir."""
        try:
            chat = self.model.start_chat(history=self.history)
            prompt = f"Çevir: {glos_input}"
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Hata: {str(e)}"

# --- KULLANIM ---
# translator = TIDTranslator("API_KEY_BURAYA")
# sonuc = translator.translate("BUGÜN ARKADAŞ BEN BULUŞMAK BERABER ALIŞVERİŞ GEZMEK GEZMEK MUTLU OLMAK BİTMEK")
# print(sonuc) 
# Çıktı: Bugün arkadaşımla buluştuk; beraber alışveriş yapıp bol bol gezdik ve çok mutlu olduk.
```

---

## 4. RAG (Retrieval-Augmented Generation) Entegrasyonu

Eğer sisteminizde bir RAG yapısı varsa, Gemini'ye gönderdiğiniz prompt'u şu şekilde zenginleştirmeniz doğruluğu artırır:

* **Deyim Kontrolü:** Eğer transkript içinde özel bir işaret kalıbı varsa, RAG'dan gelen anlamı prompt'a ekleyin.
* **Örnek:**
  > Bağlam (RAG): Bu kullanıcının veri setinde "EL SALLAMAK + GİTMEK" kalıbı "Veda edip ayrılmak" anlamına gelir.
  >
  > Giriş: ARKADAŞ EL SALLAMAK GİTMEK.
  >
  > Sonuç: Arkadaşım veda ederek yanımızdan ayrıldı.
  >

---

## 5. İpuçları ve Optimizasyon

* **Model Seçimi:** Hız ve maliyet için `gemini-1.5-flash` idealdir. Çok daha karmaşık edebi çeviriler için `gemini-1.5-pro` denenebilir.
* **Hata Payı:** Eğer model hala devrik cümle kuruyorsa, `system_instruction` kısmına **"Cümle sonuna mutlaka uygun yüklemi getir"** ibaresini ekleyin.
* **Çoklu Çıktı:** Gerekirse modelden tek bir çeviri yerine 3 farklı alternatif isteyip arasından en yüksek skorluyu seçebilirsiniz.

---

**Doküman Sonu**

Bu yapıyı kurduğunuzda, projenizdeki transkriptler sadece kelime yığını olmaktan çıkıp profesyonel bir çeviri hizmetine dönüşecektir. Başka bir ekleme yapmamı ister misiniz?
