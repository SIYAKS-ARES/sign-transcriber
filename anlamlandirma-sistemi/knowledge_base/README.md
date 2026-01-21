# Knowledge Base (KB) Rehberi

Bu klasör, RAG modülü için düz metin/Markdown bilgi parçalarını içerir. TİD dilbilgisi, örnek cümleler, domain notları gibi küçük, odaklı paragraflar tutmak için kullanılır.

---

## Nasıl Kullanılır?

- KB parçalarını `.md` veya `.txt` olarak ekleyin.  
- Her dosya **tek konuya** odaklı, **kısa ve öz** olsun (örn. “Olumsuzluk”, “Zaman belirteçleri”, “Mekân bağlamı”).  
- Çalışma anında KB yolu:  
  - Varsayılan: `anlamlandirma-sistemi/knowledge_base/`  
  - Override: `RAG_KB_DIR` ortam değişkeni ile farklı bir klasör gösterebilirsiniz.

Örnek (CLI veya kod):
```bash
export RAG_KB_DIR=/path/to/custom_kb
```

---

## Örnek Dosya

- `sample_kb.md`: Olumsuzluk, bağlamlı çok anlamlılık ve örnek gloss → çeviri parçaları içerir.

---

## En İyi Uygulamalar

1) **Tek konu / tek dosya**: Arama ve retrieval daha isabetli olur.  
2) **Kısa paragraflar**: LLM prompt’una eklenecek bağlam kısa tutulmalı.  
3) **Güncelleme notu**: Dosya başına tarih ve kısa açıklama ekleyin:
   ```text
   # Olumsuzluk (DEĞİL / YOK)
   Güncelleme: 2026-01-20
   ```
4) **Dil**: TİD gloss’ları ve Türkçe açıklamaları birlikte tutun (gerekirse İngilizce açıklama ekleyin).

---

## RAG İçinde Kullanım

`anlamlandirma-sistemi/preprocessor.py` → RAG hazırsa KB içeriği, sözlük ve hafıza ile birleştirilerek prompt’a dahil edilir. KB’yi güncelledikten sonra uygulamayı yeniden başlatmanız önerilir.
