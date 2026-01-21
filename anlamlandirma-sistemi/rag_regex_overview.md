# Anlamlandırma Sistemi: RAG ve Regex Bileşen Özeti

Bu doküman, `anlamlandirma-sistemi` içinde yer alan RAG (Retrieval-Augmented Generation) ve regex tabanlı işleme adımlarının nasıl çalıştığını tek noktada toplar. Amaç, yerel modelden gelen ham transkripsiyonların anlamlı bir LLM girdisine dönüştürülmesini ve LLM çıktılarının güvenilir biçimde ayrıştırılmasını açıklamaktır.

## Genel Akış
- `local_model_handler.py` ham işaret dizisini işler ve eşik üstü tahminleri `transkripsiyon` formatında üretir.
- `preprocessor.py` regex kurallarıyla transkripsiyonu temizler, anlamsal ipuçları ekler ve RAG bağlamı ile nihai promptu oluşturur.
- `llm_services.py` üretilen promptu seçilen LLM sağlayıcısına gönderir ve gelen formatlı yanıtı regex ile ayrıştırır.

## RAG Bileşenleri
- **Getirici**: `rag_simple.py` içinde TF-IDF tabanlı hafif bir getirici bulunur. Çalışma anında `knowledge_base/` klasörünü tarar, `.md` ve `.txt` belgelerden bir TF-IDF matrisi oluşturur.
- **Bilgi Tabanı**: Varsayılan klasör `knowledge_base/`. `RAG_KB_DIR` ortam değişkeni ile farklı bir konuma işaret edilebilir. Klasöre konu odaklı kısa metinler eklenerek bağlam zenginleştirilir.
- **Prompt Enjeksiyonu**: `create_final_prompt` fonksiyonu, hem kural tabanlı sözlük (`RAG_KNOWLEDGE_BASE`) hem de getiriciden dönen en alakalı `top_k` pasajları "Dinamik RAG Bağlamı" bölümüne enjekte eder.
- **Bağımlılık Yönetimi**: `scikit-learn` yüklenmemişse veya bilgi tabanı boşsa RAG otomatik olarak devre dışı kalır; sistem yalnızca kural tabanlı bağlamla devam eder.

```55:83:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/rag_simple.py
        _VECTORIZER = TfidfVectorizer(min_df=1, max_df=0.95)
        _MATRIX = _VECTORIZER.fit_transform(docs)
    # ...
        snippet = text.strip().replace("\n", " ")[:500]
        results.append((snippet, _DOC_PATHS[idx], float(score)))
```

## Regex Tabanlı Ön İşleme
- **Kelime Birleştirme**: `^` karakterini `_` ile değiştirerek LSTM çıktılarındaki token birleşmelerini normalize eder.
- **Tekrar Desenleri**: Belirli tekrar kombinasyonlarını tanıyıp semantik etiket ekler (ör. `ADAM ADAM → ADAM(çoğul)`).
- **Negasyon Etiketi**: `DEĞİL` ve `YOK` kalıplarını `(negasyon:...)` etiketi ile zenginleştirir.
- **Boşluk Temizliği**: Fazla boşlukları normalize eder.

```18:30:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/preprocessor.py
    processed = transcription.replace('^', '_')
    processed = re.sub(r"\b(ADAM)\s+\1\b", r"\1(çoğul)", processed)
    processed = re.sub(r"\b(GİTMEK)\s+\1\b", r"\1(süreç/ısrar)", processed)
    processed = re.sub(r"(\w+)\s+(DEĞİL|YOK)\b", r"\1 (negasyon:\2)", processed)
    processed = re.sub(r"\s+", " ", processed).strip()
```

## Regex Tabanlı Çıktı Ayrıştırma
- LLM sağlayıcılarından dönen metin `Çeviri`, `Güven` ve `Açıklama` alanları için regex kalıplarıyla ayrıştırılır.
- Veri eksik olduğunda hata mesajı üretilir ve ham yanıt döndürülür; bu sayede istemci tarafında güvenilirlik kontrolü yapılabilir.

```8:30:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/llm_services.py
        translation_match = re.search(r"Çeviri:\s*(.*)", llm_response)
        confidence_match = re.search(r"Güven:\s*(\d+)\s*/\s*10", llm_response)
        explanation_match = re.search(r"Açıklama:\s*(.*)", llm_response, re.DOTALL)
        # ...
        return {
            "translation": translation,
            "confidence": confidence,
            "explanation": explanation,
            "error": None,
        }
```

## Uygulama ve Test Önerileri
- `knowledge_base/` içine alan uzmanlığına yönelik yeni `.md` dosyaları ekleyin; uygulamayı yeniden başlattığınızda yeni içerikler otomatik olarak indekslenir.
- Regex kurallarını genişletirken `anlamlandirma-sistemi-staj.md` içindeki örnekleri referans alın ve yeni kalıplar için birim testleri yazmayı planlayın.
- LLM yanıt formatını değiştirecek güncellemelerde `llm_services.parse_structured_output` fonksiyonunu güncellemeyi unutmayın.

## Yol Haritası
- TF-IDF getiriciyi, cümle gömme tabanlı (ör. `sentence-transformers`) bir getiriciye yükseltmek.
- Regex kurallarını konfigürasyon tabanlı hale getirip kolayca genişletilebilir bir kural motoru geliştirmek.
- LLM çıktısını şema tabanlı doğrulamak için `pydantic` benzeri tip doğrulama katmanı eklemek.


