# Basit RAG Entegrasyonu (TF-IDF)

Bu doküman, `anlamlandirma-sistemi` içinde eklenen temel RAG (Retrieval-Augmented Generation) yapısını açıklar.

## Neler Eklendi?
- `rag_simple.py`: TF-IDF tabanlı hafif getirici. `.md` / `.txt` belgelerden sözlük oluşturur ve kosinüs benzerliğiyle en alakalı parçaları döndürür.
- `knowledge_base/`: Varsayılan bilgi klasörü. İçinde örnek olarak `sample_kb.md` ve bir `README.md` bulunur.
- `preprocessor.py`: `create_final_prompt` artık iki kaynaklı bağlam enjekte eder:
  - Kural tabanlı mini sözlük (`RAG_KNOWLEDGE_BASE`).
  - TF-IDF getirici ile bulunan en alakalı kısa alıntılar.

## Nasıl Çalışır?
1. Uygulama çalıştığında ilk sorguda `rag_simple.py` bilgi klasörünü tarar ve TF-IDF indeksini oluşturur.
2. `create_final_prompt(processed_transcription)` çağrısı, işlenmiş transkripsiyondan kısa bir sorgu türetir ve getiriciye gönderir.
3. Dönen ilk birkaç alıntı, prompt içindeki "Dinamik RAG Bağlamı" bölümüne eklenir.

## Kurulum
- Ek bağımlılık: scikit-learn

```bash
conda activate anlamlandirma
python -m pip install -U scikit-learn
```

> Not: scikit-learn yoksa sistem otomatik olarak RAG'ı kapatır ve sadece kural tabanlı bağlamı kullanır.

## Bilgi Tabanı Konumu
- Varsayılan: `anlamlandirma-sistemi/knowledge_base/`
- Ortam değişkeni ile değiştirilebilir:

```bash
export RAG_KB_DIR=/absolute/path/to/your/kb
```

Klasöre `.md` veya `.txt` dosyaları ekleyin. Her dosya kısa ve tek konu odaklı olmalıdır.

## Entegrasyon Noktası
- Kod referansı:

```34:88:/Users/siyaksares/Developer/GitHub/klassifier-sign-language/msh-sign-language-tryouts/anlamlandirma-sistemi/preprocessor.py
def create_final_prompt(processed_transcription: str) -> str:
    """Dinamik (kural tabanlı + basit TF-IDF RAG) bağlamla nihai prompt oluşturur."""
    # ... kelime tabanlı bağlam ...
    if get_retrieved_context is not None:
        for snippet, path, score in get_retrieved_context(query, top_k=3):
            retrieved_parts.append(f"[KB:{os.path.basename(path)} | skor={score:.3f}] {snippet}")
```

## Hızlı Test
- `knowledge_base/sample_kb.md` içeriğini düzenleyin veya yeni bir `.md` ekleyin.
- Uygulamayı çalıştırın ve çeviri akışını tetikleyin; prompt içine "Dinamik RAG Bağlamı" satırlarının eklendiğini gözlemleyin.

## Sınırlar ve Sonraki Adımlar
- TF-IDF küçük/orta veri setleri için yeterlidir; büyük veri için FAISS/Chroma gibi vektör veri tabanlarına geçilebilir.
- Cümle gömme (ör. `sentence-transformers`) eklenerek anlamsal eşleme güçlendirilebilir.
- Belgeler parçalara (chunk) bölünüp başlık/etiketlerle zenginleştirilebilir.


