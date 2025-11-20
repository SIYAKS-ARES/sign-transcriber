import re
import os
from typing import List

try:
    # RAG entegrasyonu opsiyonel: sklearn yoksa sessizce devre dışı kalır
    from rag_simple import get_retrieved_context
except Exception:
    get_retrieved_context = None  # type: ignore


def preprocess_text_for_llm(transcription: str) -> str:
    """Transkripsiyonu LLM için regex ve basit anlamsal kurallarla işler."""
    if not transcription:
        return ""

    # 1) Birleşik kelimeleri işle: ARABA^SÜRMEK -> ARABA_SÜRMEK
    processed = transcription.replace('^', '_')

    # 2) Anlamsal tekrarları işle (örnek tabanlı)
    # ADAM ADAM -> ADAM(çoğul)
    processed = re.sub(r"\b(ADAM)\s+\1\b", r"\1(çoğul)", processed)
    # GİTMEK GİTMEK -> GİTMEK(süreç/ısrar)
    processed = re.sub(r"\b(GİTMEK)\s+\1\b", r"\1(süreç/ısrar)", processed)

    # 3) Negasyon yapılarını işaretle: "X DEĞİL" / "X YOK"
    processed = re.sub(r"(\w+)\s+(DEĞİL|YOK)\b", r"\1 (negasyon:\2)", processed)

    # 4) Fazla boşlukları temizle
    processed = re.sub(r"\s+", " ", processed).strip()

    return processed


# Basit dinamik RAG bilgi tabanı
RAG_KNOWLEDGE_BASE = {
    "AY": "Ek Bilgi: 'AY' işareti hem 'gök cismi' hem de 'takvim ayı' anlamına gelebilir. Bağlama göre yorumla.",
    "YÜZMEK": "Ek Bilgi: 'YÜZMEK' işareti hem 'suda yüzmek' eylemi hem de 'deri yüzmek' anlamına gelebilir.",
}


def create_final_prompt(processed_transcription: str) -> str:
    """Dinamik (kural tabanlı + basit TF-IDF RAG) bağlamla nihai prompt oluşturur."""
    # 1) Dinamik RAG Bağlamı (kural tabanlı sözlük)
    dynamic_context_parts: List[str] = []
    words = processed_transcription.replace('(', ' ').replace(')', ' ').split()
    for word in words:
        if word in RAG_KNOWLEDGE_BASE:
            dynamic_context_parts.append(RAG_KNOWLEDGE_BASE[word])

    # 1.b) Basit vektör tabanlı RAG (opsiyonel)
    retrieved_parts: List[str] = []
    if get_retrieved_context is not None:
        try:
            query = " ".join(words[:12])  # kısa sorgu
            for snippet, path, score in get_retrieved_context(query, top_k=3):
                retrieved_parts.append(f"[KB:{os.path.basename(path)} | skor={score:.3f}] {snippet}")
        except Exception:
            # retriever yoksa veya hata alırsa bağlamı boş bırak
            pass

    dynamic_context = "\n".join(dynamic_context_parts + retrieved_parts)

    # 2) Gelişmiş Prompt Şablonu
    prompt_template = f"""
# GÖREV: UZMAN TİD TERCÜMANI

## KİMLİK (PERSONA)
Sen, 20 yıllık deneyime sahip, Türk İşaret Dili (TİD) ve Türkçe dilbilimine derinlemesine hakim bir simultane tercümansın.

## SÜREÇ (CHAIN-OF-THOUGHT)
Çeviriyi yaparken şu adımları izle:
1.  **Analiz Et:** Aşağıdaki "Ham Transkripsiyonu" ve "Dinamik RAG Bağlamı" bölümünü dikkatlice oku.
2.  **Yapıyı Çözümle:** TİD'in genellikle Özne-Nesne-Yüklem olan yapısını ve eksik ekleri tespit et.
3.  **Yeniden Kur:** Cümleyi, Türkçe'nin kurallı ve akıcı yapısına dönüştür. Eksik zaman ve kişi eklerini bağlamdan çıkararak ekle.
4.  **Sonuçlandır:** Çıktını aşağıdaki `İSTENEN ÇIKTI FORMATI`na birebir uygun şekilde, 3 bölüm halinde oluştur.

## ÖRNEKLER (FEW-SHOT LEARNING)
-   **Örnek 1 Transkripsiyon:** BEN OKUL GİTMEK(süreç/ısrar) AMA KAPI KİLİTLİ
-   **Örnek 1 Çıktı:**
    Çeviri: Okula sürekli gittim ama kapı kilitliydi.
    Güven: 10/10
    Açıklama: Tekrar eden 'gitmek' eylemi, sürecin devamlılığını belirttiği için 'sürekli gittim' olarak çevrilmiştir.
-   **Örnek 2 Transkripsiyon:** SEN YEMEK BEĞENMEK (negasyon:DEĞİL)
-   **Örnek 2 Çıktı:**
    Çeviri: Yemeği beğenmedin.
    Güven: 9/10
    Açıklama: Cümlenin sonunda soru mimiği bilgisi olmadığı için ifade soru olarak değil, olumsuz bir tespit olarak çevrilmiştir.

## İSTENEN ÇIKTI FORMATI
(Aşağıdaki başlıkları koruyarak ÇIKTI ÜRET)
Çeviri: [Sadece çevrilmiş akıcı Türkçe cümleyi buraya yaz]
Güven: [Çevirinin doğruluğuna dair 1-10 arası bir puan ver]
Açıklama: [Çeviriyi yaparken hangi varsayımlarda bulunduğunu veya karşılaştığın bir belirsizliği kısaca açıkla]

---
## ÇEVİRİ GÖREVİ

### DİNAMİK RAG BAĞLAMI
* Genel Kural: Transkripsiyon, devrik cümle yapısına sahip olabilir ve zaman ekleri içermeyebilir.
* {dynamic_context}

### HAM TRANSKRİPSİYON (REGEX İLE İŞLENMİŞ)
`{processed_transcription}`
"""
    return prompt_template.strip()


