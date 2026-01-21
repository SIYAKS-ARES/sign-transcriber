"""
RAG-Augmented Prompt Builder
============================
Builds LLM prompts augmented with retrieval results.
"""

from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_CONTEXT_TOKENS
from retriever.dual_retriever import DualRetriever, RetrievalResult


class AugmentedPromptBuilder:
    """
    Builds prompts for LLM translation with RAG augmentation.
    """
    
    # Approximate characters per token (for Turkish)
    CHARS_PER_TOKEN = 4
    
    def __init__(self, retriever: Optional[DualRetriever] = None):
        """
        Initialize the prompt builder.
        
        Args:
            retriever: Optional pre-initialized DualRetriever
        """
        self.retriever = retriever or DualRetriever()
    
    def build_prompt(
        self, 
        transcription: str,
        max_context_chars: Optional[int] = None,
    ) -> str:
        """
        Build a complete prompt for LLM translation.
        
        Args:
            transcription: The TID gloss transcription to translate
            max_context_chars: Maximum characters for context (defaults to config)
            
        Returns:
            Complete prompt string for LLM
        """
        # Calculate max chars from tokens if not specified
        if max_context_chars is None:
            max_context_chars = MAX_CONTEXT_TOKENS * self.CHARS_PER_TOKEN
        
        # Get retrieval results
        retrieval_result = self.retriever.retrieve(transcription)
        
        # Build context string (may need truncation)
        context = retrieval_result.to_context_string()
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[Baglamsal bilgi kesildi...]"
        
        # Build the complete prompt
        prompt = self._format_prompt(transcription, context)
        
        return prompt
    
    def _format_prompt(self, transcription: str, context: str) -> str:
        """
        Format the final prompt with all components.
        
        Args:
            transcription: The TID transcription
            context: The RAG context string
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""# GOREV: UZMAN TID TERCUMANI

## KIMLIK (PERSONA)
Sen, 20 yillik deneyime sahip, Turk Isaret Dili (TID) ve Turkce dilbilimine derinlemesine hakim bir simultane tercumansin.

## SUREC (CHAIN-OF-THOUGHT)
Ceviriyi yaparken su adimlari izle:
1. **Analiz Et:** Asagidaki "Ham Transkripsiyon" ve "Dinamik RAG Baglami" bolumunu dikkatlice oku.
2. **Yapiyi Cozumle:** TID'in genellikle Ozne-Nesne-Yuklem olan yapisini ve eksik ekleri tespit et.
3. **Yeniden Kur:** Cumleyi, Turkce'nin kuralli ve akici yapisina donustur. Eksik zaman ve kisi eklerini baglamdan cikararak ekle.
4. **Sonuclandir:** Ciktini asagidaki `ISTENEN CIKTI FORMATI`na birebir uygun sekilde, 3 bolum halinde olustur.

## ORNEKLER (FEW-SHOT LEARNING)
- **Ornek 1 Transkripsiyon:** BEN OKUL GITMEK AMA KAPI KILITLI
- **Ornek 1 Cikti:**
    Ceviri: Okula gittim ama kapi kilitliydi.
    Guven: 9/10
    Aciklama: Ozne 'BEN' oldugu icin birinci tekil sahis olarak cevrildi, zaman eki baglamdan cikarildi.

- **Ornek 2 Transkripsiyon:** SEN YEMEK BEGENMEK DEGIL
- **Ornek 2 Cikti:**
    Ceviri: Yemegi begenmedin.
    Guven: 9/10
    Aciklama: DEGIL negasyon ifade ettigi icin olumsuz cumle olusturuldu.

## ISTENEN CIKTI FORMATI
(Asagidaki basliklari koruyarak CIKTI URET)
Ceviri: [Sadece cevrilmis akici Turkce cumleyi buraya yaz]
Guven: [Cevirinin dogruluguna dair 1-10 arasi bir puan ver]
Aciklama: [Ceviriyi yaparken hangi varsayimlarda bulundugunu veya karsilastigin bir belirsizligi kisaca acikla]

---
## CEVIRI GOREVI

### DINAMIK RAG BAGLAMI
* Genel Kural: Transkripsiyon, devrik cumle yapisina sahip olabilir ve zaman ekleri icermeyebilir.
{context}

### HAM TRANSKRIPSIYON
`{transcription}`
"""
        return prompt.strip()
    
    def build_simple_prompt(self, transcription: str) -> str:
        """
        Build a simple prompt without RAG augmentation (for baseline comparison).
        
        Args:
            transcription: The TID gloss transcription
            
        Returns:
            Simple prompt string for baseline LLM translation
        """
        return f"""Turk Isaret Dili (TID) transkripsiyonunu Turkce'ye cevir.

Transkripsiyon: {transcription}

Ceviri:"""


def create_augmented_prompt(transcription: str) -> str:
    """
    Convenience function to create an augmented prompt.
    
    Args:
        transcription: The TID gloss transcription
        
    Returns:
        Complete augmented prompt
    """
    builder = AugmentedPromptBuilder()
    return builder.build_prompt(transcription)


if __name__ == "__main__":
    # Test the prompt builder
    print("Testing Augmented Prompt Builder...")
    print("=" * 60)
    
    builder = AugmentedPromptBuilder()
    
    test_query = "OKUL GITMEK ISTEMEK"
    print(f"Test query: '{test_query}'")
    print("-" * 60)
    
    prompt = builder.build_prompt(test_query)
    print(prompt)
    
    print("\n" + "=" * 60)
    print("Baseline prompt (no RAG):")
    print("-" * 60)
    print(builder.build_simple_prompt(test_query))
