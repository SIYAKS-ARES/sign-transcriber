"""
Prompt Templates for TID Translation
=====================================
User prompt templates with multi-alternative output format.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass


# =============================================================================
# OUTPUT FORMAT SPECIFICATION
# =============================================================================

OUTPUT_FORMAT_TEMPLATE = """
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
"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """
# TID TRANSKRIPSIYON CEVIRISI

{linguistic_context}

{few_shot_examples}

{rag_context}

---

## CEVIRILECEK TRANSKRIPSIYON

```
{transcription}
```

{output_format}
"""


# =============================================================================
# LINGUISTIC CONTEXT TEMPLATES
# =============================================================================

def build_linguistic_context(
    detected_tense: Optional[str] = None,
    tense_source: Optional[str] = None,
    is_question: bool = False,
    is_negative: bool = False,
    repetitions: Optional[Dict[str, int]] = None,
    linguistic_hints: Optional[Dict] = None,
) -> str:
    """
    Build linguistic context section for the prompt.
    
    Args:
        detected_tense: Detected tense ("past", "present", "future", None)
        tense_source: How tense was detected ("explicit", "inferred", None)
        is_question: Whether question markers were detected
        is_negative: Whether negation markers were detected
        repetitions: Dict of repeated words and their counts
        linguistic_hints: Additional linguistic hints from preprocessor
        
    Returns:
        Formatted linguistic context string
    """
    parts = ["## DILBILGISI IPUCLARI"]
    
    # Tense information
    if detected_tense:
        tense_map = {
            "past": "Gecmis zaman (-di/-mis)",
            "present": "Simdiki/genis zaman (-iyor/-ir)",
            "future": "Gelecek zaman (-ecek/-acak)",
        }
        tense_str = tense_map.get(detected_tense, detected_tense)
        source_str = "(acik zarf)" if tense_source == "explicit" else "(cikarimsal)"
        parts.append(f"- Tespit edilen zaman: {tense_str} {source_str}")
    else:
        parts.append("- Zaman: Belirsiz - baglamdan cikar veya alternatiflerde varyasyon sun")
    
    # Question indicator
    if is_question:
        parts.append("- Cumle tipi: SORU cumlesi (soru kelimesi tespit edildi)")
    
    # Negation indicator
    if is_negative:
        parts.append("- Cumle tipi: OLUMSUZ cumle (negasyon belirteci tespit edildi)")
    
    # Repetition information
    if repetitions:
        rep_items = [f"{word} ({count}x)" for word, count in repetitions.items()]
        parts.append(f"- Pekistirme/tekrar: {', '.join(rep_items)}")
        parts.append("  -> 'bol bol', 'cok', 'surekli' gibi yogunluk ifadeleri kullan")
    
    # Additional hints
    if linguistic_hints:
        if linguistic_hints.get("likely_topic"):
            parts.append(f"- Muhtemel konu (topic): {linguistic_hints['likely_topic']}")
        if linguistic_hints.get("verbs"):
            parts.append(f"- Fiiller: {', '.join(linguistic_hints['verbs'])}")
        if linguistic_hints.get("pronouns"):
            parts.append(f"- Zamirler: {', '.join(linguistic_hints['pronouns'])}")
    
    # If no special features detected
    if len(parts) == 1:
        parts.append("- Ozel dilbilgisi ozelligi tespit edilmedi")
        parts.append("- Varsayilan: simdiki/genis zaman, olumlu cumle")
    
    return "\n".join(parts)


# =============================================================================
# MAIN PROMPT BUILDER
# =============================================================================

def build_user_prompt(
    transcription: str,
    few_shot_examples: str = "",
    rag_context: str = "",
    detected_tense: Optional[str] = None,
    tense_source: Optional[str] = None,
    is_question: bool = False,
    is_negative: bool = False,
    repetitions: Optional[Dict[str, int]] = None,
    linguistic_hints: Optional[Dict] = None,
    include_output_format: bool = True,
) -> str:
    """
    Build the complete user prompt for LLM.
    
    Args:
        transcription: The preprocessed TID transcription
        few_shot_examples: Formatted few-shot examples string
        rag_context: RAG context string from retriever
        detected_tense: Detected tense from preprocessor
        tense_source: Tense detection source
        is_question: Question indicator
        is_negative: Negation indicator
        repetitions: Word repetitions
        linguistic_hints: Additional hints
        include_output_format: Whether to include output format specification
        
    Returns:
        Complete user prompt string
    """
    # Build linguistic context
    linguistic_context = build_linguistic_context(
        detected_tense=detected_tense,
        tense_source=tense_source,
        is_question=is_question,
        is_negative=is_negative,
        repetitions=repetitions,
        linguistic_hints=linguistic_hints,
    )
    
    # Build output format section
    output_format = OUTPUT_FORMAT_TEMPLATE if include_output_format else ""
    
    # Build complete prompt
    prompt = USER_PROMPT_TEMPLATE.format(
        linguistic_context=linguistic_context,
        few_shot_examples=few_shot_examples if few_shot_examples else "## ORNEKLER\n(Referans ornek bulunamadi)",
        rag_context=rag_context if rag_context else "## RAG BAGLAMI\n(Sozluk bilgisi bulunamadi)",
        transcription=transcription,
        output_format=output_format,
    )
    
    return prompt.strip()


# =============================================================================
# SIMPLE PROMPT (FOR BASELINE)
# =============================================================================

SIMPLE_PROMPT_TEMPLATE = """
Turk Isaret Dili (TID) transkripiyonunu Turkce'ye cevir.

Transkripsiyon: {transcription}

Kurallari:
1. TID'de fiiller mastar halinde (GITMEK, YAPMAK). Kisi ve zaman eklerini ekle.
2. TID Topic-Comment yapisini SOV (Ozne-Nesne-Yuklem) yapisina cevir.
3. Eksik ekleri (iyelik, hal ekleri) tamamla.

Ceviri:
"""


def build_simple_prompt(transcription: str) -> str:
    """Build a simple prompt without RAG augmentation (for baseline)."""
    return SIMPLE_PROMPT_TEMPLATE.format(transcription=transcription).strip()


# =============================================================================
# DATACLASS FOR PROMPT RESULT
# =============================================================================

@dataclass
class PromptComponents:
    """Components of a built prompt."""
    system_instruction: str
    user_prompt: str
    transcription: str
    
    # Metadata
    detected_tense: Optional[str] = None
    is_question: bool = False
    is_negative: bool = False
    has_repetitions: bool = False
    
    def get_full_prompt(self) -> str:
        """Get the complete prompt (system + user) for non-chat APIs."""
        return f"{self.system_instruction}\n\n---\n\n{self.user_prompt}"
