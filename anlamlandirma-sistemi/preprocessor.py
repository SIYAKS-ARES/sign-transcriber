"""
TID Preprocessing with RAG Integration
======================================
RAG-enhanced preprocessor for Turkish Sign Language transcription.
Provides fallback to simple preprocessing if RAG system fails.
"""

import re
import os
from typing import List, Dict, Optional, Any

# Global state for lazy initialization
_pipeline = None
_fallback_mode = False
_rag_initialized = False


def _try_init_rag():
    """Try to initialize the RAG pipeline."""
    global _pipeline, _fallback_mode, _rag_initialized
    
    if _rag_initialized:
        return
    
    _rag_initialized = True
    
    try:
        from rag.pipeline.translation_pipeline import TranslationPipeline
        _pipeline = TranslationPipeline(provider="gemini")
        print("RAG sistemi basariyla baslatildi.")
    except Exception as e:
        print(f"RAG sistemi baslatilamadi: {e}")
        print("Fallback moduna geciliyor...")
        _fallback_mode = True


def get_pipeline():
    """Get or create the translation pipeline."""
    global _pipeline, _fallback_mode
    _try_init_rag()
    return _pipeline


def is_rag_available() -> bool:
    """Check if RAG system is available."""
    _try_init_rag()
    return not _fallback_mode and _pipeline is not None


# =============================================================================
# SIMPLE FALLBACK PREPROCESSING (used when RAG fails)
# =============================================================================

def _simple_preprocess(transcription: str) -> str:
    """Simple preprocessing without RAG (fallback mode)."""
    if not transcription:
        return ""

    # 1) Handle compound words: ARABA^SURMEK -> ARABA_SURMEK
    processed = transcription.replace('^', '_')

    # 2) Handle repetitions
    processed = re.sub(r"\b(ADAM)\s+\1\b", r"\1(cogul)", processed)
    processed = re.sub(r"\b(GITMEK)\s+\1\b", r"\1(surec/israr)", processed)

    # 3) Mark negation: "X DEGIL" / "X YOK"
    processed = re.sub(r"(\w+)\s+(DEGIL|YOK)\b", r"\1 (negasyon:\2)", processed, flags=re.IGNORECASE)

    # 4) Clean whitespace
    processed = re.sub(r"\s+", " ", processed).strip()

    return processed


def _simple_create_prompt(processed_transcription: str) -> str:
    """Create simple prompt without RAG augmentation (fallback mode)."""
    prompt_template = f"""
# GOREV: UZMAN TID TERCUMANI

## KIMLIK (PERSONA)
Sen, 20 yillik deneyime sahip, Turk Isaret Dili (TID) ve Turkce dilbilimine derinlemesine hakim bir simultane tercumansin.

## SUREC (CHAIN-OF-THOUGHT)
Ceviriyi yaparken su adimlari izle:
1. Asagidaki "Ham Transkripsiyonu" dikkatlice oku.
2. TID'in genellikle Ozne-Nesne-Yuklem olan yapisini ve eksik ekleri tespit et.
3. Cumleyi, Turkce'nin kuralli ve akici yapisina donustur.
4. Ciktini asagidaki formata uygun sekilde olustur.

## ORNEKLER (FEW-SHOT LEARNING)
- Ornek 1 Transkripsiyon: BEN OKUL GITMEK
- Ornek 1 Cikti:
    Ceviri: Okula gidiyorum.
    Guven: 9/10
    Aciklama: Simdiki zaman varsayildi.

## ISTENEN CIKTI FORMATI
Ceviri: [Sadece cevirilmis akici Turkce cumleyi buraya yaz]
Guven: [Cevirinin dogruluguna dair 1-10 arasi bir puan ver]
Aciklama: [Ceviriyi yaparken hangi varsayimlarda bulundugunu kısaca acikla]

---
## CEVIRI GOREVI

### HAM TRANSKRIPSIYON
`{processed_transcription}`
"""
    return prompt_template.strip()


# =============================================================================
# RAG-ENHANCED PREPROCESSING
# =============================================================================

def preprocess_text_for_llm(transcription: str) -> str:
    """
    Preprocess transcription for LLM.
    Uses RAG-enhanced preprocessing if available, fallback otherwise.
    
    Args:
        transcription: Raw TID transcription
        
    Returns:
        Processed transcription string
    """
    if not transcription:
        return ""
    
    if is_rag_available():
        try:
            from rag.preprocessing.tid_preprocessor import TIDPreprocessor
            preprocessor = TIDPreprocessor()
            result = preprocessor.preprocess(transcription)
            return result.processed
        except Exception as e:
            print(f"RAG preprocessing hatasi: {e}")
    
    # Fallback to simple preprocessing
    return _simple_preprocess(transcription)


def create_final_prompt(processed_transcription: str) -> str:
    """
    Create final prompt for LLM translation.
    Uses RAG-augmented prompt if available, fallback otherwise.
    
    Args:
        processed_transcription: Preprocessed transcription
        
    Returns:
        Complete LLM prompt string
    """
    if not processed_transcription:
        return ""
    
    pipeline = get_pipeline()
    
    if pipeline is not None:
        try:
            from rag.preprocessing.tid_preprocessor import TIDPreprocessor
            from rag.prompt_builder.templates import build_user_prompt
            from rag.prompt_builder.system_instructions import build_dynamic_system_instruction
            from rag.prompt_builder.few_shot_builder import FewShotBuilder
            
            # Preprocess to get linguistic features
            preprocessor = TIDPreprocessor()
            preprocessed = preprocessor.preprocess(processed_transcription)
            
            # Retrieve context
            retrieval_result = pipeline.retriever.retrieve(preprocessed.processed)
            
            # Build few-shot examples
            few_shot_builder = FewShotBuilder()
            few_shot_examples = few_shot_builder.build_examples(
                detected_tense=preprocessed.detected_tense,
                is_question=preprocessed.is_question,
                is_negative=preprocessed.is_negative,
                repetitions=preprocessed.repetitions,
                compound_words=preprocessed.compound_words,
                hafiza_results=retrieval_result.similar_translations,
            )
            
            # Build RAG context
            rag_context = retrieval_result.to_context_string()
            
            # Build complete prompt
            user_prompt = build_user_prompt(
                transcription=preprocessed.processed,
                few_shot_examples=few_shot_examples,
                rag_context=rag_context,
                detected_tense=preprocessed.detected_tense,
                tense_source=preprocessed.tense_source,
                is_question=preprocessed.is_question,
                is_negative=preprocessed.is_negative,
                repetitions=preprocessed.repetitions,
                linguistic_hints=preprocessed.linguistic_hints,
            )
            
            return user_prompt
            
        except Exception as e:
            print(f"RAG prompt building hatasi: {e}")
    
    # Fallback to simple prompt
    return _simple_create_prompt(processed_transcription)


def translate_with_rag(transcription: str) -> Dict[str, Any]:
    """
    Full RAG translation pipeline.
    
    Args:
        transcription: Raw TID transcription
        
    Returns:
        Dictionary with translation results including alternatives
    """
    pipeline = get_pipeline()
    
    if pipeline is None:
        return {
            "translation": "",
            "confidence": 0,
            "alternatives": [],
            "error": "RAG sistemi kulanilamiyor",
            "rag_used": False,
        }
    
    try:
        result = pipeline.translate(transcription)
        
        alternatives = []
        if result.translation_result and result.translation_result.alternatives:
            for alt in result.translation_result.alternatives:
                alternatives.append({
                    "translation": alt.translation,
                    "confidence": alt.confidence,
                    "explanation": alt.explanation,
                })
        
        return {
            "translation": result.best_translation,
            "confidence": result.confidence,
            "alternatives": alternatives,
            "explanation": alternatives[0]["explanation"] if alternatives else "",
            "error": None,
            "rag_used": True,
        }
        
    except Exception as e:
        import traceback
        print(f"RAG ceviri hatasi: {traceback.format_exc()}")
        return {
            "translation": "",
            "confidence": 0,
            "alternatives": [],
            "error": str(e),
            "rag_used": False,
        }


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Basit dinamik RAG bilgi tabani (fallback icin)
RAG_KNOWLEDGE_BASE = {
    "AY": "Ek Bilgi: 'AY' isareti hem 'gok cismi' hem de 'takvim ayi' anlamina gelebilir.",
    "YUZMEK": "Ek Bilgi: 'YUZMEK' isareti hem 'suda yuzmek' hem de 'deri yuzmek' anlamina gelebilir.",
}
