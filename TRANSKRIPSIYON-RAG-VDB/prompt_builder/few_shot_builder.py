"""
Few-Shot Example Builder for TID Translation
=============================================
Builds dynamic few-shot examples based on linguistic features.
Combines static examples with dynamic examples from TID_Hafiza.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FewShotExample:
    """A single few-shot example."""
    input_transcription: str
    output_translation: str
    category: str  # linguistic category


class FewShotBuilder:
    """
    Builds few-shot examples for LLM prompts.
    
    Strategy:
    1. Select static examples based on detected linguistic features
    2. Add dynamic examples from TID_Hafiza if similar and high quality
    3. Limit total examples to avoid prompt bloat
    """
    
    # =========================================================================
    # STATIC EXAMPLES - Categorized by linguistic feature
    # =========================================================================
    
    STATIC_EXAMPLES = {
        # Topic-Comment to SOV transformation
        "topic_comment": [
            FewShotExample(
                input_transcription="ARABA BEN ALMAK DÜN",
                output_translation="Dun araba aldim.",
                category="topic_comment"
            ),
            FewShotExample(
                input_transcription="OKUL COCUK GITMEK",
                output_translation="Cocuk okula gitti.",
                category="topic_comment"
            ),
        ],
        
        # Past tense examples
        "tense_past": [
            FewShotExample(
                input_transcription="OKUL GITMEK _GECMIS_ZAMAN_",
                output_translation="Okula gittim.",
                category="tense_past"
            ),
            FewShotExample(
                input_transcription="DÜN ARKADAŞ BULUŞMAK KAHVE İÇMEK",
                output_translation="Dun arkadasimla bulustum ve kahve ictik.",
                category="tense_past"
            ),
            FewShotExample(
                input_transcription="YEMEK YEMEK _GECMIS_ZAMAN_",
                output_translation="Yemegi yedim.",
                category="tense_past"
            ),
        ],
        
        # Future tense examples
        "tense_future": [
            FewShotExample(
                input_transcription="YARIN TOPLANTI VAR",
                output_translation="Yarin toplanti var.",
                category="tense_future"
            ),
            FewShotExample(
                input_transcription="BEN YARIN OKUL GITMEK",
                output_translation="Yarin okula gidecegim.",
                category="tense_future"
            ),
        ],
        
        # Present tense examples
        "tense_present": [
            FewShotExample(
                input_transcription="BEN SIMDI YEMEK YEMEK",
                output_translation="Simdi yemek yiyorum.",
                category="tense_present"
            ),
            FewShotExample(
                input_transcription="O HER GÜN ÇALIŞMAK",
                output_translation="O her gun calisiyor.",
                category="tense_present"
            ),
        ],
        
        # Negation examples
        "negation": [
            FewShotExample(
                input_transcription="BEN GITMEK _NEGASYON_",
                output_translation="Gitmiyorum.",
                category="negation"
            ),
            FewShotExample(
                input_transcription="SEN YEMEK BEGENMEK _NEGASYON_",
                output_translation="Yemegi begenmedin.",
                category="negation"
            ),
            FewShotExample(
                input_transcription="PARA YOK",
                output_translation="Param yok.",
                category="negation"
            ),
        ],
        
        # Question examples
        "question": [
            FewShotExample(
                input_transcription="SEN NEREDE ÇALIŞMAK",
                output_translation="Nerede calisiyorsun?",
                category="question"
            ),
            FewShotExample(
                input_transcription="BU NE",
                output_translation="Bu ne?",
                category="question"
            ),
            FewShotExample(
                input_transcription="SEN YARIN GELMEK",
                output_translation="Yarin gelecek misin?",
                category="question"
            ),
        ],
        
        # Repetition/Emphasis examples
        "repetition": [
            FewShotExample(
                input_transcription="YÜRÜMEK_TEKRAR YORULMAK",
                output_translation="Cok yuruyup yoruldum.",
                category="repetition"
            ),
            FewShotExample(
                input_transcription="GEZMEK_TEKRAR EĞLENMEK",
                output_translation="Bol bol gezdik ve eglendik.",
                category="repetition"
            ),
            FewShotExample(
                input_transcription="BEKLEMEK_TEKRAR AMA GELMEK _NEGASYON_",
                output_translation="Uzun sure bekledim ama gelmedi.",
                category="repetition"
            ),
        ],
        
        # Compound word examples
        "compound": [
            FewShotExample(
                input_transcription="BEN ARABA_SÜRMEK BİLMEK",
                output_translation="Araba kullanmayi biliyorum.",
                category="compound"
            ),
        ],
        
        # General/Default examples
        "general": [
            FewShotExample(
                input_transcription="BEN OKUL GITMEK AMA KAPI KILITLI",
                output_translation="Okula gittim ama kapi kilitliydi.",
                category="general"
            ),
            FewShotExample(
                input_transcription="ANNE YEMEK YAPMAK COCUK YARDIM ETMEK",
                output_translation="Anne yemek yapti, cocuk yardim etti.",
                category="general"
            ),
        ],
    }
    
    # Maximum examples per category from static
    MAX_STATIC_PER_CATEGORY = 2
    
    # Maximum examples from Hafiza (dynamic)
    MAX_DYNAMIC_EXAMPLES = 2
    
    # Minimum similarity threshold for dynamic examples
    SIMILARITY_THRESHOLD = 0.6
    
    # Total maximum examples
    MAX_TOTAL_EXAMPLES = 5
    
    def __init__(self):
        """Initialize the FewShotBuilder."""
        pass
    
    def build_examples(
        self,
        detected_tense: Optional[str] = None,
        is_question: bool = False,
        is_negative: bool = False,
        repetitions: Optional[Dict[str, int]] = None,
        compound_words: Optional[List[str]] = None,
        hafiza_results: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build few-shot examples based on detected linguistic features.
        
        Args:
            detected_tense: Detected tense ("past", "present", "future", None)
            is_question: Whether question markers were detected
            is_negative: Whether negation markers were detected
            repetitions: Dict of repeated words
            compound_words: List of compound words
            hafiza_results: Similar translations from TID_Hafiza
            
        Returns:
            Formatted few-shot examples string
        """
        examples: List[FewShotExample] = []
        
        # 1. Select static examples based on features
        static_examples = self._select_static_examples(
            detected_tense=detected_tense,
            is_question=is_question,
            is_negative=is_negative,
            has_repetition=bool(repetitions),
            has_compound=bool(compound_words),
        )
        examples.extend(static_examples)
        
        # 2. Add dynamic examples from Hafiza
        if hafiza_results:
            dynamic_examples = self._select_dynamic_examples(hafiza_results)
            examples.extend(dynamic_examples)
        
        # 3. Ensure we have at least some examples
        if len(examples) < 2:
            examples.extend(self._get_default_examples())
        
        # 4. Limit total examples
        examples = examples[:self.MAX_TOTAL_EXAMPLES]
        
        # 5. Format as string
        return self._format_examples(examples)
    
    def _select_static_examples(
        self,
        detected_tense: Optional[str],
        is_question: bool,
        is_negative: bool,
        has_repetition: bool,
        has_compound: bool,
    ) -> List[FewShotExample]:
        """Select static examples based on detected features."""
        selected = []
        
        # Priority order: most specific first
        
        # Question examples (high priority)
        if is_question:
            selected.extend(self._get_from_category("question", 1))
        
        # Negation examples
        if is_negative:
            selected.extend(self._get_from_category("negation", 1))
        
        # Repetition examples
        if has_repetition:
            selected.extend(self._get_from_category("repetition", 1))
        
        # Compound word examples
        if has_compound:
            selected.extend(self._get_from_category("compound", 1))
        
        # Tense-specific examples
        if detected_tense == "past":
            selected.extend(self._get_from_category("tense_past", 1))
        elif detected_tense == "future":
            selected.extend(self._get_from_category("tense_future", 1))
        elif detected_tense == "present":
            selected.extend(self._get_from_category("tense_present", 1))
        
        # Add topic-comment example if we have room
        if len(selected) < 3:
            selected.extend(self._get_from_category("topic_comment", 1))
        
        return selected
    
    def _get_from_category(self, category: str, count: int) -> List[FewShotExample]:
        """Get examples from a specific category."""
        examples = self.STATIC_EXAMPLES.get(category, [])
        return examples[:count]
    
    def _get_default_examples(self) -> List[FewShotExample]:
        """Get default general examples."""
        return self.STATIC_EXAMPLES.get("general", [])[:2]
    
    def _select_dynamic_examples(
        self, 
        hafiza_results: List[Dict]
    ) -> List[FewShotExample]:
        """
        Select dynamic examples from Hafiza results.
        
        Args:
            hafiza_results: List of similar translations from TID_Hafiza
            
        Returns:
            List of FewShotExample from Hafiza
        """
        dynamic = []
        
        for result in hafiza_results[:self.MAX_DYNAMIC_EXAMPLES]:
            # Check similarity threshold
            similarity = result.get("similarity", 0)
            if similarity < self.SIMILARITY_THRESHOLD:
                continue
            
            # Extract transcription and translation
            transcription = result.get("transkripsiyon", "")
            translation = result.get("ceviri", "")
            
            if transcription and translation:
                dynamic.append(FewShotExample(
                    input_transcription=transcription,
                    output_translation=translation,
                    category="hafiza_dynamic"
                ))
        
        return dynamic
    
    def _format_examples(self, examples: List[FewShotExample]) -> str:
        """Format examples as a string for the prompt."""
        if not examples:
            return ""
        
        lines = ["## ORNEKLER (FEW-SHOT)"]
        
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n### Ornek {i}")
            lines.append(f"Transkripsiyon: {ex.input_transcription}")
            lines.append(f"Ceviri: {ex.output_translation}")
        
        return "\n".join(lines)
    
    def get_all_static_examples(self) -> List[FewShotExample]:
        """Get all static examples (for testing/debugging)."""
        all_examples = []
        for category, examples in self.STATIC_EXAMPLES.items():
            all_examples.extend(examples)
        return all_examples


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def build_few_shot_examples(
    detected_tense: Optional[str] = None,
    is_question: bool = False,
    is_negative: bool = False,
    repetitions: Optional[Dict[str, int]] = None,
    compound_words: Optional[List[str]] = None,
    hafiza_results: Optional[List[Dict]] = None,
) -> str:
    """
    Convenience function to build few-shot examples.
    
    Args:
        Same as FewShotBuilder.build_examples()
        
    Returns:
        Formatted few-shot examples string
    """
    builder = FewShotBuilder()
    return builder.build_examples(
        detected_tense=detected_tense,
        is_question=is_question,
        is_negative=is_negative,
        repetitions=repetitions,
        compound_words=compound_words,
        hafiza_results=hafiza_results,
    )
