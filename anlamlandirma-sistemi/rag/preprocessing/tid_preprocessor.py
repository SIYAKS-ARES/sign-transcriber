"""
TID Preprocessor - Turkish Sign Language Transcription Preprocessing
=====================================================================
Handles TID-specific markers, linguistic analysis, and preprocessing.
Implements hybrid approach: basic detection in preprocessing + hints for LLM.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PreprocessedInput:
    """Preprocessed transcription with linguistic analysis."""
    
    # Original and processed text
    original: str
    processed: str
    word_list: List[str]
    
    # Temporal analysis
    detected_tense: Optional[str] = None  # "past", "present", "future", None
    tense_source: Optional[str] = None    # "explicit", "inferred", None
    
    # Sentence type hints
    is_question: bool = False
    is_negative: bool = False
    
    # TID-specific features
    repetitions: Dict[str, int] = field(default_factory=dict)  # {"GEZMEK": 2}
    compound_words: List[str] = field(default_factory=list)    # ["ARABA_SURMEK"]
    
    # Linguistic hints for LLM
    linguistic_hints: Dict[str, any] = field(default_factory=dict)
    
    # New grammar analysis fields
    grammar_hints: Dict[str, str] = field(default_factory=dict) # {"BERABER": "vasita"}
    verb_classes: List[str] = field(default_factory=list)       # ["yonelimli", "duygu"]
    
    # Detected markers (for debugging/logging)
    detected_markers: List[str] = field(default_factory=list)


class TIDPreprocessor:
    """
    TID Transcription Preprocessor
    
    Detects and processes TID-specific linguistic features:
    - Tense markers (DUN, BUGUN, YARIN, BITMEK, etc.)
    - Question markers (NEREDE, NE, KIM, etc.)
    - Negation markers (DEGIL, YOK)
    - Repetition (GEZMEK GEZMEK -> pekistirme)
    - Compound words (ARABA^SURMEK)
    """
    
    # =========================================================================
    # MARKER DEFINITIONS
    # =========================================================================
    
    # Explicit tense markers (zarflar)
    # Includes both Turkish special characters and ASCII variants
    TENSE_MARKERS = {
        # Past tense markers (Turkish + ASCII variants)
        "DÜN": "past", "DUN": "past",
        "DÜNKÜ": "past", "DUNKU": "past",
        "GEÇEN": "past", "GECEN": "past",
        "GEÇMIS": "past", "GECMIS": "past",
        "GEÇMIŞ": "past", "GECMIS": "past",
        "ÖNCE": "past", "ONCE": "past",
        "EVVEL": "past",
        
        # Present tense markers (Turkish + ASCII variants)
        "BUGÜN": "present", "BUGUN": "present",
        "BUGÜNKÜ": "present", "BUGUNKU": "present",
        "ŞIMDI": "present", "SIMDI": "present",
        "ŞİMDİ": "present",
        "HEMEN": "present",
        "ŞU AN": "present", "SU AN": "present",
        
        # Future tense markers (Turkish + ASCII variants)
        "YARIN": "future",
        "GELECEK": "future",
        "SONRA": "future",
        "İLERİDE": "future", "ILERIDE": "future",
    }
    
    # Completion markers (indicate past tense)
    COMPLETION_MARKERS = {
        "BİTMEK": "past",
        "BITMEK": "past",
        "TAMAM": "past",
        "OLDU": "past",
        "BİTTİ": "past",
        "BITTI": "past",
    }
    
    # Negation markers
    NEGATION_MARKERS = {
        "DEĞİL", "DEGIL", "DEGİL",
        "YOK",
        "HAYIR",
        "ASLA",
        "HİÇ", "HIC", "HIÇBIR",
    }
    
    # Question markers
    QUESTION_MARKERS = {
        "NEREDE", "NEREYE", "NEREDEN", "NERESI",
        "NE", "NEYI", "NEYİ",
        "KIM", "KİM", "KIMI", "KİMİ",
        "NASIL", "NASIL",
        "NEDEN", "NIÇIN", "NICIN",
        "NİYE", "NIYE",
        "KAÇTA", "KACTA", "KAÇ", "KAC",
        "HANGI", "HANGİSİ",
        "MI", "MU", "MI", "MÜ",  # Question suffixes
    }
    
    # Personal pronouns (for context)
    PRONOUNS = {
        "BEN": "1sg",
        "SEN": "2sg", 
        "O": "3sg",
        "BIZ": "1pl", "BİZ": "1pl",
        "SIZ": "2pl", "SİZ": "2pl",
        "ONLAR": "3pl",
    }
    
    # Special replacement markers for processed output
    MARKER_REPLACEMENTS = {
        "BİTMEK": "_GECMIS_ZAMAN_",
        "BITMEK": "_GECMIS_ZAMAN_",
        "TAMAM": "_GECMIS_ZAMAN_",
        "DEĞİL": "_NEGASYON_",
        "DEGIL": "_NEGASYON_",
        "YOK": "_NEGASYON_",
    }
    
    # Grammar Markers (New)
    GRAMMAR_MARKERS = {
        "BERABER": "vasita_birliktelik", # -la / -le
        "LAZIM": "gereklilik",           # -malı / -meli
        "ICIN": "amac_sonuc",            # -mek için
        "IÇIN": "amac_sonuc",
        "BILMEK": "yeterlilik",          # -ebilmek
        "BİLMEK": "yeterlilik",
        "HIC": "yoksunluk",              # -sız / -suz
        "HİÇ": "yoksunluk",
    }

    # Verb Class definitions (New)
    VERB_CLASSES = {
        "GELMEK": "yonelimli", "GITMEK": "yonelimli", "GİTMEK": "yonelimli",
        "ANLATMAK": "yonelimli", "VERMEK": "yonelimli",
        "SOYLEMEK": "yonelimli", "SÖYLEMEK": "yonelimli",
        
        "BIKMAK": "duygu", "YORULMAK": "duygu",
        "SEVMEK": "duygu", "BEGENMEK": "duygu", "BEĞENMEK": "duygu",
        "ISTEMEK": "duygu", "İSTEMEK": "duygu",
        "UZULMEK": "duygu", "ÜZÜLMEK": "duygu",

        "ANLAMAK": "bilissel", "UNUTMAK": "bilissel",
        "DUSUNMEK": "bilissel", "DÜŞÜNMEK": "bilissel",
        "HATIRLAMAK": "bilissel", "EZBERLEMEK": "bilissel",
        
        "TELEFON_ARAMAK": "ikonik", "YAZMAK": "ikonik",
        "DOKMEK": "ikonik", "DÖKMEK": "ikonik",
    }
    
    # =========================================================================
    # MAIN PREPROCESSING METHOD
    # =========================================================================
    
    def preprocess(self, transcription: str) -> PreprocessedInput:
        """
        Preprocess TID transcription and extract linguistic features.
        
        Args:
            transcription: Raw TID transcription string
            
        Returns:
            PreprocessedInput with all linguistic analysis
        """
        # Normalize input
        original = transcription.strip()
        text = original.upper()
        
        # Split into words
        words = text.split()
        
        # 1. Detect and process repetitions
        words, repetitions = self._detect_repetitions(words)
        
        # 2. Handle compound words (ARABA^SURMEK -> ARABA_SURMEK)
        words, compound_words = self._handle_compounds(words)
        
        # 3. Detect tense
        tense, tense_source = self._detect_tense(words)
        
        # 4. Detect question markers
        is_question = self._detect_question(words)
        
        # 5. Detect negation
        is_negative = self._detect_negation(words)
        
        # 6. Detect grammar markers (New)
        grammar_hints = self._detect_grammar_markers(words)
        
        # 7. Detect verb classes (New)
        verb_classes = self._detect_verb_classes(words)

        # 8. Replace special markers with semantic markers
        processed_words, detected_markers = self._replace_markers(words)
        
        # 7. Build linguistic hints for LLM
        linguistic_hints = self._build_linguistic_hints(
            words=words,
            tense=tense,
            is_question=is_question,
            is_negative=is_negative,
            repetitions=repetitions,
        )
        
        return PreprocessedInput(
            original=original,
            processed=" ".join(processed_words),
            word_list=words,
            detected_tense=tense,
            tense_source=tense_source,
            is_question=is_question,
            is_negative=is_negative,
            repetitions=repetitions,
            compound_words=compound_words,
            linguistic_hints=linguistic_hints,
            detected_markers=detected_markers,
            grammar_hints=grammar_hints,
            verb_classes=verb_classes,
        )
    
    # =========================================================================
    # DETECTION METHODS
    # =========================================================================
    
    def _detect_repetitions(self, words: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        Detect consecutive word repetitions (pekistirme/emphasis).
        
        TID uses repetition for:
        - Continuous action: GEZMEK GEZMEK -> "bol bol gezmek"
        - Intensity: GUZEL GUZEL -> "cok guzel"
        - Duration: BEKLEMEK BEKLEMEK -> "uzun sure beklemek"
        
        Returns:
            Tuple of (processed words, repetition dict)
        """
        if len(words) < 2:
            return words, {}
        
        processed = []
        repetitions = {}
        i = 0
        
        while i < len(words):
            word = words[i]
            count = 1
            
            # Count consecutive repetitions
            while i + count < len(words) and words[i + count] == word:
                count += 1
            
            if count > 1:
                # Mark as repetition
                repetitions[word] = count
                processed.append(f"{word}_TEKRAR")
                # Skip the repeated words
                i += count
            else:
                processed.append(word)
                i += 1
        
        return processed, repetitions
    
    def _handle_compounds(self, words: List[str]) -> Tuple[List[str], List[str]]:
        """
        Handle compound words marked with ^ character.
        
        Example: ARABA^SURMEK -> ARABA_SURMEK (single concept)
        
        Returns:
            Tuple of (processed words, compound word list)
        """
        processed = []
        compounds = []
        
        for word in words:
            if "^" in word:
                # Replace ^ with _ to indicate compound
                compound = word.replace("^", "_")
                processed.append(compound)
                compounds.append(compound)
            else:
                processed.append(word)
        
        return processed, compounds
    
    def _detect_tense(self, words: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect tense from explicit markers or completion markers.
        
        Priority:
        1. Explicit time adverbs (DUN, YARIN, etc.)
        2. Completion markers (BITMEK, TAMAM)
        3. None (let LLM infer from context)
        
        Returns:
            Tuple of (tense, source) where source is "explicit" or "inferred"
        """
        # Check explicit tense markers first
        for word in words:
            # Remove _TEKRAR suffix for checking
            clean_word = word.replace("_TEKRAR", "")
            
            if clean_word in self.TENSE_MARKERS:
                return self.TENSE_MARKERS[clean_word], "explicit"
        
        # Check completion markers (usually at end of sentence)
        for word in reversed(words[-3:]) if len(words) >= 3 else reversed(words):
            clean_word = word.replace("_TEKRAR", "")
            
            if clean_word in self.COMPLETION_MARKERS:
                return self.COMPLETION_MARKERS[clean_word], "inferred"
        
        return None, None
    
    def _detect_question(self, words: List[str]) -> bool:
        """
        Detect if the transcription is a question.
        
        Detection:
        - Explicit question words (NEREDE, NE, KIM, etc.)
        - Question particles (MI, MU)
        """
        for word in words:
            clean_word = word.replace("_TEKRAR", "")
            if clean_word in self.QUESTION_MARKERS:
                return True
        return False
    
    def _detect_negation(self, words: List[str]) -> bool:
        """
        Detect if the transcription contains negation.
        
        Note: In TID, negation is often expressed through NMMs (head shake),
        which the visual model cannot capture. This only detects explicit markers.
        """
        for word in words:
            clean_word = word.replace("_TEKRAR", "")
            if clean_word in self.NEGATION_MARKERS:
                return True
        return False
    
    def _replace_markers(self, words: List[str]) -> Tuple[List[str], List[str]]:
        """
        Replace special TID markers with semantic markers.
        
        Example: BITMEK -> _GECMIS_ZAMAN_
        
        Returns:
            Tuple of (processed words, detected markers list)
        """
        processed = []
        detected = []
        
        for word in words:
            clean_word = word.replace("_TEKRAR", "")
            
            if clean_word in self.MARKER_REPLACEMENTS:
                replacement = self.MARKER_REPLACEMENTS[clean_word]
                # Keep _TEKRAR suffix if present
                if "_TEKRAR" in word:
                    processed.append(f"{replacement}_TEKRAR")
                else:
                    processed.append(replacement)
                detected.append(f"{clean_word}->{replacement}")
            else:
                processed.append(word)
        
        return processed, detected
    
    def _build_linguistic_hints(
        self,
        words: List[str],
        tense: Optional[str],
        is_question: bool,
        is_negative: bool,
        repetitions: Dict[str, int],
    ) -> Dict[str, any]:
        """
        Build linguistic hints dictionary for LLM guidance.
        """
        hints = {
            "has_temporal_marker": tense is not None,
            "has_explicit_question": is_question,
            "has_explicit_negation": is_negative,
            "has_repetition": len(repetitions) > 0,
            "word_count": len(words),
        }
        
        # Detect topic (first noun/pronoun is usually topic in TID)
        topic = self._detect_topic(words)
        if topic:
            hints["likely_topic"] = topic
        
        # Detect pronouns for subject inference
        pronouns = [w for w in words if w.replace("_TEKRAR", "") in self.PRONOUNS]
        if pronouns:
            hints["pronouns"] = pronouns
        
        # Detect if there's a verb (ends with -MEK/-MAK typically)
        verbs = [w for w in words if w.replace("_TEKRAR", "").endswith(("MEK", "MAK"))]
        if verbs:
            hints["verbs"] = verbs
        
        return hints
    
    def _detect_topic(self, words: List[str]) -> Optional[str]:
        """
        Detect likely topic in Topic-Comment structure.
        
        In TID, the topic (main focus) often comes first.
        Usually a noun or pronoun.
        """
        if not words:
            return None
        
        # First word is often the topic
        first_word = words[0].replace("_TEKRAR", "")
        
        # Skip time adverbs as topic
        if first_word in self.TENSE_MARKERS:
            if len(words) > 1:
                return words[1].replace("_TEKRAR", "")
            return None
        
        return first_word
    
    def _detect_grammar_markers(self, words: List[str]) -> Dict[str, str]:
        """
        Detect grammatical markers like BERABER, LAZIM.
        """
        hints = {}
        for word in words:
            clean = word.replace("_TEKRAR", "")
            if clean in self.GRAMMAR_MARKERS:
                marker_type = self.GRAMMAR_MARKERS[clean]
                hints[clean] = marker_type
        return hints

    def _detect_verb_classes(self, words: List[str]) -> List[str]:
         """Detect verb classes for context cues."""
         classes = set()
         for word in words:
             clean = word.replace("_TEKRAR", "")
             # Check if word is exactly a verb class key
             if clean in self.VERB_CLASSES:
                 classes.add(self.VERB_CLASSES[clean])
         return list(classes)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_words_for_rag(preprocessed: PreprocessedInput) -> List[str]:
    """
    Extract words from preprocessed input for RAG lookup.
    Removes markers and returns clean word list.
    """
    words = []
    
    for word in preprocessed.word_list:
        # Remove special markers
        clean = word
        for marker in ["_TEKRAR", "_GECMIS_ZAMAN_", "_NEGASYON_"]:
            clean = clean.replace(marker, "")
        
        # Skip empty or marker-only entries
        if clean and clean not in {"_GECMIS_ZAMAN_", "_NEGASYON_"}:
            words.append(clean)
    
    return words


def get_tense_hint(preprocessed: PreprocessedInput) -> str:
    """
    Get human-readable tense hint for prompts.
    """
    if preprocessed.detected_tense == "past":
        return "Gecmis zaman (-di, -mis) kullan."
    elif preprocessed.detected_tense == "future":
        return "Gelecek zaman (-ecek, -acak) kullan."
    elif preprocessed.detected_tense == "present":
        return "Simdiki veya genis zaman (-iyor, -ir) kullan."
    else:
        return "Zaman belirsiz - baglamdan cikar veya alternatiflerde varyasyon sun."
