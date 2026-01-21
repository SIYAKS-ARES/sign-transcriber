"""
Response Parser for LLM Output
==============================
Parses multi-alternative translation responses from LLM using regex.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TranslationAlternative:
    """A single translation alternative."""
    translation: str
    confidence: int  # 1-10
    explanation: str
    alternative_number: int = 1
    
    def __str__(self) -> str:
        return f"[{self.confidence}/10] {self.translation}"


@dataclass
class TranslationResult:
    """Complete translation result with all alternatives."""
    alternatives: List[TranslationAlternative] = field(default_factory=list)
    raw_response: str = ""
    parse_errors: List[str] = field(default_factory=list)
    
    @property
    def best(self) -> Optional[TranslationAlternative]:
        """Get the highest confidence alternative."""
        if not self.alternatives:
            return None
        return max(self.alternatives, key=lambda x: x.confidence)
    
    @property
    def best_translation(self) -> str:
        """Get the best translation text."""
        best = self.best
        return best.translation if best else ""
    
    @property
    def is_successful(self) -> bool:
        """Check if parsing was successful (at least one alternative)."""
        return len(self.alternatives) > 0
    
    def get_sorted_alternatives(self) -> List[TranslationAlternative]:
        """Get alternatives sorted by confidence (highest first)."""
        return sorted(self.alternatives, key=lambda x: x.confidence, reverse=True)
    
    def to_display_string(self) -> str:
        """Format result for display."""
        if not self.alternatives:
            return f"Ceviri basarisiz. Ham yanit:\n{self.raw_response}"
        
        lines = []
        sorted_alts = self.get_sorted_alternatives()
        
        lines.append(f"En iyi ceviri: {sorted_alts[0].translation}")
        lines.append(f"Guven: {sorted_alts[0].confidence}/10")
        lines.append("")
        lines.append("Tum alternatifler:")
        
        for i, alt in enumerate(sorted_alts, 1):
            lines.append(f"  {i}. [{alt.confidence}/10] {alt.translation}")
            if alt.explanation:
                lines.append(f"     ({alt.explanation})")
        
        return "\n".join(lines)


class ResponseParser:
    """
    Parser for LLM translation responses.
    
    Expects responses in the format:
    
    ## ALTERNATIF 1
    Ceviri: [translation]
    Guven: [1-10]/10
    Aciklama: [explanation]
    
    ## ALTERNATIF 2
    ...
    """
    
    # Main pattern for parsing alternatives
    # Matches:
    # ## ALTERNATIF 1 (or ALTERNATİF, Alternatif, etc.)
    # Ceviri: [text]
    # Guven: [number]/10
    # Aciklama: [text]
    ALTERNATIVE_PATTERN = re.compile(
        r"##\s*(?:ALTERNATİF|ALTERNATIF|Alternatif)\s*(\d+)\s*\n+"
        r"(?:Çeviri|Ceviri|CEVIRI):\s*(.+?)\n+"
        r"(?:Güven|Guven|GUVEN):\s*(\d+)\s*/\s*10\s*\n+"
        r"(?:Açıklama|Aciklama|ACIKLAMA):\s*(.+?)(?=##\s*(?:ALTERNATİF|ALTERNATIF|Alternatif)|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    
    # Fallback pattern for simpler responses
    SIMPLE_PATTERN = re.compile(
        r"(?:Çeviri|Ceviri|CEVIRI):\s*(.+?)(?:\n|$)",
        re.IGNORECASE
    )
    
    # Pattern for confidence extraction (fallback)
    CONFIDENCE_PATTERN = re.compile(
        r"(?:Güven|Guven|GUVEN):\s*(\d+)\s*/\s*10",
        re.IGNORECASE
    )
    
    # Pattern for explanation extraction (fallback)
    EXPLANATION_PATTERN = re.compile(
        r"(?:Açıklama|Aciklama|ACIKLAMA):\s*(.+?)(?:\n\n|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    
    def parse(self, response: str) -> TranslationResult:
        """
        Parse LLM response into TranslationResult.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            TranslationResult with parsed alternatives
        """
        result = TranslationResult(raw_response=response)
        
        # Try main pattern first
        alternatives = self._parse_alternatives(response)
        
        if alternatives:
            result.alternatives = alternatives
        else:
            # Try fallback parsing
            fallback = self._parse_fallback(response)
            if fallback:
                result.alternatives = [fallback]
                result.parse_errors.append("Standart format bulunamadi, fallback kullanildi")
            else:
                result.parse_errors.append("Ceviri parse edilemedi")
        
        return result
    
    def _parse_alternatives(self, response: str) -> List[TranslationAlternative]:
        """Parse alternatives using the main pattern."""
        alternatives = []
        
        matches = self.ALTERNATIVE_PATTERN.findall(response)
        
        for match in matches:
            alt_num, translation, confidence, explanation = match
            
            # Clean up the extracted values
            translation = translation.strip()
            explanation = explanation.strip()
            
            # Handle multi-line translations/explanations
            translation = re.sub(r'\s+', ' ', translation)
            explanation = re.sub(r'\s+', ' ', explanation)
            
            try:
                confidence_int = int(confidence)
                # Clamp to valid range
                confidence_int = max(1, min(10, confidence_int))
            except ValueError:
                confidence_int = 5  # Default
            
            alternatives.append(TranslationAlternative(
                translation=translation,
                confidence=confidence_int,
                explanation=explanation,
                alternative_number=int(alt_num),
            ))
        
        return alternatives
    
    def _parse_fallback(self, response: str) -> Optional[TranslationAlternative]:
        """Fallback parsing for non-standard responses."""
        
        # Try to extract translation
        translation_match = self.SIMPLE_PATTERN.search(response)
        if not translation_match:
            # Last resort: just use the first non-empty line
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            if lines:
                translation = lines[0]
                # Remove common prefixes
                for prefix in ['Ceviri:', 'Çeviri:', '-', '*']:
                    if translation.startswith(prefix):
                        translation = translation[len(prefix):].strip()
            else:
                return None
        else:
            translation = translation_match.group(1).strip()
        
        # Try to extract confidence
        confidence_match = self.CONFIDENCE_PATTERN.search(response)
        confidence = int(confidence_match.group(1)) if confidence_match else 7
        
        # Try to extract explanation
        explanation_match = self.EXPLANATION_PATTERN.search(response)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        return TranslationAlternative(
            translation=translation,
            confidence=confidence,
            explanation=explanation,
            alternative_number=1,
        )
    
    def extract_first_translation(self, response: str) -> str:
        """
        Quick extraction of just the first translation.
        Useful for simple use cases.
        """
        result = self.parse(response)
        if result.best:
            return result.best.translation
        
        # Absolute fallback: return first line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[0] if lines else response


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def parse_translation_response(response: str) -> TranslationResult:
    """
    Convenience function to parse an LLM response.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        TranslationResult with parsed alternatives
    """
    parser = ResponseParser()
    return parser.parse(response)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample response
    sample_response = """
## ALTERNATIF 1
Ceviri: Dun arkadasimla bulustum ve kahve ictik.
Guven: 9/10
Aciklama: DUN zaman zarfi gecmis zaman gerektiriyor.

## ALTERNATIF 2
Ceviri: Dun arkadasla bulusup kahve ictim.
Guven: 8/10
Aciklama: Daha kisa ve gunluk dil.

## ALTERNATIF 3
Ceviri: Arkadasimla dun kahve icmeye gittim.
Guven: 7/10
Aciklama: Farkli cumle yapisi.
"""
    
    parser = ResponseParser()
    result = parser.parse(sample_response)
    
    print("Parse Result:")
    print("=" * 50)
    print(result.to_display_string())
    print()
    print(f"Parse errors: {result.parse_errors}")
    print(f"Is successful: {result.is_successful}")
