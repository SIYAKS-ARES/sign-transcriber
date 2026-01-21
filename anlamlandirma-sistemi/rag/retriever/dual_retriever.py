"""
Dual-Level Retrieval System
===========================
Two-stage retrieval: Memory (Hafiza) for similar sentences, 
Dictionary (Sozluk) for word-level information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from rag.config import (
    HAFIZA_TOP_K,
    SOZLUK_TOP_K,
    SIMILARITY_THRESHOLD,
)
from rag.tid_collections.sozluk_collection import SozlukCollection
from rag.tid_collections.hafiza_collection import HafizaCollection


@dataclass
class RetrievalResult:
    """Result from dual retrieval."""
    # Level 1: Similar sentences from memory
    similar_translations: List[Dict] = field(default_factory=list)
    
    # Level 2: Word-level information from dictionary
    word_info: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # Original query
    query: str = ""
    
    # Not found words (for tracking)
    not_found_words: List[str] = field(default_factory=list)
    
    def has_memory_results(self) -> bool:
        """Check if any similar translations were found."""
        return len(self.similar_translations) > 0
    
    def has_word_info(self) -> bool:
        """Check if any word information was found."""
        return any(len(v) > 0 for v in self.word_info.values())
    
    def get_word_definitions(self) -> Dict[str, Dict]:
        """
        Get simplified word definitions for prompt building.
        Returns dict mapping word to its definition info.
        """
        definitions = {}
        for word, infos in self.word_info.items():
            if infos:
                info = infos[0]
                metadata = info.get("metadata", {})
                match_type = info.get("match_type", "unknown")
                
                if match_type in ["exact", "partial"]:
                    definitions[word] = {
                        "kelime": metadata.get("kelime", word),
                        "tur": metadata.get("tur", "Bilinmiyor"),
                        "aciklama": metadata.get("aciklama", ""),
                        "ornek_transkripsiyon": metadata.get("ornek_transkripsiyon", ""),
                        "ornek_ceviri": metadata.get("ornek_ceviri", ""),
                        "match_type": match_type,
                    }
        return definitions
    
    def to_context_string(self) -> str:
        """Convert retrieval results to a formatted context string for LLM."""
        parts = []
        
        # =====================================================================
        # Section 1: Similar translations from memory (few-shot reference)
        # =====================================================================
        parts.append("## BENZER CEVIRI ORNEKLERI (Hafiza)")
        if self.similar_translations:
            for i, trans in enumerate(self.similar_translations[:3], 1):
                transkripsiyon = trans.get('transkripsiyon', '')
                ceviri = trans.get('ceviri', '')
                similarity = trans.get('similarity', 0)
                parts.append(f"- {transkripsiyon} -> {ceviri} (benzerlik: {similarity:.2f})")
        else:
            parts.append("- Benzer ceviri bulunamadi.")
        
        # =====================================================================
        # Section 2: Word-level information from dictionary
        # =====================================================================
        parts.append("")
        parts.append("## KELIME BILGILERI (Sozluk)")
        
        found_words = []
        not_found_words = []
        
        for word, infos in self.word_info.items():
            if infos:
                info = infos[0]  # Take first result
                metadata = info.get("metadata", {})
                match_type = info.get("match_type", "unknown")
                
                # Only include exact or partial matches (same word family)
                if match_type in ["exact", "partial"]:
                    found_words.append((word, info))
                else:
                    not_found_words.append(word)
            else:
                not_found_words.append(word)
        
        # Show found words with their details
        for word, info in found_words:
            metadata = info.get("metadata", {})
            match_type = info.get("match_type", "unknown")
            found_word = metadata.get("kelime", word)
            
            # Word header with match type
            if match_type == "exact":
                parts.append(f"\n### {word} [TAM ESLEME]")
            else:
                parts.append(f"\n### {word} -> {found_word} [KISMI ESLEME]")
            
            # Word type
            tur = metadata.get('tur', 'Bilinmiyor')
            parts.append(f"- Tur: {tur}")
            
            # Description (truncated)
            aciklama = metadata.get('aciklama', '')
            if aciklama:
                if len(aciklama) > 150:
                    aciklama = aciklama[:150] + "..."
                parts.append(f"- Aciklama: {aciklama}")
            
            # Example sentence (TID structure example)
            ornek_transkripsiyon = metadata.get('ornek_transkripsiyon', '')
            ornek_ceviri = metadata.get('ornek_ceviri', '')
            
            if ornek_transkripsiyon and ornek_ceviri:
                # Truncate if too long
                if len(ornek_ceviri) > 100:
                    ornek_ceviri = ornek_ceviri[:100] + "..."
                parts.append(f"- TID Ornegi: {ornek_transkripsiyon} -> {ornek_ceviri}")
        
        # =====================================================================
        # Section 3: Not found words (LLM should use general knowledge)
        # =====================================================================
        if not_found_words:
            parts.append("")
            parts.append(f"## SOZLUKTE BULUNMAYAN: {', '.join(not_found_words)}")
            parts.append("(Bu kelimeler icin genel Turkce bilgini kullan)")
        
        # Store not found words for later use
        self.not_found_words = not_found_words
        
        if not found_words and not not_found_words:
            parts.append("\nSozlukte eslesen kelime bulunamadi.")
        
        return "\n".join(parts)
    
    def to_compact_context(self) -> str:
        """
        Generate a compact context string for token-limited prompts.
        """
        parts = []
        
        # Similar translations (max 2)
        if self.similar_translations:
            parts.append("Benzer ceviriler:")
            for trans in self.similar_translations[:2]:
                parts.append(f"- {trans['transkripsiyon']} -> {trans['ceviri']}")
        
        # Word info (compact)
        definitions = self.get_word_definitions()
        if definitions:
            parts.append("\nKelimeler:")
            for word, info in definitions.items():
                tur = info.get('tur', '')
                parts.append(f"- {word}: {tur}")
        
        # Not found
        not_found = [w for w, infos in self.word_info.items() 
                     if not infos or infos[0].get("match_type") not in ["exact", "partial"]]
        if not_found:
            parts.append(f"\nBulunamayan: {', '.join(not_found)}")
        
        return "\n".join(parts)


class DualRetriever:
    """
    Two-level retrieval system for TID translation.
    
    Level 1 (Hafiza): Find similar transcriptions that have been translated before
    Level 2 (Sozluk): Get word-level information for each gloss in the query
    """
    
    def __init__(
        self,
        sozluk: Optional[SozlukCollection] = None,
        hafiza: Optional[HafizaCollection] = None,
    ):
        """
        Initialize the dual retriever.
        
        Args:
            sozluk: Optional pre-initialized SozlukCollection
            hafiza: Optional pre-initialized HafizaCollection
        """
        self.sozluk = sozluk or SozlukCollection()
        self.hafiza = hafiza or HafizaCollection()
    
    def retrieve(
        self,
        transcription: str,
        hafiza_top_k: int = HAFIZA_TOP_K,
        sozluk_top_k: int = SOZLUK_TOP_K,
    ) -> RetrievalResult:
        """
        Perform dual-level retrieval.
        
        Args:
            transcription: The TID gloss transcription to translate
            hafiza_top_k: Number of similar translations to retrieve
            sozluk_top_k: Number of dictionary entries per word
            
        Returns:
            RetrievalResult with memory and dictionary results
        """
        result = RetrievalResult(query=transcription)
        
        # Normalize transcription
        transcription = transcription.upper().strip()
        
        # Level 1: Memory lookup (similar sentences)
        result.similar_translations = self._retrieve_from_memory(
            transcription, 
            top_k=hafiza_top_k
        )
        
        # Level 2: Word-level dictionary lookup
        words = transcription.split()
        result.word_info = self._retrieve_word_info(
            words, 
            top_k=sozluk_top_k
        )
        
        return result
    
    def _retrieve_from_memory(
        self, 
        transcription: str, 
        top_k: int
    ) -> List[Dict]:
        """
        Level 1: Retrieve similar translations from memory.
        
        Args:
            transcription: The query transcription
            top_k: Number of results to return
            
        Returns:
            List of similar translations with similarity scores
        """
        return self.hafiza.query(transcription, n_results=top_k)
    
    def _retrieve_word_info(
        self, 
        words: List[str], 
        top_k: int
    ) -> Dict[str, List[Dict]]:
        """
        Level 2: Retrieve dictionary information for each word.
        
        Args:
            words: List of gloss words
            top_k: Number of results per word
            
        Returns:
            Dictionary mapping each word to its dictionary entries
        """
        return self.sozluk.query_multiple(words, n_results_per_word=top_k)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the collections."""
        return {
            "sozluk_count": self.sozluk.get_count(),
            "hafiza_count": self.hafiza.get_count(),
        }


def prepare_rag_context(transcription: str) -> str:
    """
    Convenience function to prepare RAG context for a transcription.
    
    Args:
        transcription: The TID gloss transcription
        
    Returns:
        Formatted context string for LLM prompt
    """
    retriever = DualRetriever()
    result = retriever.retrieve(transcription)
    return result.to_context_string()


if __name__ == "__main__":
    # Test the retriever
    print("Testing Dual Retriever...")
    print("=" * 60)
    
    retriever = DualRetriever()
    stats = retriever.get_stats()
    print(f"Sozluk documents: {stats['sozluk_count']}")
    print(f"Hafiza documents: {stats['hafiza_count']}")
    
    # Test query
    test_query = "OKUL GITMEK ISTEMEK"
    print(f"\nTest query: '{test_query}'")
    print("-" * 60)
    
    result = retriever.retrieve(test_query)
    print(result.to_context_string())
