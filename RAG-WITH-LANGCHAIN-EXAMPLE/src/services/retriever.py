"""
R-03: Benzer Soru Getirme Modulu
retrieve_examples(config) fonksiyonu ve katmanli arama stratejisi
"""

from typing import Optional
from dataclasses import dataclass

from ..models.question import RetrievedQuestion
from .embedding_service import EmbeddingPipeline
from .filter_service import FilterService, FilterCriteria, create_fallback_filter


@dataclass
class RetrievalConfig:
    """Retrieval konfigurasyonu"""

    top_k: int = 5
    min_results: int = 3
    max_results: int = 8
    similarity_threshold: float = 0.5
    prefer_lgs: bool = True


class QuestionRetriever:
    """
    Kombinasyona uygun sorulari getiren retrieval servisi
    
    Strateji (rag.md sozlesmesine gore):
    1. Ayni alt_konu
    2. Yakin zorluk (+-1)
    3. Ayni/benzer gorsel_tipi
    4. Metin benzerligi ile siralama
    
    4 seviyeli fallback mekanizmasi ile minimum 3 ornek garantisi
    """

    # Kaynak tipi agirliklari (siralama icin)
    KAYNAK_WEIGHTS = {
        "cikmis": 1.0,
        "ornek": 0.8,
        "baslangic": 0.3,
    }

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        filter_service: FilterService,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Args:
            embedding_pipeline: Embedding ve FAISS index servisi
            filter_service: Metadata filtreleme servisi
            config: Retrieval konfigurasyonu
        """
        self.embedding = embedding_pipeline
        self.filter = filter_service
        self.config = config or RetrievalConfig()

    def retrieve_examples(
        self,
        combination: dict,
        top_k: Optional[int] = None,
    ) -> list[RetrievedQuestion]:
        """
        Kombinasyona uygun ornek sorulari getir
        
        Args:
            combination: {alt_konu, zorluk, gorsel_tipi, lgs_skor}
            top_k: Dondurelecek soru sayisi (None = config'den)
            
        Returns:
            RetrievedQuestion listesi (sirali)
            
        Raises:
            ValueError: Yeterli ornek bulunamazsa
        """
        top_k = top_k or self.config.top_k
        results: list[RetrievedQuestion] = []

        # 4 seviyeli fallback stratejisi
        for level in range(1, 5):
            criteria = create_fallback_filter(combination, level)
            level_results = self._search_with_filter(criteria, top_k)

            # Mevcut sonuclara ekle (tekrarsiz)
            existing_texts = {r.soru_metni for r in results}
            for r in level_results:
                if r.soru_metni not in existing_texts:
                    results.append(r)
                    existing_texts.add(r.soru_metni)

            # Yeterli sonuc varsa cik
            if len(results) >= top_k:
                break

        # Sonuclari yeniden sirala ve kes
        results = self._rerank(results, combination)
        return results[: min(top_k, self.config.max_results)]

    def _search_with_filter(
        self,
        criteria: FilterCriteria,
        top_k: int,
    ) -> list[RetrievedQuestion]:
        """
        Filtreli arama yap
        
        Args:
            criteria: Filtreleme kriterleri
            top_k: Dondurelecek sonuc sayisi
            
        Returns:
            RetrievedQuestion listesi
        """
        # Filtre uygula
        valid_indices = self.filter.filter_indices(criteria)

        if not valid_indices:
            return []

        # Filtrelenmis metadata'lari al
        filtered_metadata = [
            self.embedding.id_to_metadata[i]
            for i in valid_indices
            if i in self.embedding.id_to_metadata
        ]

        # RetrievedQuestion'lara cevir
        results = []
        for meta in filtered_metadata[:top_k]:
            results.append(
                RetrievedQuestion(
                    soru_metni=meta.get("Soru_MetniOCR", ""),
                    alt_konu=meta.get("Alt_Konu", ""),
                    zorluk=meta.get("Zorluk", 0),
                    gorsel_tipi=meta.get("Gorsel_Tipi", ""),
                    kaynak_tipi=meta.get("Kaynak_Tipi", ""),
                    similarity_score=1.0,  # Filtreleme-based
                    metadata=meta,
                )
            )

        return results

    def _search_with_embedding(
        self,
        query: str,
        criteria: FilterCriteria,
        top_k: int,
    ) -> list[RetrievedQuestion]:
        """
        Embedding benzerligi ile arama
        
        Args:
            query: Arama sorgusu (ornek soru metni veya kombinasyon tanimi)
            criteria: On filtreleme kriterleri
            top_k: Dondurelecek sonuc sayisi
            
        Returns:
            RetrievedQuestion listesi (benzerlik skoruna gore sirali)
        """
        # Filtre uygula
        valid_indices = self.filter.filter_indices(criteria)

        if not valid_indices:
            return []

        # Embedding benzerligi ile ara
        search_results = self.embedding.search(
            query=query,
            top_k=top_k * 2,  # Daha fazla ara, sonra filtrele
            filter_indices=valid_indices,
        )

        # RetrievedQuestion'lara cevir
        results = []
        for score, meta in search_results:
            if score >= self.config.similarity_threshold:
                results.append(
                    RetrievedQuestion(
                        soru_metni=meta.get("Soru_MetniOCR", ""),
                        alt_konu=meta.get("Alt_Konu", ""),
                        zorluk=meta.get("Zorluk", 0),
                        gorsel_tipi=meta.get("Gorsel_Tipi", ""),
                        kaynak_tipi=meta.get("Kaynak_Tipi", ""),
                        similarity_score=score,
                        metadata=meta,
                    )
                )

        return results[:top_k]

    def _rerank(
        self,
        results: list[RetrievedQuestion],
        combination: dict,
    ) -> list[RetrievedQuestion]:
        """
        Sonuclari yeniden sirala
        
        Siralama kriterleri:
        1. Kaynak tipi: cikmis (1.0) > ornek (0.8) > baslangic (0.3)
        2. Zorluk yakinligi
        3. Gorsel tipi eslesmesi
        4. Similarity score
        
        Args:
            results: Mevcut sonuclar
            combination: Hedef kombinasyon
            
        Returns:
            Yeniden siralanmis sonuclar
        """

        def score(q: RetrievedQuestion) -> float:
            s = 0.0

            # Kaynak agirligi (en onemli)
            s += self.KAYNAK_WEIGHTS.get(q.kaynak_tipi, 0.5) * 0.4

            # Zorluk yakinligi (0-1 arasi, 0 = ayni)
            target_zorluk = combination.get("zorluk", 3)
            zorluk_diff = abs(q.zorluk - target_zorluk)
            s += (1 - zorluk_diff / 4) * 0.25

            # Gorsel tipi eslesmesi
            target_gorsel = combination.get("gorsel_tipi", "yok")
            if q.gorsel_tipi == target_gorsel:
                s += 0.2

            # Similarity score
            s += q.similarity_score * 0.15

            return s

        return sorted(results, key=score, reverse=True)

    def retrieve_by_embedding(
        self,
        query_text: str,
        combination: dict,
        top_k: Optional[int] = None,
    ) -> list[RetrievedQuestion]:
        """
        Metin benzerligi ile soru getir
        
        Args:
            query_text: Arama metni
            combination: Filtreleme icin kombinasyon
            top_k: Dondurelecek soru sayisi
            
        Returns:
            RetrievedQuestion listesi
        """
        top_k = top_k or self.config.top_k

        # Temel filtre olustur
        criteria = FilterCriteria(alt_konu=combination.get("alt_konu"))

        # Embedding ile ara
        results = self._search_with_embedding(query_text, criteria, top_k)

        # Yeniden sirala
        return self._rerank(results, combination)


def format_examples_for_prompt(examples: list[RetrievedQuestion]) -> str:
    """
    Ornekleri LLM promptu icin formatla
    
    Args:
        examples: RetrievedQuestion listesi
        
    Returns:
        Formatli string
    """
    lines = []

    for i, ex in enumerate(examples, 1):
        lines.append(f"### Ornek {i}")
        lines.append(f"**Kaynak:** {ex.kaynak_tipi.upper()}")
        lines.append(f"**Zorluk:** {ex.zorluk}/5")
        lines.append(f"**Gorsel:** {ex.gorsel_tipi}")
        lines.append("")
        lines.append(f"**Soru:** {ex.soru_metni}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)

