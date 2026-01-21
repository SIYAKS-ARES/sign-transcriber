"""
R-02: Filtreleme Sistemi
Alt konu, zorluk ve gorsel tipine gore filtreleme
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class FilterCriteria:
    """
    Soru filtreleme kriterleri
    """

    alt_konu: Optional[str] = None
    zorluk: Optional[int] = None
    zorluk_range: Optional[tuple[int, int]] = None  # (min, max)
    gorsel_tipi: Optional[str] = None
    kaynak_tipi: Optional[list[str]] = None  # ["cikmis", "ornek"]
    is_lgs: Optional[int] = None

    def is_empty(self) -> bool:
        """Tum kriterler bos mu?"""
        return all(
            v is None
            for v in [
                self.alt_konu,
                self.zorluk,
                self.zorluk_range,
                self.gorsel_tipi,
                self.kaynak_tipi,
                self.is_lgs,
            ]
        )


class FilterService:
    """
    Soru metadata filtreleme servisi
    
    Kullanim:
        filter_service = FilterService(metadata_list)
        filtered = filter_service.filter(FilterCriteria(alt_konu="ebob_ekok"))
    """

    def __init__(self, metadata_list: list[dict]):
        """
        Args:
            metadata_list: Soru metadata listesi
        """
        self.all_metadata = metadata_list

    def filter(self, criteria: FilterCriteria) -> list[dict]:
        """
        Kriterlere gore metadata'lari filtrele
        
        Args:
            criteria: Filtreleme kriterleri
            
        Returns:
            Filtrelenmis metadata listesi
        """
        if criteria.is_empty():
            return self.all_metadata.copy()

        results = self.all_metadata.copy()

        # Alt konu filtresi
        if criteria.alt_konu:
            results = [m for m in results if m.get("Alt_Konu") == criteria.alt_konu]

        # Zorluk filtresi (tam eslesme)
        if criteria.zorluk:
            results = [m for m in results if m.get("Zorluk") == criteria.zorluk]

        # Zorluk aralik filtresi
        if criteria.zorluk_range:
            min_z, max_z = criteria.zorluk_range
            results = [
                m for m in results if min_z <= m.get("Zorluk", 0) <= max_z
            ]

        # Gorsel tipi filtresi
        if criteria.gorsel_tipi:
            results = [m for m in results if m.get("Gorsel_Tipi") == criteria.gorsel_tipi]

        # Kaynak tipi filtresi
        if criteria.kaynak_tipi:
            results = [m for m in results if m.get("Kaynak_Tipi") in criteria.kaynak_tipi]

        # is_LGS filtresi
        if criteria.is_lgs is not None:
            results = [m for m in results if m.get("is_LGS") == criteria.is_lgs]

        return results

    def filter_indices(self, criteria: FilterCriteria) -> list[int]:
        """
        Filtrelenmis metadata'larin index'lerini don
        FAISS ile kullanmak icin
        
        Args:
            criteria: Filtreleme kriterleri
            
        Returns:
            Gecerli FAISS index listesi (sequential 0-based indices)
        """
        indices = []
        for i, meta in enumerate(self.all_metadata):
            if self._matches(meta, criteria):
                # Use sequential position index (i), not original CSV row index
                # This matches the FAISS index keys in id_to_metadata
                indices.append(i)
        return indices

    def _matches(self, meta: dict, criteria: FilterCriteria) -> bool:
        """
        Tek metadata'nin kriterlere uyup uymadigini kontrol et
        """
        if criteria.alt_konu and meta.get("Alt_Konu") != criteria.alt_konu:
            return False

        if criteria.zorluk and meta.get("Zorluk") != criteria.zorluk:
            return False

        if criteria.zorluk_range:
            z = meta.get("Zorluk", 0)
            if not (criteria.zorluk_range[0] <= z <= criteria.zorluk_range[1]):
                return False

        if criteria.gorsel_tipi and meta.get("Gorsel_Tipi") != criteria.gorsel_tipi:
            return False

        if criteria.kaynak_tipi and meta.get("Kaynak_Tipi") not in criteria.kaynak_tipi:
            return False

        if criteria.is_lgs is not None and meta.get("is_LGS") != criteria.is_lgs:
            return False

        return True

    def get_statistics(self) -> dict:
        """
        Metadata istatistiklerini don
        """
        stats = {
            "total": len(self.all_metadata),
            "alt_konu": {},
            "zorluk": {},
            "gorsel_tipi": {},
            "kaynak_tipi": {},
            "is_lgs": {0: 0, 1: 0},
        }

        for meta in self.all_metadata:
            # Alt konu
            ak = meta.get("Alt_Konu", "bilinmiyor")
            stats["alt_konu"][ak] = stats["alt_konu"].get(ak, 0) + 1

            # Zorluk
            z = meta.get("Zorluk", 0)
            stats["zorluk"][z] = stats["zorluk"].get(z, 0) + 1

            # Gorsel tipi
            gt = meta.get("Gorsel_Tipi", "bilinmiyor")
            stats["gorsel_tipi"][gt] = stats["gorsel_tipi"].get(gt, 0) + 1

            # Kaynak tipi
            kt = meta.get("Kaynak_Tipi", "bilinmiyor")
            stats["kaynak_tipi"][kt] = stats["kaynak_tipi"].get(kt, 0) + 1

            # is_LGS
            is_lgs = meta.get("is_LGS", 0)
            stats["is_lgs"][is_lgs] = stats["is_lgs"].get(is_lgs, 0) + 1

        return stats


def create_retrieval_filter(combination: dict) -> FilterCriteria:
    """
    Kombinasyona gore optimal retrieval filtresi olustur
    
    Strateji (rag.md sozlesmesine gore):
    1. Ayni alt_konu (zorunlu)
    2. Zorluk: hedef +-1 (esneklik)
    3. Gorsel tipi: tercih edilir ama zorunlu degil
    4. Kaynak: cikmis > ornek > baslangic (siralama icin)
    
    Args:
        combination: {alt_konu, zorluk, gorsel_tipi, lgs_skor}
        
    Returns:
        FilterCriteria instance
    """
    zorluk = combination.get("zorluk", 3)

    return FilterCriteria(
        alt_konu=combination.get("alt_konu"),
        zorluk_range=(
            max(1, zorluk - 1),
            min(5, zorluk + 1),
        ),
        gorsel_tipi=None,  # Esneklik icin None
        kaynak_tipi=["cikmis", "ornek"],  # LGS profili oncelikli
        is_lgs=1,
    )


def create_fallback_filter(combination: dict, level: int = 1) -> FilterCriteria:
    """
    Fallback filtreleri olustur
    
    Args:
        combination: Hedef kombinasyon
        level: Fallback seviyesi (1-4)
        
    Returns:
        FilterCriteria instance
    """
    if level == 1:
        # Seviye 1: Tam eslesme + LGS
        return FilterCriteria(
            alt_konu=combination.get("alt_konu"),
            zorluk=combination.get("zorluk"),
            gorsel_tipi=combination.get("gorsel_tipi"),
            is_lgs=1,
        )
    elif level == 2:
        # Seviye 2: Zorluk +-1
        zorluk = combination.get("zorluk", 3)
        return FilterCriteria(
            alt_konu=combination.get("alt_konu"),
            zorluk_range=(max(1, zorluk - 1), min(5, zorluk + 1)),
            gorsel_tipi=combination.get("gorsel_tipi"),
            is_lgs=1,
        )
    elif level == 3:
        # Seviye 3: Sadece alt_konu + LGS
        return FilterCriteria(
            alt_konu=combination.get("alt_konu"),
            is_lgs=1,
        )
    else:
        # Seviye 4: Sadece alt_konu (baslangic dahil)
        return FilterCriteria(
            alt_konu=combination.get("alt_konu"),
        )

