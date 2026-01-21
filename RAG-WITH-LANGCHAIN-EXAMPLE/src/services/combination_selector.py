"""
Kombinasyon Secici
Weighted sampling ile kombinasyon secimi
"""

import random
from typing import Optional
import numpy as np

from ..models.combination import Combination, SelectionPolicy, CombinationFilters
from ..models.config_schema import ConfigSchema


class CombinationSelector:
    """
    Model configs.json'dan kombinasyon secici
    
    Strateji (rag.md sozlesmesine gore):
    1. Threshold ile filtrele
    2. Weighted sampling: agirlik = lgs_skor^alpha
    3. Temperature ile dagilimi yumusat
    """

    def __init__(
        self,
        config: ConfigSchema,
        policy: Optional[SelectionPolicy] = None,
    ):
        """
        Args:
            config: Model configs.json
            policy: Secim politikasi (None ise config'den alinir)
        """
        self.config = config
        self.policy = policy or config.selection_policy or SelectionPolicy()
        self.combinations = config.combinations

    def select(
        self,
        filters: Optional[CombinationFilters] = None,
        exclude: Optional[list[dict]] = None,
    ) -> Combination:
        """
        Weighted sampling ile kombinasyon sec
        
        Args:
            filters: alt_konu, zorluk, gorsel_tipi filtreleri
            exclude: Haric tutulacak kombinasyonlar (cesitlilik icin)
            
        Returns:
            Secilen Combination
        """
        # Filtreleme
        candidates = self._apply_filters(self.combinations, filters)

        # Haric tutma
        if exclude:
            exclude_set = {self._combo_key(e) for e in exclude}
            candidates = [c for c in candidates if self._combo_key(c.model_dump()) not in exclude_set]

        if not candidates:
            raise ValueError("Filtreleme sonrasi kombinasyon kalmadi")

        # Secim moduna gore sec
        if self.policy.mode == "weighted_sampling":
            return self._weighted_sample(candidates)
        elif self.policy.mode == "top_k":
            return self._top_k_sample(candidates)
        else:  # random
            return random.choice(candidates)

    def select_multiple(
        self,
        count: int,
        filters: Optional[CombinationFilters] = None,
        ensure_diversity: bool = True,
    ) -> list[Combination]:
        """
        Birden fazla kombinasyon sec
        
        Args:
            count: Secilecek sayi
            filters: Filtreler
            ensure_diversity: Ayni kombinasyonu tekrar secme
            
        Returns:
            Combination listesi
        """
        results = []
        exclude = []

        for _ in range(count):
            try:
                combo = self.select(
                    filters=filters,
                    exclude=exclude if ensure_diversity else None,
                )
                results.append(combo)
                if ensure_diversity:
                    exclude.append(combo.model_dump())
            except ValueError:
                # Kombinasyon kalmadi
                break

        return results

    def _apply_filters(
        self,
        combinations: list[Combination],
        filters: Optional[CombinationFilters],
    ) -> list[Combination]:
        """Filtreleri uygula"""
        if filters is None:
            return combinations

        results = combinations

        if filters.alt_konu:
            results = [c for c in results if c.alt_konu == filters.alt_konu]

        if filters.zorluk:
            results = [c for c in results if c.zorluk == filters.zorluk]

        if filters.zorluk_min is not None:
            results = [c for c in results if c.zorluk >= filters.zorluk_min]

        if filters.zorluk_max is not None:
            results = [c for c in results if c.zorluk <= filters.zorluk_max]

        if filters.gorsel_tipi:
            results = [c for c in results if c.gorsel_tipi == filters.gorsel_tipi]

        if filters.min_lgs_skor is not None:
            results = [c for c in results if c.lgs_skor >= filters.min_lgs_skor]

        return results

    def _weighted_sample(self, candidates: list[Combination]) -> Combination:
        """
        Weighted sampling ile secim
        
        Agirlik = lgs_skor^alpha
        Softmax with temperature
        """
        # Agirliklari hesapla
        weights = np.array([c.lgs_skor ** self.policy.alpha for c in candidates])

        # Softmax with temperature
        if self.policy.temperature != 1.0:
            weights = np.exp(np.log(weights + 1e-10) / self.policy.temperature)

        # Normalize
        weights = weights / weights.sum()

        # Sec
        idx = np.random.choice(len(candidates), p=weights)
        return candidates[idx]

    def _top_k_sample(self, candidates: list[Combination]) -> Combination:
        """Top-K icerisinden rastgele sec"""
        # Skorlara gore sirala
        sorted_candidates = sorted(candidates, key=lambda c: c.lgs_skor, reverse=True)

        # Top-K al
        top_k = sorted_candidates[: self.policy.top_k]

        # Rastgele sec
        return random.choice(top_k)

    def _combo_key(self, combo: dict) -> str:
        """Kombinasyon icin unique key"""
        return f"{combo.get('alt_konu')}_{combo.get('zorluk')}_{combo.get('gorsel_tipi')}"

    def get_statistics(self) -> dict:
        """Kombinasyon istatistikleri"""
        stats = {
            "total": len(self.combinations),
            "by_alt_konu": {},
            "by_zorluk": {},
            "by_gorsel_tipi": {},
            "score_distribution": {
                "min": min(c.lgs_skor for c in self.combinations),
                "max": max(c.lgs_skor for c in self.combinations),
                "mean": sum(c.lgs_skor for c in self.combinations) / len(self.combinations),
            },
        }

        for combo in self.combinations:
            # Alt konu
            ak = combo.alt_konu
            stats["by_alt_konu"][ak] = stats["by_alt_konu"].get(ak, 0) + 1

            # Zorluk
            z = combo.zorluk
            stats["by_zorluk"][z] = stats["by_zorluk"].get(z, 0) + 1

            # Gorsel tipi
            gt = combo.gorsel_tipi
            stats["by_gorsel_tipi"][gt] = stats["by_gorsel_tipi"].get(gt, 0) + 1

        return stats

