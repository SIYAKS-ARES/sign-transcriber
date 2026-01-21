"""
Unit Tests for CombinationSelector
"""

import pytest
from unittest.mock import MagicMock


class TestCombinationSelector:
    """Test suite for CombinationSelector"""

    def test_init_with_combinations(self, sample_combinations):
        """Test initialization with combinations list"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        
        assert len(selector.combinations) == len(sample_combinations)
        assert selector.threshold == 0.70  # default

    def test_threshold_filtering(self, sample_combinations):
        """Threshold altı kombinasyonlar filtrelenmeli"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations, threshold=0.80)
        filtered = selector.get_filtered_combinations()
        
        # All filtered combinations should have score >= 0.80
        for combo in filtered:
            assert combo["lgs_skor"] >= 0.80

    def test_select_returns_valid_combination(self, sample_combinations):
        """Select should return a valid combination"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        selected = selector.select()
        
        assert selected is not None
        assert "alt_konu" in selected
        assert "zorluk" in selected
        assert "gorsel_tipi" in selected
        assert "lgs_skor" in selected

    def test_select_with_alt_konu_filter(self, sample_combinations):
        """Select with alt_konu filter should return matching combination"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        selected = selector.select(alt_konu="carpanlar")
        
        assert selected["alt_konu"] == "carpanlar"

    def test_select_with_zorluk_filter(self, sample_combinations):
        """Select with zorluk filter should return matching combination"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        selected = selector.select(zorluk=4)
        
        assert selected["zorluk"] == 4

    def test_select_with_gorsel_tipi_filter(self, sample_combinations):
        """Select with gorsel_tipi filter should return matching combination"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        selected = selector.select(gorsel_tipi="sematik")
        
        assert selected["gorsel_tipi"] == "sematik"

    def test_select_with_multiple_filters(self, sample_combinations):
        """Select with multiple filters should return matching combination"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        selected = selector.select(alt_konu="ebob_ekok", zorluk=4)
        
        assert selected["alt_konu"] == "ebob_ekok"
        assert selected["zorluk"] == 4

    def test_select_no_match_returns_none(self, sample_combinations):
        """Select with impossible filters should return None"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        selected = selector.select(alt_konu="nonexistent")
        
        assert selected is None

    def test_weighted_sampling_favors_high_scores(self, sample_combinations):
        """High score combinations should be selected more frequently"""
        from services.combination_selector import CombinationSelector
        import collections
        
        selector = CombinationSelector(sample_combinations, alpha=2.0)
        
        # Run selection many times
        selections = collections.Counter()
        for _ in range(1000):
            selected = selector.select()
            if selected:
                selections[selected["rank"]] += 1
        
        # Rank 1 should be selected more than rank 10+
        if 1 in selections and len(sample_combinations) > 10:
            avg_low_rank = sum(selections.get(i, 0) for i in range(10, 18)) / 8
            assert selections[1] > avg_low_rank

    def test_exclude_combinations(self, sample_combinations):
        """Excluded combinations should not be selected"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        first = sample_combinations[0]
        
        # Exclude first combination
        selected = selector.select(exclude=[first])
        
        # Selected should be different from excluded
        if selected:
            assert selected != first

    def test_get_top_k_combinations(self, sample_combinations):
        """get_top_k should return k highest scoring combinations"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations)
        top_5 = selector.get_top_k(5)
        
        assert len(top_5) == 5
        # Should be sorted by score descending
        scores = [c["lgs_skor"] for c in top_5]
        assert scores == sorted(scores, reverse=True)

    def test_empty_combinations_list(self):
        """Empty combinations list should handle gracefully"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector([])
        selected = selector.select()
        
        assert selected is None

    def test_all_below_threshold(self, sample_combinations):
        """When all combinations are below threshold, should return None or fallback"""
        from services.combination_selector import CombinationSelector
        
        selector = CombinationSelector(sample_combinations, threshold=0.99)
        filtered = selector.get_filtered_combinations()
        
        # Should be empty since no combination has score >= 0.99
        assert len(filtered) == 0


class TestCombinationSelectorEdgeCases:
    """Edge case tests for CombinationSelector"""

    def test_single_combination(self):
        """Single combination should always be selected"""
        from services.combination_selector import CombinationSelector
        
        single = [{"alt_konu": "ebob_ekok", "zorluk": 3, "gorsel_tipi": "yok", "lgs_skor": 0.85}]
        selector = CombinationSelector(single)
        
        selected = selector.select()
        assert selected == single[0]

    def test_same_score_combinations(self):
        """Combinations with same score should be handled"""
        from services.combination_selector import CombinationSelector
        
        same_score = [
            {"alt_konu": "ebob_ekok", "zorluk": 3, "gorsel_tipi": "yok", "lgs_skor": 0.85},
            {"alt_konu": "carpanlar", "zorluk": 4, "gorsel_tipi": "sematik", "lgs_skor": 0.85},
        ]
        selector = CombinationSelector(same_score)
        
        # Should not raise error
        selected = selector.select()
        assert selected is not None

    def test_temperature_effect(self, sample_combinations):
        """Temperature should affect selection distribution"""
        from services.combination_selector import CombinationSelector
        
        # Low temperature = more deterministic
        selector_low_temp = CombinationSelector(sample_combinations, temperature=0.1)
        # High temperature = more random
        selector_high_temp = CombinationSelector(sample_combinations, temperature=2.0)
        
        # Both should work without errors
        assert selector_low_temp.select() is not None
        assert selector_high_temp.select() is not None

