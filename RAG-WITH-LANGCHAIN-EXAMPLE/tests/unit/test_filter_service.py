"""
Unit Tests for FilterService
"""

import pytest


class TestFilterCriteria:
    """Test suite for FilterCriteria dataclass"""

    def test_default_values(self):
        """FilterCriteria should have None defaults"""
        from services.filter_service import FilterCriteria
        
        criteria = FilterCriteria()
        
        assert criteria.alt_konu is None
        assert criteria.zorluk is None
        assert criteria.zorluk_range is None
        assert criteria.gorsel_tipi is None
        assert criteria.kaynak_tipi is None
        assert criteria.is_lgs is None

    def test_with_all_values(self):
        """FilterCriteria should accept all values"""
        from services.filter_service import FilterCriteria
        
        criteria = FilterCriteria(
            alt_konu="ebob_ekok",
            zorluk=4,
            zorluk_range=(3, 5),
            gorsel_tipi="sematik",
            kaynak_tipi=["cikmis", "ornek"],
            is_lgs=1
        )
        
        assert criteria.alt_konu == "ebob_ekok"
        assert criteria.zorluk == 4
        assert criteria.zorluk_range == (3, 5)
        assert criteria.gorsel_tipi == "sematik"
        assert criteria.kaynak_tipi == ["cikmis", "ornek"]
        assert criteria.is_lgs == 1


class TestFilterService:
    """Test suite for FilterService"""

    def test_init_with_metadata(self, sample_question_metadata):
        """FilterService should initialize with metadata list"""
        from services.filter_service import FilterService
        
        service = FilterService(sample_question_metadata)
        
        assert len(service.all_metadata) == len(sample_question_metadata)

    def test_filter_by_alt_konu(self, sample_question_metadata):
        """Filter by alt_konu should return matching items"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(alt_konu="ebob_ekok")
        
        results = service.filter(criteria)
        
        assert len(results) > 0
        for item in results:
            assert item["Alt_Konu"] == "ebob_ekok"

    def test_filter_by_zorluk(self, sample_question_metadata):
        """Filter by zorluk should return matching items"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(zorluk=2)
        
        results = service.filter(criteria)
        
        assert len(results) > 0
        for item in results:
            assert item["Zorluk"] == 2

    def test_filter_by_zorluk_range(self, sample_question_metadata):
        """Filter by zorluk_range should return items within range"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(zorluk_range=(2, 4))
        
        results = service.filter(criteria)
        
        assert len(results) > 0
        for item in results:
            assert 2 <= item["Zorluk"] <= 4

    def test_filter_by_gorsel_tipi(self, sample_question_metadata):
        """Filter by gorsel_tipi should return matching items"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(gorsel_tipi="yok")
        
        results = service.filter(criteria)
        
        for item in results:
            assert item["Gorsel_Tipi"] == "yok"

    def test_filter_by_kaynak_tipi(self, sample_question_metadata):
        """Filter by kaynak_tipi list should return matching items"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(kaynak_tipi=["cikmis", "ornek"])
        
        results = service.filter(criteria)
        
        for item in results:
            assert item["Kaynak_Tipi"] in ["cikmis", "ornek"]

    def test_filter_by_is_lgs(self, sample_question_metadata):
        """Filter by is_lgs should return matching items"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(is_lgs=1)
        
        results = service.filter(criteria)
        
        for item in results:
            assert item["is_LGS"] == 1

    def test_filter_combined_criteria(self, sample_question_metadata):
        """Combined filter criteria should AND together"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(
            alt_konu="ebob_ekok",
            is_lgs=1
        )
        
        results = service.filter(criteria)
        
        for item in results:
            assert item["Alt_Konu"] == "ebob_ekok"
            assert item["is_LGS"] == 1

    def test_filter_no_match_returns_empty(self, sample_question_metadata):
        """Filter with no matches should return empty list"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(alt_konu="nonexistent_topic")
        
        results = service.filter(criteria)
        
        assert len(results) == 0

    def test_filter_empty_criteria_returns_all(self, sample_question_metadata):
        """Empty filter criteria should return all items"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria()
        
        results = service.filter(criteria)
        
        assert len(results) == len(sample_question_metadata)

    def test_filter_indices(self, sample_question_metadata):
        """filter_indices should return list of matching indices"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        criteria = FilterCriteria(alt_konu="ebob_ekok")
        
        indices = service.filter_indices(criteria)
        
        assert isinstance(indices, list)
        for idx in indices:
            assert sample_question_metadata[idx]["Alt_Konu"] == "ebob_ekok"


class TestFilterServiceEdgeCases:
    """Edge case tests for FilterService"""

    def test_empty_metadata_list(self):
        """Empty metadata list should handle gracefully"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService([])
        criteria = FilterCriteria(alt_konu="ebob_ekok")
        
        results = service.filter(criteria)
        
        assert results == []

    def test_missing_field_in_metadata(self):
        """Missing field in metadata should not crash"""
        from services.filter_service import FilterService, FilterCriteria
        
        incomplete_metadata = [
            {"Alt_Konu": "ebob_ekok"},  # Missing other fields
        ]
        
        service = FilterService(incomplete_metadata)
        criteria = FilterCriteria(alt_konu="ebob_ekok")
        
        # Should not raise error
        results = service.filter(criteria)
        assert len(results) == 1

    def test_zorluk_range_boundary(self, sample_question_metadata):
        """Zorluk range should be inclusive"""
        from services.filter_service import FilterService, FilterCriteria
        
        service = FilterService(sample_question_metadata)
        
        # Range exactly matching a value
        criteria = FilterCriteria(zorluk_range=(4, 4))
        results = service.filter(criteria)
        
        for item in results:
            assert item["Zorluk"] == 4

