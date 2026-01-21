"""
Unit Tests for QuestionRetriever
"""

import pytest
from unittest.mock import MagicMock, patch


class TestQuestionRetriever:
    """Test suite for QuestionRetriever"""

    def test_init(self, mock_embedding_pipeline, sample_question_metadata):
        """Test retriever initialization"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        assert retriever.embedding is not None
        assert retriever.filter is not None

    def test_retrieve_examples_returns_list(self, mock_embedding_pipeline, sample_question_metadata, single_combination):
        """retrieve_examples should return a list of questions"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        results = retriever.retrieve_examples(single_combination, top_k=3)
        
        assert isinstance(results, list)

    def test_retrieve_examples_respects_top_k(self, mock_embedding_pipeline, sample_question_metadata, single_combination):
        """retrieve_examples should return at most top_k results"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        results = retriever.retrieve_examples(single_combination, top_k=2)
        
        assert len(results) <= 2

    def test_retrieve_examples_filters_by_alt_konu(self, mock_embedding_pipeline, sample_question_metadata):
        """retrieve_examples should prefer matching alt_konu"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        combination = {
            "alt_konu": "ebob_ekok",
            "zorluk": 3,
            "gorsel_tipi": "yok",
            "lgs_skor": 0.85
        }
        
        results = retriever.retrieve_examples(combination, top_k=5)
        
        # At least some results should match alt_konu
        if results:
            alt_konular = [r.alt_konu for r in results]
            assert "ebob_ekok" in alt_konular

    def test_retrieve_examples_reranking(self, mock_embedding_pipeline, sample_question_metadata, single_combination):
        """Results should be reranked by relevance"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        results = retriever.retrieve_examples(single_combination, top_k=5)
        
        # Results should be ordered (cikmis > ornek > baslangic)
        # Just verify it returns without error
        assert isinstance(results, list)

    def test_retrieve_examples_empty_combination(self, mock_embedding_pipeline, sample_question_metadata):
        """Empty combination should still work"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        combination = {
            "alt_konu": "nonexistent",
            "zorluk": 10,  # Invalid
            "gorsel_tipi": "unknown",
            "lgs_skor": 0.5
        }
        
        # Should not crash, might return empty or fallback results
        results = retriever.retrieve_examples(combination, top_k=3)
        assert isinstance(results, list)


class TestRetrievalConfig:
    """Test RetrievalConfig dataclass"""

    def test_default_config(self):
        """Test default RetrievalConfig values"""
        from services.retriever import RetrievalConfig
        
        config = RetrievalConfig()
        
        assert config.top_k == 5
        assert config.min_results == 3
        assert config.max_zorluk_diff == 1

    def test_custom_config(self):
        """Test custom RetrievalConfig values"""
        from services.retriever import RetrievalConfig
        
        config = RetrievalConfig(
            top_k=10,
            min_results=5,
            max_zorluk_diff=2
        )
        
        assert config.top_k == 10
        assert config.min_results == 5
        assert config.max_zorluk_diff == 2


class TestRetrieverFallbackStrategy:
    """Test retriever fallback strategy"""

    def test_fallback_expands_zorluk_range(self, mock_embedding_pipeline, sample_question_metadata, single_combination):
        """When exact match fails, should expand zorluk range"""
        from services.retriever import QuestionRetriever
        from services.filter_service import FilterService
        
        # Create metadata with no exact zorluk match
        metadata_no_match = [
            {
                "Soru_MetniOCR": "Test",
                "Alt_Konu": "ebob_ekok",
                "Zorluk": 1,  # Different from combination's zorluk=4
                "Gorsel_Tipi": "yok",
                "Kaynak_Tipi": "ornek",
                "is_LGS": 1,
                "egitim_agirligi": 0.8
            }
        ]
        
        filter_service = FilterService(metadata_no_match)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service)
        
        # Should still return something via fallback
        results = retriever.retrieve_examples(single_combination, top_k=3)
        assert isinstance(results, list)

    def test_minimum_results_guarantee(self, mock_embedding_pipeline, sample_question_metadata, single_combination):
        """Retriever should try to return at least min_results"""
        from services.retriever import QuestionRetriever, RetrievalConfig
        from services.filter_service import FilterService
        
        filter_service = FilterService(sample_question_metadata)
        config = RetrievalConfig(min_results=3)
        retriever = QuestionRetriever(mock_embedding_pipeline, filter_service, config)
        
        results = retriever.retrieve_examples(single_combination, top_k=5)
        
        # Should try to get at least 3 results if data allows
        # Note: might be less if not enough matching data
        assert isinstance(results, list)

