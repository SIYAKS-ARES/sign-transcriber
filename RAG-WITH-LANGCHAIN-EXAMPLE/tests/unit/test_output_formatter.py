"""
Unit Tests for RAGOutputFormatter
"""

import pytest
import json


class TestRAGOutputFormatter:
    """Test suite for RAGOutputFormatter"""

    def test_init_default_format(self):
        """Default format should be markdown"""
        from services.output_formatter import RAGOutputFormatter
        
        formatter = RAGOutputFormatter()
        
        assert formatter.format_type == "markdown"

    def test_init_custom_format(self):
        """Should accept custom format type"""
        from services.output_formatter import RAGOutputFormatter
        
        formatter = RAGOutputFormatter(format_type="json")
        
        assert formatter.format_type == "json"

    def test_format_examples_markdown(self, single_combination):
        """format_examples should return markdown string"""
        from services.output_formatter import RAGOutputFormatter
        from services.retriever import RetrievedQuestion
        
        formatter = RAGOutputFormatter(format_type="markdown")
        
        examples = [
            RetrievedQuestion(
                soru_metni="Test soru 1",
                alt_konu="ebob_ekok",
                zorluk=4,
                gorsel_tipi="sematik",
                kaynak_tipi="cikmis",
                similarity_score=0.95,
                metadata={}
            )
        ]
        
        result = formatter.format_examples(examples, single_combination)
        
        assert isinstance(result, str)
        assert "ebob_ekok" in result or "EBOB" in result.upper()

    def test_format_examples_json(self, single_combination):
        """format_examples should return valid JSON"""
        from services.output_formatter import RAGOutputFormatter
        from services.retriever import RetrievedQuestion
        
        formatter = RAGOutputFormatter(format_type="json")
        
        examples = [
            RetrievedQuestion(
                soru_metni="Test soru 1",
                alt_konu="ebob_ekok",
                zorluk=4,
                gorsel_tipi="sematik",
                kaynak_tipi="cikmis",
                similarity_score=0.95,
                metadata={}
            )
        ]
        
        result = formatter.format_examples(examples, single_combination)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert "hedef_kombinasyon" in parsed or "ornek_sorular" in parsed

    def test_format_examples_text(self, single_combination):
        """format_examples should return plain text"""
        from services.output_formatter import RAGOutputFormatter
        from services.retriever import RetrievedQuestion
        
        formatter = RAGOutputFormatter(format_type="text")
        
        examples = [
            RetrievedQuestion(
                soru_metni="Test soru 1",
                alt_konu="ebob_ekok",
                zorluk=4,
                gorsel_tipi="sematik",
                kaynak_tipi="cikmis",
                similarity_score=0.95,
                metadata={}
            )
        ]
        
        result = formatter.format_examples(examples, single_combination)
        
        assert isinstance(result, str)
        assert "Test soru 1" in result

    def test_format_examples_includes_combination_info(self, single_combination):
        """Formatted output should include combination info"""
        from services.output_formatter import RAGOutputFormatter
        from services.retriever import RetrievedQuestion
        
        formatter = RAGOutputFormatter(format_type="markdown")
        
        examples = [
            RetrievedQuestion(
                soru_metni="Test",
                alt_konu="ebob_ekok",
                zorluk=4,
                gorsel_tipi="sematik",
                kaynak_tipi="cikmis",
                similarity_score=0.95,
                metadata={}
            )
        ]
        
        result = formatter.format_examples(examples, single_combination)
        
        # Should include combination details
        assert single_combination["alt_konu"] in result

    def test_format_examples_multiple_examples(self, single_combination):
        """Should format multiple examples correctly"""
        from services.output_formatter import RAGOutputFormatter
        from services.retriever import RetrievedQuestion
        
        formatter = RAGOutputFormatter(format_type="markdown")
        
        examples = [
            RetrievedQuestion(
                soru_metni=f"Test soru {i}",
                alt_konu="ebob_ekok",
                zorluk=i+2,
                gorsel_tipi="yok",
                kaynak_tipi="ornek",
                similarity_score=0.9-i*0.1,
                metadata={}
            )
            for i in range(3)
        ]
        
        result = formatter.format_examples(examples, single_combination)
        
        # All examples should be in output
        assert "Test soru 0" in result
        assert "Test soru 1" in result
        assert "Test soru 2" in result

    def test_format_examples_empty_list(self, single_combination):
        """Empty examples list should not crash"""
        from services.output_formatter import RAGOutputFormatter
        
        formatter = RAGOutputFormatter()
        
        result = formatter.format_examples([], single_combination)
        
        assert isinstance(result, str)

    def test_format_examples_turkish_characters(self, single_combination):
        """Should handle Turkish characters correctly"""
        from services.output_formatter import RAGOutputFormatter
        from services.retriever import RetrievedQuestion
        
        formatter = RAGOutputFormatter()
        
        examples = [
            RetrievedQuestion(
                soru_metni="Türkçe karakterler: ğüşıöç ĞÜŞİÖÇ",
                alt_konu="ebob_ekok",
                zorluk=4,
                gorsel_tipi="yok",
                kaynak_tipi="ornek",
                similarity_score=0.95,
                metadata={}
            )
        ]
        
        result = formatter.format_examples(examples, single_combination)
        
        assert "Türkçe" in result or "türkçe" in result.lower()


class TestFormatQuestionOutput:
    """Test format_question_output function"""

    def test_format_question_output(self, mock_generated_question):
        """Should format generated question for display"""
        from services.output_formatter import format_question_output
        
        result = format_question_output(mock_generated_question)
        
        assert isinstance(result, str)
        assert mock_generated_question["hikaye"][:20] in result or "hikaye" in result.lower()

    def test_format_question_output_includes_options(self, mock_generated_question):
        """Should include all options"""
        from services.output_formatter import format_question_output
        
        result = format_question_output(mock_generated_question)
        
        # Should have A, B, C, D
        for key in ["A", "B", "C", "D"]:
            assert key in result

    def test_format_question_output_includes_answer(self, mock_generated_question):
        """Should include correct answer"""
        from services.output_formatter import format_question_output
        
        result = format_question_output(mock_generated_question)
        
        assert mock_generated_question["dogru_cevap"] in result

