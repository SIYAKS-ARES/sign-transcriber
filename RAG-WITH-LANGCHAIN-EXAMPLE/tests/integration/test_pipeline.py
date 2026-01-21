"""
Integration Tests for RAG Pipeline
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for the full RAG pipeline"""

    @pytest.fixture
    def mock_services(self, sample_question_metadata, sample_combinations, mock_llm_client):
        """Create mock services for integration testing"""
        from services.embedding_service import EmbeddingPipeline
        from services.filter_service import FilterService
        from services.retriever import QuestionRetriever
        from services.output_formatter import RAGOutputFormatter
        from services.prompt_builder import PromptBuilder
        from services.combination_selector import CombinationSelector
        
        # Mock embedding pipeline
        embedding = MagicMock(spec=EmbeddingPipeline)
        embedding.dimension = 768
        embedding.search.return_value = [
            (0.95, meta) for meta in sample_question_metadata[:3]
        ]
        
        filter_service = FilterService(sample_question_metadata)
        retriever = QuestionRetriever(embedding, filter_service)
        formatter = RAGOutputFormatter()
        prompt_builder = PromptBuilder()
        selector = CombinationSelector(sample_combinations)
        
        return {
            "embedding": embedding,
            "filter_service": filter_service,
            "retriever": retriever,
            "formatter": formatter,
            "prompt_builder": prompt_builder,
            "selector": selector,
            "llm_client": mock_llm_client
        }

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, mock_services, single_combination):
        """Test the complete pipeline from combination to question"""
        from services.question_generator import QuestionGenerator
        
        generator = QuestionGenerator(
            retriever=mock_services["retriever"],
            formatter=mock_services["formatter"],
            prompt_builder=mock_services["prompt_builder"],
            llm_client=mock_services["llm_client"]
        )
        
        # Generate question
        question = await generator.generate_question(single_combination)
        
        # Verify result structure
        assert question is not None
        assert hasattr(question, "hikaye")
        assert hasattr(question, "soru")
        assert hasattr(question, "secenekler")
        assert hasattr(question, "dogru_cevap")
        assert hasattr(question, "cozum")

    @pytest.mark.asyncio
    async def test_pipeline_with_style_instruction(self, mock_services, single_combination):
        """Test pipeline with style instruction"""
        from services.question_generator import QuestionGenerator
        
        generator = QuestionGenerator(
            retriever=mock_services["retriever"],
            formatter=mock_services["formatter"],
            prompt_builder=mock_services["prompt_builder"],
            llm_client=mock_services["llm_client"]
        )
        
        question = await generator.generate_question(
            single_combination,
            style_instruction="Günlük hayattan örnek kullan."
        )
        
        assert question is not None
        assert question.metadata.get("style_instruction") == "Günlük hayattan örnek kullan."

    @pytest.mark.asyncio
    async def test_pipeline_retry_on_invalid_response(self, mock_services, single_combination):
        """Test pipeline retries on invalid LLM response"""
        from services.question_generator import QuestionGenerator
        
        # First call returns invalid, second returns valid
        mock_services["llm_client"].generate.side_effect = [
            "invalid json",
            json.dumps({
                "hikaye": "Valid",
                "soru": "Valid?",
                "secenekler": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "dogru_cevap": "A",
                "cozum": ["Step 1"]
            })
        ]
        
        generator = QuestionGenerator(
            retriever=mock_services["retriever"],
            formatter=mock_services["formatter"],
            prompt_builder=mock_services["prompt_builder"],
            llm_client=mock_services["llm_client"]
        )
        
        question = await generator.generate_question(single_combination, max_attempts=2)
        
        assert question is not None
        # Should have tried twice
        assert mock_services["llm_client"].generate.call_count == 2

    def test_combination_selection_to_retrieval(self, mock_services):
        """Test combination selection flows to retrieval correctly"""
        selector = mock_services["selector"]
        retriever = mock_services["retriever"]
        
        # Select a combination
        combination = selector.select()
        
        # Retrieve examples for it
        examples = retriever.retrieve_examples(combination, top_k=3)
        
        assert combination is not None
        assert isinstance(examples, list)

    def test_retrieval_to_formatting(self, mock_services, single_combination):
        """Test retrieval results are properly formatted"""
        retriever = mock_services["retriever"]
        formatter = mock_services["formatter"]
        
        # Get examples
        examples = retriever.retrieve_examples(single_combination, top_k=3)
        
        # Format them
        formatted = formatter.format_examples(examples, single_combination)
        
        assert formatted is not None
        assert len(formatted) > 0

    def test_formatting_to_prompt(self, mock_services, single_combination):
        """Test formatted examples flow into prompt correctly"""
        retriever = mock_services["retriever"]
        formatter = mock_services["formatter"]
        prompt_builder = mock_services["prompt_builder"]
        
        # Get and format examples
        examples = retriever.retrieve_examples(single_combination, top_k=3)
        formatted = formatter.format_examples(examples, single_combination)
        
        # Build prompt
        prompt = prompt_builder.build_user_prompt(single_combination, formatted)
        
        assert prompt is not None
        assert single_combination["alt_konu"] in prompt


@pytest.mark.integration
class TestDiversityIntegration:
    """Integration tests for diversity features"""

    @pytest.mark.asyncio
    async def test_diverse_batch_generation(self, mock_services, single_combination):
        """Test batch generation produces diverse questions"""
        from services.diversity_service import DiverseQuestionGenerator, DiversityService
        from services.question_generator import QuestionGenerator
        
        # Create generator
        generator = QuestionGenerator(
            retriever=mock_services["retriever"],
            formatter=mock_services["formatter"],
            prompt_builder=mock_services["prompt_builder"],
            llm_client=mock_services["llm_client"]
        )
        
        diversity = DiversityService(mock_services["prompt_builder"])
        diverse_gen = DiverseQuestionGenerator(generator, diversity)
        
        # Generate batch
        questions = await diverse_gen.generate_diverse_batch(
            single_combination,
            count=3
        )
        
        assert len(questions) == 3

    def test_style_variation_selection(self, mock_services):
        """Test style variations are properly selected"""
        from services.diversity_service import DiversityService
        
        diversity = DiversityService(mock_services["prompt_builder"])
        
        # Get multiple styles
        styles = [diversity.get_random_style() for _ in range(5)]
        
        # Should not all be the same
        assert len(set(styles)) > 1

    def test_duplicate_detection(self, mock_services):
        """Test duplicate detection works"""
        from services.diversity_service import DiversityService
        
        diversity = DiversityService(mock_services["prompt_builder"])
        
        question1 = "24 ve 36 sayılarının EBOB'u nedir?"
        
        # Add to history
        diversity.add_to_history(question1)
        
        # Same question should be detected as duplicate
        assert diversity.is_duplicate(question1, threshold=0.9)
        
        # Different question should not
        question2 = "Bir fabrikada üretilen ürünler nasıl paketlenir?"
        assert not diversity.is_duplicate(question2, threshold=0.9)


@pytest.mark.integration
class TestConfigLoaderIntegration:
    """Integration tests for config loading"""

    def test_load_configs_json(self, sample_configs_path):
        """Test loading configs.json file"""
        from services.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        configs = loader.load_configs(str(sample_configs_path))
        
        assert configs is not None
        assert "combinations" in configs
        assert len(configs["combinations"]) > 0

    def test_load_questions_csv(self, sample_questions_path):
        """Test loading questions CSV file"""
        from services.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        questions = loader.load_questions(str(sample_questions_path))
        
        assert questions is not None
        assert len(questions) > 0

    def test_config_to_selector_flow(self, sample_configs_path):
        """Test configs flow into selector correctly"""
        from services.config_loader import ConfigLoader
        from services.combination_selector import CombinationSelector
        
        loader = ConfigLoader()
        configs = loader.load_configs(str(sample_configs_path))
        
        selector = CombinationSelector(
            configs["combinations"],
            threshold=configs.get("threshold", 0.70)
        )
        
        combination = selector.select()
        
        assert combination is not None
        assert combination["lgs_skor"] >= configs.get("threshold", 0.70)

