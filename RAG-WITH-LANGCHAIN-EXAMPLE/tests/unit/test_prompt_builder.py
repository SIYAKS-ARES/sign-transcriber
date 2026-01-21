"""
Unit Tests for PromptBuilder
"""

import pytest


class TestPromptBuilder:
    """Test suite for PromptBuilder"""

    def test_init(self):
        """Test PromptBuilder initialization"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        
        assert builder.system_prompt is not None
        assert len(builder.system_prompt) > 0

    def test_system_prompt_contains_lgs_context(self):
        """System prompt should contain LGS context"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        
        assert "LGS" in builder.system_prompt
        assert "matematik" in builder.system_prompt.lower()

    def test_system_prompt_contains_topics(self):
        """System prompt should mention topic areas"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        
        # Should mention at least one topic
        topics = ["carpanlar", "ebob", "ekok", "asal"]
        assert any(topic in builder.system_prompt.lower() for topic in topics)

    def test_system_prompt_contains_format_rules(self):
        """System prompt should contain format rules"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        
        # Should mention 4 options
        assert "4" in builder.system_prompt or "dört" in builder.system_prompt.lower()

    def test_build_user_prompt_basic(self, single_combination):
        """build_user_prompt should create valid prompt"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        formatted_examples = "## Örnek 1\nTest soru..."
        
        prompt = builder.build_user_prompt(single_combination, formatted_examples)
        
        assert prompt is not None
        assert len(prompt) > 0

    def test_build_user_prompt_includes_combination(self, single_combination):
        """User prompt should include combination details"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        formatted_examples = "## Örnek 1\nTest soru..."
        
        prompt = builder.build_user_prompt(single_combination, formatted_examples)
        
        assert single_combination["alt_konu"] in prompt
        assert str(single_combination["zorluk"]) in prompt

    def test_build_user_prompt_includes_examples(self, single_combination):
        """User prompt should include formatted examples"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        formatted_examples = "## Örnek Soru 1\nBu bir test sorusudur."
        
        prompt = builder.build_user_prompt(single_combination, formatted_examples)
        
        assert "test sorusu" in prompt.lower()

    def test_build_user_prompt_includes_json_format(self, single_combination):
        """User prompt should include JSON output format"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        formatted_examples = "## Örnek 1\nTest..."
        
        prompt = builder.build_user_prompt(single_combination, formatted_examples)
        
        assert "json" in prompt.lower()
        assert "hikaye" in prompt.lower()
        assert "secenekler" in prompt.lower()

    def test_build_user_prompt_with_additional_instructions(self, single_combination):
        """User prompt should include additional instructions if provided"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        formatted_examples = "## Örnek 1\nTest..."
        additional = "Günlük hayattan örnek kullan."
        
        prompt = builder.build_user_prompt(
            single_combination, 
            formatted_examples,
            additional_instructions=additional
        )
        
        assert additional in prompt

    def test_build_user_prompt_gorsel_instruction(self):
        """User prompt should mention gorsel if needed"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        combination_with_gorsel = {
            "alt_konu": "ebob_ekok",
            "zorluk": 4,
            "gorsel_tipi": "sematik",
            "lgs_skor": 0.85
        }
        
        prompt = builder.build_user_prompt(combination_with_gorsel, "Örnekler...")
        
        # Should mention that gorsel is needed
        assert "sematik" in prompt.lower() or "görsel" in prompt.lower()

    def test_build_user_prompt_no_gorsel(self):
        """User prompt should not require gorsel when yok"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        combination_no_gorsel = {
            "alt_konu": "ebob_ekok",
            "zorluk": 4,
            "gorsel_tipi": "yok",
            "lgs_skor": 0.85
        }
        
        prompt = builder.build_user_prompt(combination_no_gorsel, "Örnekler...")
        
        # Should be valid
        assert prompt is not None


class TestStyleVariations:
    """Test style variations for diversity"""

    def test_get_style_variations_returns_list(self):
        """get_style_variations should return list of strings"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        variations = builder.get_style_variations()
        
        assert isinstance(variations, list)
        assert len(variations) > 0
        assert all(isinstance(v, str) for v in variations)

    def test_style_variations_are_diverse(self):
        """Style variations should be different from each other"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        variations = builder.get_style_variations()
        
        # All variations should be unique
        assert len(variations) == len(set(variations))

    def test_style_variations_not_empty(self):
        """Each style variation should have content"""
        from services.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        variations = builder.get_style_variations()
        
        for variation in variations:
            assert len(variation.strip()) > 0

