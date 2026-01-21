"""
Pytest Fixtures for LGS RAG Tests
"""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# FIXTURES PATH
# =============================================================================

@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_configs_path(fixtures_path: Path) -> Path:
    """Path to sample configs.json"""
    return fixtures_path / "sample_configs.json"


@pytest.fixture
def sample_questions_path(fixtures_path: Path) -> Path:
    """Path to sample questions CSV"""
    return fixtures_path / "sample_questions.csv"


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def sample_configs(sample_configs_path: Path) -> dict[str, Any]:
    """Load sample configs.json for testing"""
    with open(sample_configs_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_combinations(sample_configs: dict) -> list[dict]:
    """Extract combinations from sample configs"""
    return sample_configs["combinations"]


@pytest.fixture
def single_combination() -> dict:
    """Single combination for unit tests"""
    return {
        "alt_konu": "ebob_ekok",
        "zorluk": 4,
        "gorsel_tipi": "sematik",
        "lgs_skor": 0.85,
        "rank": 1
    }


# =============================================================================
# QUESTION FIXTURES
# =============================================================================

@pytest.fixture
def sample_question_metadata() -> list[dict]:
    """Sample question metadata list"""
    return [
        {
            "Soru_MetniOCR": "24 ve 36 sayılarının EBOB'u kaçtır?",
            "Alt_Konu": "ebob_ekok",
            "Zorluk": 2,
            "Gorsel_Tipi": "yok",
            "Kaynak_Tipi": "baslangic",
            "is_LGS": 0,
            "egitim_agirligi": 0.3
        },
        {
            "Soru_MetniOCR": "Ali 24 elmayı, Ayşe 36 elmayı eşit gruplara ayırmak istiyor.",
            "Alt_Konu": "ebob_ekok",
            "Zorluk": 3,
            "Gorsel_Tipi": "yok",
            "Kaynak_Tipi": "ornek",
            "is_LGS": 1,
            "egitim_agirligi": 0.8
        },
        {
            "Soru_MetniOCR": "Bir fabrikanın iki üretim hattı vardır.",
            "Alt_Konu": "ebob_ekok",
            "Zorluk": 4,
            "Gorsel_Tipi": "sematik",
            "Kaynak_Tipi": "cikmis",
            "is_LGS": 1,
            "egitim_agirligi": 1.0
        },
        {
            "Soru_MetniOCR": "48 sayısının asal çarpanlarını yazınız.",
            "Alt_Konu": "carpanlar",
            "Zorluk": 2,
            "Gorsel_Tipi": "yok",
            "Kaynak_Tipi": "baslangic",
            "is_LGS": 0,
            "egitim_agirligi": 0.3
        },
        {
            "Soru_MetniOCR": "15 ve 28 sayıları aralarında asal mıdır?",
            "Alt_Konu": "aralarinda_asal",
            "Zorluk": 2,
            "Gorsel_Tipi": "yok",
            "Kaynak_Tipi": "baslangic",
            "is_LGS": 0,
            "egitim_agirligi": 0.3
        }
    ]


@pytest.fixture
def mock_generated_question() -> dict:
    """Mock LLM generated question response"""
    return {
        "hikaye": "Bir fabrikada üretilen ürünler kutulara yerleştiriliyor. "
                  "Birinci üretim hattından 24, ikinci hattan 36 ürün çıkıyor.",
        "soru": "Buna göre, her kutuya eşit sayıda ürün konulacak şekilde "
                "en fazla kaçar ürün konulabilir?",
        "gorsel_aciklama": "İki üretim hattını gösteren bir şema. "
                          "Sol tarafta 24 ürün, sağ tarafta 36 ürün var.",
        "secenekler": {
            "A": "6",
            "B": "8",
            "C": "12",
            "D": "18"
        },
        "dogru_cevap": "C",
        "cozum": [
            "Adım 1: 24 sayısının çarpanlarını bulalım: 1, 2, 3, 4, 6, 8, 12, 24",
            "Adım 2: 36 sayısının çarpanlarını bulalım: 1, 2, 3, 4, 6, 9, 12, 18, 36",
            "Adım 3: Ortak çarpanlar: 1, 2, 3, 4, 6, 12",
            "Adım 4: En büyük ortak çarpan (EBOB) = 12"
        ]
    }


@pytest.fixture
def invalid_generated_question() -> dict:
    """Invalid LLM response for testing validation"""
    return {
        "hikaye": "Test hikayesi",
        "soru": "Test sorusu",
        # Missing secenekler
        "dogru_cevap": "E",  # Invalid option
        "cozum": []  # Empty solution
    }


# =============================================================================
# SERVICE MOCKS
# =============================================================================

@pytest.fixture
def mock_embedding_pipeline():
    """Mock EmbeddingPipeline for testing"""
    mock = MagicMock()
    mock.dimension = 768
    mock.index = MagicMock()
    mock.index.ntotal = 15
    mock.id_to_metadata = {}
    
    # Mock search results
    mock.search.return_value = [
        (0.95, {"Soru_MetniOCR": "Test soru 1", "Alt_Konu": "ebob_ekok", "Zorluk": 4}),
        (0.90, {"Soru_MetniOCR": "Test soru 2", "Alt_Konu": "ebob_ekok", "Zorluk": 3}),
        (0.85, {"Soru_MetniOCR": "Test soru 3", "Alt_Konu": "ebob_ekok", "Zorluk": 4}),
    ]
    
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    mock = AsyncMock()
    mock.generate.return_value = json.dumps({
        "hikaye": "Test hikayesi",
        "soru": "Test sorusu?",
        "gorsel_aciklama": None,
        "secenekler": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "dogru_cevap": "C",
        "cozum": ["Adım 1: ...", "Adım 2: ..."]
    })
    return mock


@pytest.fixture
def mock_llm_client_with_json_block():
    """Mock LLM client that returns JSON in markdown block"""
    mock = AsyncMock()
    mock.generate.return_value = """```json
{
    "hikaye": "Test hikayesi",
    "soru": "Test sorusu?",
    "gorsel_aciklama": null,
    "secenekler": {"A": "1", "B": "2", "C": "3", "D": "4"},
    "dogru_cevap": "C",
    "cozum": ["Adım 1: ...", "Adım 2: ..."]
}
```"""
    return mock


# =============================================================================
# FILTER FIXTURES
# =============================================================================

@pytest.fixture
def filter_criteria_ebob():
    """Filter criteria for ebob_ekok topic"""
    from services.filter_service import FilterCriteria
    return FilterCriteria(
        alt_konu="ebob_ekok",
        zorluk=4,
        gorsel_tipi="sematik",
        is_lgs=1
    )


@pytest.fixture
def filter_criteria_with_range():
    """Filter criteria with difficulty range"""
    from services.filter_service import FilterCriteria
    return FilterCriteria(
        alt_konu="ebob_ekok",
        zorluk_range=(3, 5),
        kaynak_tipi=["cikmis", "ornek"]
    )


# =============================================================================
# API TEST FIXTURES
# =============================================================================

@pytest.fixture
def api_generate_request() -> dict:
    """Sample API generate request"""
    return {
        "filters": {
            "alt_konu": "ebob_ekok",
            "zorluk": 4
        },
        "style_instruction": "Günlük hayattan örnek kullan."
    }


@pytest.fixture
def api_batch_request() -> dict:
    """Sample API batch generate request"""
    return {
        "count": 5,
        "filters": {
            "alt_konu": "carpanlar"
        },
        "ensure_diversity": True
    }


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )

