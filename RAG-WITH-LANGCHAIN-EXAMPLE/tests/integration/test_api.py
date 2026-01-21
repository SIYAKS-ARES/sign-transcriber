"""
Integration Tests for FastAPI Endpoints
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json


@pytest.mark.api
class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_endpoint(self):
        """Health endpoint should return 200"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_includes_version(self):
        """Health endpoint should include version info"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        response = client.get("/api/v1/health")
        
        data = response.json()
        assert "version" in data


@pytest.mark.api
class TestGenerateEndpoint:
    """Test question generation endpoint"""

    @pytest.fixture
    def mock_generator(self, mock_generated_question):
        """Mock question generator"""
        from models.question import GeneratedQuestion
        
        mock = AsyncMock()
        mock.generate_question.return_value = GeneratedQuestion(
            hikaye=mock_generated_question["hikaye"],
            soru=mock_generated_question["soru"],
            gorsel_aciklama=mock_generated_question.get("gorsel_aciklama"),
            secenekler=mock_generated_question["secenekler"],
            dogru_cevap=mock_generated_question["dogru_cevap"],
            cozum=mock_generated_question["cozum"],
            combination={"alt_konu": "ebob_ekok", "zorluk": 4, "gorsel_tipi": "sematik"},
            metadata={"attempt": 1}
        )
        return mock

    def test_generate_endpoint_success(self, mock_generator, api_generate_request):
        """Generate endpoint should return question"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        with patch("api.main.get_generator", return_value=mock_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate",
                json=api_generate_request
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data

    def test_generate_endpoint_returns_question_fields(self, mock_generator, api_generate_request):
        """Generate endpoint should return all question fields"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        with patch("api.main.get_generator", return_value=mock_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate",
                json=api_generate_request
            )
        
        data = response.json()
        question = data["data"]
        
        assert "hikaye" in question
        assert "soru" in question
        assert "secenekler" in question
        assert "dogru_cevap" in question
        assert "cozum" in question

    def test_generate_endpoint_with_filters(self, mock_generator):
        """Generate endpoint should accept filters"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        request = {
            "filters": {
                "alt_konu": "carpanlar",
                "zorluk": 3
            }
        }
        
        with patch("api.main.get_generator", return_value=mock_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate",
                json=request
            )
        
        assert response.status_code == 200

    def test_generate_endpoint_validation_error(self, mock_generator):
        """Generate endpoint should return 422 for invalid request"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        invalid_request = {
            "filters": {
                "zorluk": "invalid"  # Should be int
            }
        }
        
        with patch("api.main.get_generator", return_value=mock_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate",
                json=invalid_request
            )
        
        # FastAPI returns 422 for validation errors
        assert response.status_code in [400, 422]


@pytest.mark.api
class TestBatchGenerateEndpoint:
    """Test batch generation endpoint"""

    @pytest.fixture
    def mock_batch_generator(self, mock_generated_question):
        """Mock generator for batch"""
        from models.question import GeneratedQuestion
        
        question = GeneratedQuestion(
            hikaye=mock_generated_question["hikaye"],
            soru=mock_generated_question["soru"],
            gorsel_aciklama=mock_generated_question.get("gorsel_aciklama"),
            secenekler=mock_generated_question["secenekler"],
            dogru_cevap=mock_generated_question["dogru_cevap"],
            cozum=mock_generated_question["cozum"],
            combination={"alt_konu": "ebob_ekok", "zorluk": 4, "gorsel_tipi": "sematik"},
            metadata={"attempt": 1}
        )
        
        mock = AsyncMock()
        mock.generate_diverse_batch.return_value = [question] * 3
        return mock

    def test_batch_endpoint_success(self, mock_batch_generator, api_batch_request):
        """Batch endpoint should return multiple questions"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        with patch("api.main.get_diverse_generator", return_value=mock_batch_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate/batch",
                json=api_batch_request
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "questions" in data["data"]

    def test_batch_endpoint_respects_count(self, mock_batch_generator, api_batch_request):
        """Batch endpoint should return requested count"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        with patch("api.main.get_diverse_generator", return_value=mock_batch_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate/batch",
                json=api_batch_request
            )
        
        data = response.json()
        assert len(data["data"]["questions"]) <= api_batch_request["count"]


@pytest.mark.api
class TestCombinationsEndpoint:
    """Test combinations listing endpoint"""

    def test_combinations_endpoint(self, sample_combinations):
        """Combinations endpoint should return list"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        with patch("api.main.get_combinations", return_value=sample_combinations):
            client = TestClient(app)
            response = client.get("/api/v1/combinations")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_combinations_endpoint_with_filter(self, sample_combinations):
        """Combinations endpoint should accept filters"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        filtered = [c for c in sample_combinations if c["alt_konu"] == "ebob_ekok"]
        
        with patch("api.main.get_combinations", return_value=filtered):
            client = TestClient(app)
            response = client.get(
                "/api/v1/combinations",
                params={"alt_konu": "ebob_ekok"}
            )
        
        assert response.status_code == 200
        data = response.json()
        for combo in data:
            assert combo["alt_konu"] == "ebob_ekok"


@pytest.mark.api
class TestStatsEndpoint:
    """Test statistics endpoint"""

    def test_stats_endpoint(self):
        """Stats endpoint should return statistics"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have some stats
        assert isinstance(data, dict)


@pytest.mark.api
class TestErrorHandling:
    """Test API error handling"""

    def test_internal_error_handling(self):
        """Internal errors should return 500"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        mock_generator = AsyncMock()
        mock_generator.generate_question.side_effect = Exception("Internal error")
        
        with patch("api.main.get_generator", return_value=mock_generator):
            client = TestClient(app)
            response = client.post(
                "/api/v1/generate",
                json={"filters": {}}
            )
        
        assert response.status_code == 500

    def test_not_found_handling(self):
        """Unknown endpoints should return 404"""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        response = client.get("/api/v1/unknown")
        
        assert response.status_code == 404

