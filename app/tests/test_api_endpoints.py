import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up test environment variables
os.environ["MONGODB_URI"] = "mongodb://test:27017"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["MONGODB_DATABASE"] = "test_db"
os.environ["MONGODB_COLLECTION"] = "test_collection"
os.environ["VERTEX_AI_LOCATION"] = "test-location"

class TestAPIEndpoints:
    """Test the main API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client with comprehensive mocking"""
        with patch('app.database.async_client'), \
             patch('app.database.sync_client'), \
             patch('app.services.search_service.text_embedding_model_service'), \
             patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel'), \
             patch('app.services.conversation_guide_service.genai.Client'), \
             patch('app.services.rag_service.genai.Client'):
            from app.main import app
            return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        with patch('app.database.async_health_check', return_value=True):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "services" in data
            assert "database" in data["services"]
            assert "vertex_ai" in data["services"]
    
    def test_health_endpoint_db_disconnected(self, client):
        """Test health endpoint when database is disconnected"""
        with patch('app.database.async_health_check', return_value=False):
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["detail"]["services"]["database"] == "disconnected"
    
    @patch('app.services.session_service.SessionService')
    @patch('app.services.conversation_guide_service.ConversationGuideService')
    def test_search_endpoint_basic(self, mock_guide_service, mock_session_service, client):
        """Test the basic search endpoint functionality"""
        # Mock session service
        mock_session = {
            "_id": "test-session-id",
            "accumulated_text": "",
            "input_round": 0
        }
        mock_session_instance = mock_session_service.return_value
        mock_session_instance.get_or_create_session = AsyncMock(return_value=mock_session)
        mock_session_instance.update_session = AsyncMock(return_value=True)
        
        # Mock guide service
        mock_guide_result = {
            "guidance_response": "Please provide more details about your feelings.",
            "accumulated_text": "I feel sad",
            "input_round": 1,
            "ready_for_search": False,
            "analysis": {
                "quality_score": 0.3,
                "reasoning": "The text needs more emotional context."
            }
        }
        mock_guide_instance = mock_guide_service.return_value
        mock_guide_instance.process_user_input = AsyncMock(return_value=mock_guide_result)
        
        response = client.post(
            "/search/",
            json={"text": "I feel sad"},
            headers={"session_id": "test-session-id"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "guidance_response" in data
        assert "ready_for_search" in data
        assert data["ready_for_search"] == False
    
    @patch('app.services.session_service.SessionService')
    @patch('app.services.search_service.perform_semantic_search')
    def test_search_execute_endpoint(self, mock_search, mock_session_service, client):
        """Test the search execute endpoint"""
        # Mock session service
        mock_session = {
            "_id": "test-session-id",
            "accumulated_text": "I feel very sad and lonely because my friend left me"
        }
        mock_session_instance = mock_session_service.return_value
        mock_session_instance.get_session = AsyncMock(return_value=mock_session)
        mock_session_instance.clear_accumulated_text = AsyncMock(return_value=True)
        
        # Mock search results
        mock_search_result = {
            "results": [
                {
                    "id": "1",
                    "text": "I feel abandoned when friends leave",
                    "emotion_label": "sadness",
                    "score": 0.95
                }
            ],
            "rag_analysis": {
                "enriched_emotion_stats": [
                    {
                        "label": "sadness",
                        "count": 10,
                        "percentage": 60.0,
                        "quote": "I feel empty when people I care about leave me behind.",
                        "analysis": "This feeling of sadness often comes from attachment and fear of abandonment."
                    }
                ],
                "summary_report": "Your feelings of sadness are valid and understandable."
            }
        }
        mock_search.return_value = mock_search_result
        
        response = client.post(
            "/search/execute",
            headers={"session_id": "test-session-id"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "results" in data
        assert "rag_analysis" in data
        assert len(data["results"]) > 0
    
    def test_search_execute_no_session(self, client):
        """Test search execute endpoint without session"""
        response = client.post("/search/execute")
        assert response.status_code == 422  # Missing required header
    
    def test_search_endpoint_empty_text(self, client):
        """Test search endpoint with empty text"""
        response = client.post(
            "/search/",
            json={"text": ""},
            headers={"session_id": "test-session-id"}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_debug_network_info_endpoint(self, client):
        """Test the debug network info endpoint"""
        with patch('app.database.sync_health_check', return_value=True):
            response = client.get("/debug/network-info")
            assert response.status_code == 200
            data = response.json()
            assert "environment" in data
            assert "mongodb_test" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 