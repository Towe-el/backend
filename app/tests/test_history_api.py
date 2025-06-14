import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.session_service import SessionService
from app.services.history_service import HistoryService

client = TestClient(app)

class TestHistoryAPI:
    """Test cases for History API endpoints"""

    @pytest.fixture
    def mock_session_service(self):
        """Mock SessionService for testing"""
        with patch('app.api.endpoints.history.session_service') as mock:
            yield mock
    
    @pytest.fixture
    def mock_history_service(self):
        """Mock HistoryService for testing"""
        with patch('app.api.endpoints.history.history_service') as mock:
            yield mock

    def test_get_title_success_ready_for_search(self, mock_session_service, mock_history_service):
        """Test successful title generation when ready_for_search is True"""
        # Mock session data
        mock_session_data = {
            "_id": "test-session-123",
            "accumulated_text": "I feel overwhelmed with work and don't know what to do anymore. The pressure is getting to me.",
            "ready_for_search": True,
            "input_round": 3
        }
        
        mock_session_service.get_session = AsyncMock(return_value=mock_session_data)
        mock_history_service.generate_title = AsyncMock(return_value="Feeling overwhelmed by work pressure")
        
        # Make request
        response = client.get("/history/title/test-session-123")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["title"] == "Feeling overwhelmed by work pressure"
        assert data["ready_for_search"] is True
        assert "successfully" in data["message"]
        
        # Verify mocks were called
        mock_session_service.get_session.assert_called_once_with("test-session-123")
        mock_history_service.generate_title.assert_called_once_with(mock_session_data["accumulated_text"])

    def test_get_title_not_ready_for_search(self, mock_session_service, mock_history_service):
        """Test when session is not ready for search"""
        # Mock session data
        mock_session_data = {
            "_id": "test-session-456",
            "accumulated_text": "I feel sad",
            "ready_for_search": False,
            "input_round": 1
        }
        
        mock_session_service.get_session = AsyncMock(return_value=mock_session_data)
        
        # Make request
        response = client.get("/history/title/test-session-456")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-456"
        assert data["title"] is None
        assert data["ready_for_search"] is False
        assert "not ready" in data["message"]
        
        # Verify title generation was not called
        mock_history_service.generate_title.assert_not_called()

    def test_get_title_empty_accumulated_text(self, mock_session_service, mock_history_service):
        """Test when session has no accumulated text"""
        # Mock session data
        mock_session_data = {
            "_id": "test-session-789",
            "accumulated_text": "",
            "ready_for_search": True,
            "input_round": 0
        }
        
        mock_session_service.get_session = AsyncMock(return_value=mock_session_data)
        
        # Make request
        response = client.get("/history/title/test-session-789")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-789"
        assert data["title"] is None
        assert data["ready_for_search"] is True
        assert "No content available" in data["message"]
        
        # Verify title generation was not called
        mock_history_service.generate_title.assert_not_called()

    def test_get_title_session_not_found(self, mock_session_service, mock_history_service):
        """Test when session doesn't exist"""
        mock_session_service.get_session = AsyncMock(return_value=None)
        
        # Make request
        response = client.get("/history/title/nonexistent-session")
        
        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "Session not found" in data["detail"]

    def test_get_title_service_error(self, mock_session_service, mock_history_service):
        """Test when session service throws an error"""
        mock_session_service.get_session = AsyncMock(side_effect=Exception("Database error"))
        
        # Make request
        response = client.get("/history/title/error-session")
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "unexpected error" in data["detail"]

    def test_get_title_generation_failure(self, mock_session_service, mock_history_service):
        """Test when title generation fails"""
        # Mock session data
        mock_session_data = {
            "_id": "test-session-failure",
            "accumulated_text": "Some text content here for testing failure scenario",
            "ready_for_search": True,
            "input_round": 2
        }
        
        mock_session_service.get_session = AsyncMock(return_value=mock_session_data)
        mock_history_service.generate_title = AsyncMock(return_value=None)  # Simulate failure
        
        # Make request
        response = client.get("/history/title/test-session-failure")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-failure"
        assert data["title"] is None
        assert data["ready_for_search"] is True
        assert "Failed to generate" in data["message"]


if __name__ == "__main__":
    # Simple integration test for development
    async def integration_test():
        print("Running integration test for History API...")
        
        # Test with real services (requires environment setup)
        try:
            session_service = SessionService()
            history_service = HistoryService()
            
            # Create a test session
            session_id = await session_service.create_session()
            print(f"Created test session: {session_id}")
            
            # Update session with some content
            test_data = {
                "accumulated_text": "I've been struggling with anxiety and depression for months now. It's affecting my work, relationships, and daily life. I feel overwhelmed and don't know where to turn for help.",
                "ready_for_search": True,
                "input_round": 3
            }
            
            await session_service.update_session(session_id, test_data)
            print("Updated session with test data")
            
            # Test title generation
            title = await history_service.generate_title(test_data["accumulated_text"])
            print(f"Generated title: '{title}'")
            
            print("Integration test completed successfully!")
            
        except Exception as e:
            print(f"Integration test failed: {e}")
    
    # Run integration test
    asyncio.run(integration_test()) 