import unittest
from unittest.mock import patch, Mock, MagicMock
import os
import sys
from google.api_core import exceptions as google_exceptions

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestRetryMechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment variables"""
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
        os.environ["MONGODB_URI"] = "mongodb://test:27017"
        os.environ["MONGODB_DATABASE"] = "test_db"
        os.environ["MONGODB_COLLECTION"] = "test_collection"

    def setUp(self):
        """Set up test fixtures"""
        self.test_text = "This is a test sentence."
        self.mock_embedding = [0.1] * 256

    def test_retry_on_resource_exhausted(self):
        """Test retry behavior on resource exhausted error"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            # Set up mock response
            mock_response = MagicMock()
            mock_response.values = self.mock_embedding

            # Set up text embedding model
            mock_model = MagicMock()
            mock_text_model.from_pretrained.return_value = mock_model
            mock_model.get_embeddings.side_effect = [
                google_exceptions.ResourceExhausted("Resource exhausted"),
                google_exceptions.ResourceExhausted("Resource exhausted"),
                [mock_response]
            ]

            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            search_service.text_embedding_model_service = mock_model
            
            from app.services.search_service import _get_vertex_embedding_service
            
            # Clear the LRU cache
            _get_vertex_embedding_service.cache_clear()
            
            result = _get_vertex_embedding_service(self.test_text)
            
            self.assertEqual(result, self.mock_embedding)
            self.assertEqual(mock_model.get_embeddings.call_count, 3)

    def test_retry_on_service_unavailable(self):
        """Test retry behavior on service unavailable error"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            mock_response = MagicMock()
            mock_response.values = self.mock_embedding

            mock_model = MagicMock()
            mock_text_model.from_pretrained.return_value = mock_model
            mock_model.get_embeddings.side_effect = [
                google_exceptions.ServiceUnavailable("Service unavailable"),
                [mock_response]
            ]

            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            search_service.text_embedding_model_service = mock_model

            from app.services.search_service import _get_vertex_embedding_service
            _get_vertex_embedding_service.cache_clear()
            
            result = _get_vertex_embedding_service(self.test_text)
            
            self.assertEqual(result, self.mock_embedding)
            self.assertEqual(mock_model.get_embeddings.call_count, 2)

    def test_retry_on_deadline_exceeded(self):
        """Test retry behavior on deadline exceeded error"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            mock_response = MagicMock()
            mock_response.values = self.mock_embedding

            mock_model = MagicMock()
            mock_text_model.from_pretrained.return_value = mock_model
            mock_model.get_embeddings.side_effect = [
                google_exceptions.DeadlineExceeded("Deadline exceeded"),
                google_exceptions.DeadlineExceeded("Deadline exceeded"),
                [mock_response]
            ]

            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            search_service.text_embedding_model_service = mock_model

            from app.services.search_service import _get_vertex_embedding_service
            _get_vertex_embedding_service.cache_clear()
            
            result = _get_vertex_embedding_service(self.test_text)
            
            self.assertEqual(result, self.mock_embedding)
            self.assertEqual(mock_model.get_embeddings.call_count, 3)

    def test_no_retry_on_value_error(self):
        """Test that non-retryable errors are not retried"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            mock_model = MagicMock()
            mock_text_model.from_pretrained.return_value = mock_model
            mock_model.get_embeddings.side_effect = ValueError("Invalid input")

            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            search_service.text_embedding_model_service = mock_model

            from app.services.search_service import _get_vertex_embedding_service
            _get_vertex_embedding_service.cache_clear()

            with self.assertRaises(ValueError):
                _get_vertex_embedding_service(self.test_text)

    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            mock_model = MagicMock()
            mock_text_model.from_pretrained.return_value = mock_model
            mock_model.get_embeddings.side_effect = google_exceptions.ResourceExhausted("Resource exhausted")

            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            search_service.text_embedding_model_service = mock_model

            from app.services.search_service import _get_vertex_embedding_service
            _get_vertex_embedding_service.cache_clear()

            with self.assertRaises(google_exceptions.ResourceExhausted):
                _get_vertex_embedding_service(self.test_text)
            
            self.assertEqual(mock_model.get_embeddings.call_count, 3)  # Should retry 3 times

    def test_cache_behavior_during_retries(self):
        """Test caching behavior during retries"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            mock_response = MagicMock()
            mock_response.values = self.mock_embedding
            
            mock_model = MagicMock()
            mock_text_model.from_pretrained.return_value = mock_model
            
            # First call: fail then succeed
            mock_model.get_embeddings.side_effect = [
                google_exceptions.ResourceExhausted("Resource exhausted"),
                [mock_response]
            ]

            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            search_service.text_embedding_model_service = mock_model

            from app.services.search_service import _get_vertex_embedding_service
            _get_vertex_embedding_service.cache_clear()
            
            # First call
            result1 = _get_vertex_embedding_service(self.test_text)
            self.assertEqual(result1, self.mock_embedding)
            self.assertEqual(mock_model.get_embeddings.call_count, 2)

            # Reset mock
            mock_model.get_embeddings.reset_mock()
            mock_model.get_embeddings.side_effect = google_exceptions.ResourceExhausted("Should not be called")

            # Second call with same text should return cached result
            result2 = _get_vertex_embedding_service(self.test_text)
            self.assertEqual(result2, self.mock_embedding)
            self.assertEqual(mock_model.get_embeddings.call_count, 0)  # Should not call API

    def test_empty_input(self):
        """Test handling of empty input"""
        with patch('app.services.search_service.aiplatform'), \
             patch('app.services.search_service.TextEmbeddingModel') as mock_text_model, \
             patch('app.services.search_service.sync_client') as mock_sync_client, \
             patch('app.services.search_service.sync_db') as mock_sync_db:
            
            # Mock database connection
            mock_sync_client.admin.command.return_value = True
            mock_collection = MagicMock()
            mock_sync_db.__getitem__.return_value = mock_collection

            # Import and patch the global variable after mocks are set up
            import app.services.search_service as search_service
            mock_model = MagicMock()
            search_service.text_embedding_model_service = mock_model

            from app.services.search_service import _get_vertex_embedding_service
            _get_vertex_embedding_service.cache_clear()
            
            result = _get_vertex_embedding_service("")
            self.assertEqual(result, [])

            result = _get_vertex_embedding_service("   ")
            self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main() 