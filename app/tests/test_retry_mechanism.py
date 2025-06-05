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

        # Set up class-level patches
        cls.aiplatform_patcher = patch('app.services.search_service.aiplatform')
        cls.mongo_client_patcher = patch('app.services.search_service.MongoClient')
        cls.text_model_patcher = patch('app.services.search_service.TextEmbeddingModel')

        # Start patches
        cls.mock_aiplatform = cls.aiplatform_patcher.start()
        cls.mock_mongo_client = cls.mongo_client_patcher.start()
        cls.mock_text_model = cls.text_model_patcher.start()

    @classmethod
    def tearDownClass(cls):
        """Clean up patches"""
        cls.aiplatform_patcher.stop()
        cls.mongo_client_patcher.stop()
        cls.text_model_patcher.stop()

    def setUp(self):
        """Set up test fixtures"""
        self.test_text = "This is a test sentence."
        self.mock_embedding = [0.1] * 256

        # Set up mock response
        mock_response = MagicMock()
        mock_response.values = self.mock_embedding

        # Set up text embedding model
        self.mock_model = MagicMock()
        self.mock_text_model.from_pretrained.return_value = self.mock_model

        # Import after mocks are set up
        import app.services.search_service as search_service
        search_service.text_embedding_model_service = self.mock_model
        self.get_embedding_core = search_service._get_vertex_embedding_service_core
        self.get_embedding = search_service._get_vertex_embedding_service

        # Clear the LRU cache
        self.get_embedding.cache_clear()

    def test_retry_on_resource_exhausted(self):
        """测试资源耗尽时的重试行为"""
        mock_response = MagicMock()
        mock_response.values = self.mock_embedding
        self.mock_model.get_embeddings.side_effect = [
            google_exceptions.ResourceExhausted("Resource exhausted"),
            google_exceptions.ResourceExhausted("Resource exhausted"),
            [mock_response]
        ]

        result = self.get_embedding(self.test_text)
        
        self.assertEqual(result, self.mock_embedding)
        self.assertEqual(self.mock_model.get_embeddings.call_count, 3)

    def test_retry_on_service_unavailable(self):
        """测试服务不可用时的重试行为"""
        mock_response = MagicMock()
        mock_response.values = self.mock_embedding
        self.mock_model.get_embeddings.side_effect = [
            google_exceptions.ServiceUnavailable("Service unavailable"),
            [mock_response]
        ]

        result = self.get_embedding(self.test_text)
        
        self.assertEqual(result, self.mock_embedding)
        self.assertEqual(self.mock_model.get_embeddings.call_count, 2)

    def test_retry_on_deadline_exceeded(self):
        """测试超时时的重试行为"""
        mock_response = MagicMock()
        mock_response.values = self.mock_embedding
        self.mock_model.get_embeddings.side_effect = [
            google_exceptions.DeadlineExceeded("Deadline exceeded"),
            google_exceptions.DeadlineExceeded("Deadline exceeded"),
            [mock_response]
        ]

        result = self.get_embedding(self.test_text)
        
        self.assertEqual(result, self.mock_embedding)
        self.assertEqual(self.mock_model.get_embeddings.call_count, 3)

    def test_retry_on_connection_error(self):
        """测试连接错误时的重试行为"""
        mock_response = MagicMock()
        mock_response.values = self.mock_embedding
        self.mock_model.get_embeddings.side_effect = [
            ConnectionError("Connection failed"),
            [mock_response]
        ]

        result = self.get_embedding(self.test_text)
        
        self.assertEqual(result, self.mock_embedding)
        self.assertEqual(self.mock_model.get_embeddings.call_count, 2)

    def test_no_retry_on_value_error(self):
        """测试不应该重试的错误类型"""
        self.mock_model.get_embeddings.side_effect = ValueError("Invalid input")

        with self.assertRaises(ValueError):
            result = self.get_embedding(self.test_text)

    def test_max_retries_exceeded(self):
        """测试超过最大重试次数的情况"""
        self.mock_model.get_embeddings.side_effect = google_exceptions.ResourceExhausted("Resource exhausted")

        with self.assertRaises(google_exceptions.ResourceExhausted):
            result = self.get_embedding(self.test_text)
        
        self.assertEqual(self.mock_model.get_embeddings.call_count, 3)  # 应该只重试3次

    def test_cache_behavior_during_retries(self):
        """测试重试过程中的缓存行为"""
        mock_response = MagicMock()
        mock_response.values = self.mock_embedding
        
        # 第一次调用：失败后重试成功
        self.mock_model.get_embeddings.side_effect = [
            google_exceptions.ResourceExhausted("Resource exhausted"),
            [mock_response]
        ]
        
        # 第一次调用
        result1 = self.get_embedding(self.test_text)
        self.assertEqual(result1, self.mock_embedding)
        self.assertEqual(self.mock_model.get_embeddings.call_count, 2)

        # 重置mock
        self.mock_model.get_embeddings.reset_mock()
        self.mock_model.get_embeddings.side_effect = google_exceptions.ResourceExhausted("Should not be called")

        # 第二次调用同样的文本应该直接返回缓存结果
        result2 = self.get_embedding(self.test_text)
        self.assertEqual(result2, self.mock_embedding)
        self.assertEqual(self.mock_model.get_embeddings.call_count, 0)  # 不应该调用API

    def test_empty_input(self):
        """测试空输入的处理"""
        result = self.get_embedding("")
        self.assertEqual(result, [])

        result = self.get_embedding("   ")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main() 