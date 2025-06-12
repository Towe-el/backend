import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up test environment variables
os.environ["MONGODB_URI"] = "mongodb://test:27017"
os.environ["MONGODB_DATABASE"] = "test_db"
os.environ["MONGODB_COLLECTION"] = "test_collection"

class TestSessionService(unittest.TestCase):
    """Test the SessionService"""
    
    @patch('app.database.async_db')
    def test_session_service_initialization(self, mock_async_db):
        """Test SessionService initialization"""
        # Mock the database
        mock_collection = MagicMock()
        mock_async_db.__getitem__.return_value = mock_collection
        
        from app.services.session_service import SessionService
        
        service = SessionService()
        self.assertIsNotNone(service.db)
        self.assertIsNotNone(service.collection)
    
    @patch('app.database.async_db')
    def test_session_service_with_none_db(self, mock_async_db):
        """Test SessionService when database is None"""
        mock_async_db = None
        
        with patch('app.services.session_service.async_db', None):
            from app.services.session_service import SessionService
            service = SessionService()
            self.assertIsNone(service.db)
            self.assertIsNone(service.collection)
    
    @patch('app.database.async_db')
    async def test_create_session(self, mock_async_db):
        """Test session creation"""
        # Mock the database and collection
        mock_collection = AsyncMock()
        mock_async_db.__getitem__.return_value = mock_collection
        
        from app.services.session_service import SessionService
        service = SessionService()
        
        # Test session creation
        session_id = await service.create_session()
        
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) > 0)
        mock_collection.insert_one.assert_called_once()
    
    @patch('app.database.async_db')
    async def test_get_session(self, mock_async_db):
        """Test session retrieval"""
        # Mock the database and collection
        mock_collection = AsyncMock()
        mock_session_data = {
            "_id": "test-session-id",
            "accumulated_text": "test text",
            "input_round": 1
        }
        mock_collection.find_one.return_value = mock_session_data
        mock_async_db.__getitem__.return_value = mock_collection
        
        from app.services.session_service import SessionService
        service = SessionService()
        
        # Test session retrieval
        result = await service.get_session("test-session-id")
        
        self.assertEqual(result, mock_session_data)
        mock_collection.find_one.assert_called_once_with({"_id": "test-session-id"})
    
    @patch('app.database.async_db')
    async def test_update_session(self, mock_async_db):
        """Test session update"""
        # Mock the database and collection
        mock_collection = AsyncMock()
        mock_result = MagicMock()
        mock_result.modified_count = 1
        mock_result.matched_count = 1
        mock_collection.update_one.return_value = mock_result
        mock_async_db.__getitem__.return_value = mock_collection
        
        from app.services.session_service import SessionService
        service = SessionService()
        
        # Test session update
        update_data = {"accumulated_text": "updated text"}
        result = await service.update_session("test-session-id", update_data)
        
        self.assertTrue(result)
        mock_collection.update_one.assert_called_once()
    
    async def test_database_unavailable_operations(self):
        """Test operations when database is unavailable"""
        with patch('app.services.session_service.async_db', None):
            from app.services.session_service import SessionService
            service = SessionService()
            
            # Test that operations handle None database gracefully
            with self.assertRaises(RuntimeError):
                await service.create_session()
            
            result = await service.get_session("test-id")
            self.assertIsNone(result)
            
            result = await service.update_session("test-id", {})
            self.assertFalse(result)

if __name__ == '__main__':
    import asyncio
    
    # Run async tests
    async def run_async_tests():
        suite = unittest.TestSuite()
        
        # Add async test methods
        test_instance = TestSessionService()
        suite.addTest(TestSessionService('test_create_session'))
        suite.addTest(TestSessionService('test_get_session'))
        suite.addTest(TestSessionService('test_update_session'))
        suite.addTest(TestSessionService('test_database_unavailable_operations'))
        
        # Note: This is a simplified runner for async tests
        # In practice, you'd use pytest-asyncio or similar
        print("Running async SessionService tests...")
        
        try:
            await test_instance.test_create_session()
            print("✅ test_create_session passed")
        except Exception as e:
            print(f"❌ test_create_session failed: {e}")
        
        try:
            await test_instance.test_get_session()
            print("✅ test_get_session passed")
        except Exception as e:
            print(f"❌ test_get_session failed: {e}")
        
        try:
            await test_instance.test_update_session()
            print("✅ test_update_session passed")
        except Exception as e:
            print(f"❌ test_update_session failed: {e}")
        
        try:
            await test_instance.test_database_unavailable_operations()
            print("✅ test_database_unavailable_operations passed")
        except Exception as e:
            print(f"❌ test_database_unavailable_operations failed: {e}")
    
    # Run sync tests first
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Then run async tests
    asyncio.run(run_async_tests()) 