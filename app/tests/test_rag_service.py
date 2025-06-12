import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up test environment variables
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

class TestRAGService(unittest.TestCase):
    """Test the RAG (Retrieval-Augmented Generation) service"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_search_results = [
            {
                "text": "I feel so sad when my friends ignore me",
                "emotion_label": '[{"tag": "sadness", "cnt": 3}, {"tag": "disappointment", "cnt": 1}]',
                "score": 0.95
            },
            {
                "text": "I am really angry about this situation",
                "emotion_label": '[{"tag": "anger", "cnt": 2}]',
                "score": 0.88
            },
            {
                "text": "I feel disappointed in myself",
                "emotion_label": '[{"tag": "disappointment", "cnt": 2}, {"tag": "sadness", "cnt": 1}]',
                "score": 0.82
            }
        ]
        
        self.user_input = "I feel really sad and disappointed because my friends didn't invite me to their party"
    
    @patch('app.services.rag_service.genai.Client')
    def test_rag_processor_initialization(self, mock_genai_client):
        """Test RAG processor initialization"""
        from app.services.rag_service import RAGProcessor
        
        processor = RAGProcessor()
        self.assertIsNotNone(processor.client)
        self.assertEqual(processor.model, "gemini-2.0-flash-001")
    
    @patch('app.services.rag_service.genai.Client')
    def test_process_search_results(self, mock_genai_client):
        """Test processing of search results"""
        from app.services.rag_service import RAGProcessor
        
        # Mock the Gemini client responses
        mock_client = MagicMock()
        mock_genai_client.return_value = mock_client
        
        # Mock emotion profile response
        mock_emotion_response = MagicMock()
        mock_emotion_response.text = '{"sadness": 0.8, "disappointment": 0.6, "anger": 0.1, "neutral": 0.0}'
        
        # Mock quote generation response
        mock_quote_response = MagicMock()
        mock_quote_response.text = "I feel empty when my friends exclude me from their plans."
        
        # Mock analysis response
        mock_analysis_response = MagicMock()
        mock_analysis_response.text = "It's completely understandable that you're feeling sadness and disappointment. Being excluded from social gatherings can trigger feelings of rejection and loneliness."
        
        # Mock summary response
        mock_summary_response = MagicMock()
        mock_summary_response.text = "Your feelings of sadness and disappointment are valid responses to social exclusion."
        
        mock_client.models.generate_content.side_effect = [
            mock_emotion_response,  # For emotion profile
            mock_quote_response,    # For quote generation
            mock_analysis_response, # For analysis
            mock_quote_response,    # For second emotion quote
            mock_analysis_response, # For second emotion analysis
            mock_summary_response   # For summary
        ]
        
        processor = RAGProcessor()
        result = processor.process_search_results(self.sample_search_results, self.user_input)
        
        # Verify the structure of the result
        self.assertIn("enriched_emotion_stats", result)
        self.assertIn("summary_report", result)
        self.assertIsInstance(result["enriched_emotion_stats"], list)
        self.assertIsInstance(result["summary_report"], str)
        
        # Verify that we got some emotion stats
        if result["enriched_emotion_stats"]:
            emotion_stat = result["enriched_emotion_stats"][0]
            self.assertIn("label", emotion_stat)
            self.assertIn("count", emotion_stat)
            self.assertIn("percentage", emotion_stat)
            self.assertIn("quote", emotion_stat)
            self.assertIn("analysis", emotion_stat)
    
    @patch('app.services.rag_service.genai.Client')
    def test_extract_primary_emotion(self, mock_genai_client):
        """Test emotion extraction from complex labels"""
        from app.services.rag_service import RAGProcessor
        
        processor = RAGProcessor()
        
        # Test with string representation of list
        emotion_label = '[{"tag": "sadness", "cnt": 3}, {"tag": "anger", "cnt": 1}]'
        result = processor._extract_primary_emotion(emotion_label)
        self.assertEqual(result, "sadness")  # Should return the one with highest count
        
        # Test with actual list
        emotion_label = [{"tag": "joy", "cnt": 2}, {"tag": "excitement", "cnt": 1}]
        result = processor._extract_primary_emotion(emotion_label)
        self.assertEqual(result, "joy")
        
        # Test with simple string
        emotion_label = "happiness"
        result = processor._extract_primary_emotion(emotion_label)
        self.assertEqual(result, "happiness")
        
        # Test with neutral emotion (should be filtered out)
        emotion_label = '[{"tag": "neutral", "cnt": 5}, {"tag": "sadness", "cnt": 1}]'
        result = processor._extract_primary_emotion(emotion_label)
        self.assertEqual(result, "sadness")  # Should skip neutral and return sadness
    
    @patch('app.services.rag_service.genai.Client')
    def test_calculate_emotion_stats_filtering(self, mock_genai_client):
        """Test emotion statistics calculation with filtering"""
        from app.services.rag_service import RAGProcessor
        
        # Mock the emotion profile response
        mock_client = MagicMock()
        mock_genai_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = '{"sadness": 0.8, "disappointment": 0.6, "anger": 0.1, "joy": 0.0, "neutral": 0.0}'
        mock_client.models.generate_content.return_value = mock_response
        
        processor = RAGProcessor()
        
        # Test with search results that include neutral emotions
        search_results_with_neutral = [
            {
                "text": "Today was okay",
                "emotion_label": '[{"tag": "neutral", "cnt": 5}]',
                "score": 0.5
            },
            {
                "text": "I feel sad",
                "emotion_label": '[{"tag": "sadness", "cnt": 3}]',
                "score": 0.9
            }
        ]
        
        result = processor._calculate_emotion_stats(search_results_with_neutral, self.user_input)
        
        # Verify that neutral emotions are filtered out
        emotion_labels = [stat["label"] for stat in result]
        self.assertNotIn("neutral", emotion_labels)
        
        # Verify that we still get valid emotions
        if result:
            self.assertIn("sadness", emotion_labels)
    
    @patch('app.services.rag_service.genai.Client')
    def test_error_handling(self, mock_genai_client):
        """Test error handling in RAG service"""
        from app.services.rag_service import RAGProcessor
        
        # Mock client that raises an exception
        mock_client = MagicMock()
        mock_genai_client.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API Error")
        
        processor = RAGProcessor()
        
        # Should not crash, should return empty results
        result = processor.process_search_results(self.sample_search_results, self.user_input)
        
        self.assertIn("enriched_emotion_stats", result)
        self.assertIn("summary_report", result)
        # Should handle errors gracefully
        self.assertIsInstance(result["enriched_emotion_stats"], list)
        self.assertIsInstance(result["summary_report"], str)

if __name__ == '__main__':
    unittest.main() 