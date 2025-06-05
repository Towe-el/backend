import unittest
from unittest.mock import patch
import os
import sys

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.conversation_guide_service import EmotionDetector

class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EmotionDetector()

    @patch('app.services.conversation_guide_service._get_vertex_embedding_service')
    def test_analyze_sentence_structure(self, mock_embedding):
        # Mock the embedding service to return a simple vector
        mock_embedding.return_value = [0.1] * 256
        
        # Test cases
        test_cases = [
            {
                'input': "I feel very sad today",
                'expected': {
                    'personal_pronoun_ratio': 0.2,  # "I" out of 5 words
                    'has_emotion_words': True
                }
            },
            {
                'input': "The weather is nice",
                'expected': {
                    'personal_pronoun_ratio': 0.0,
                    'has_emotion_words': False
                }
            },
            {
                'input': "I am feeling extremely happy and excited about my new job",
                'expected': {
                    'personal_pronoun_ratio': 0.18,  # "I", "my" out of 11 words
                    'has_emotion_words': True
                }
            }
        ]
        
        for case in test_cases:
            print(f"\nTesting input: {case['input']}")
            result = self.detector._analyze_sentence_structure(case['input'])
            print(f"Result: {result}")
            
            # Verify personal pronoun ratio
            self.assertAlmostEqual(
                result['personal_pronoun_ratio'],
                case['expected']['personal_pronoun_ratio'],
                places=2
            )
            
            # Verify emotion words presence
            has_emotion_words = result['emotion_word_ratio'] > 0
            self.assertEqual(has_emotion_words, case['expected']['has_emotion_words'])

    @patch('app.services.conversation_guide_service._get_vertex_embedding_service')
    def test_emotion_guidance(self, mock_embedding):
        # Mock the embedding service
        mock_embedding.return_value = [0.1] * 256
        
        test_cases = [
            "I went to the store today",
            "I feel very sad and lonely",
            "The weather is nice and sunny",
            "I am extremely angry and frustrated with this situation"
        ]
        
        for text in test_cases:
            print(f"\nTesting input: {text}")
            analysis = self.detector.analyze_emotion_content(text)
            guidance = self.detector.get_emotion_guidance(analysis)
            print(f"Emotion intensity: {analysis['emotion_intensity']:.2f}")
            print(f"Confidence: {analysis['confidence']:.2f}")
            print(f"Guidance: {guidance}")
            
            # Basic validation
            self.assertIsInstance(guidance, str)
            self.assertTrue(len(guidance) > 0)

if __name__ == '__main__':
    unittest.main()