# app/tests/test_conversation_guide.py

import sys
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set up test environment variables before importing services
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["VERTEX_AI_LOCATION"] = "test-location"

@pytest.mark.asyncio
@patch('app.services.conversation_guide_service.genai.Client')
async def test_conversation_guide_service(mock_genai_client):
    """Test the ConversationGuideService with various input scenarios"""
    
    # Mock the Gemini client
    mock_client = MagicMock()
    mock_genai_client.return_value = mock_client
    
    # Mock different responses for different quality levels
    def mock_generate_content(model, contents):
        prompt = contents[0] if contents else ""
        mock_response = MagicMock()
        
        # Determine response based on text length and content
        if "I went to the store today" in prompt:
            mock_response.text = '{"quality_score": 0.2, "reasoning": "The text is a brief, factual statement lacking personal feelings and contextual details."}'
        elif "I feel sad" in prompt:
            mock_response.text = '{"quality_score": 0.3, "reasoning": "The text needs more emotional context and detail."}'
        elif "frustrated and angry because my friend" in prompt:
            mock_response.text = '{"quality_score": 0.8, "reasoning": "The text provides a clear personal perspective with emotional descriptions and sufficient context."}'
        elif "weather is nice" in prompt:
            mock_response.text = '{"quality_score": 0.1, "reasoning": "The text is neutral and lacks emotional content."}'
        else:
            # For guidance generation
            mock_response.text = "I understand you're sharing something important. Could you help me understand a bit more about what you're experiencing?"
        
        return mock_response
    
    mock_client.models.generate_content.side_effect = mock_generate_content
    
    # Import after mocking
    from app.services.conversation_guide_service import ConversationGuideService
    guide_service = ConversationGuideService()
    
    # Test cases with expected outcomes
    test_cases = [
        {
            "input": "I went to the store today.",
            "description": "Simple factual statement - should need more detail",
            "expected_ready": False
        },
        {
            "input": "I feel sad.",
            "description": "Basic emotion - should need more detail",
            "expected_ready": False
        },
        {
            "input": "I feel really frustrated and angry because my friend didn't understand my feelings when I tried to explain my situation to them yesterday. It was such a difficult conversation and I felt completely misunderstood and alone.",
            "description": "Detailed emotional expression - should be ready",
            "expected_ready": True
        },
        {
            "input": "The weather is nice today.",
            "description": "Neutral statement - should need more detail",
            "expected_ready": False
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print('='*60)
        
        try:
            result = await guide_service.process_user_input(
                user_text=test_case['input'],
                current_accumulated_text="",
                current_round=0
            )
            
            # Print results
            print(f"\nResults:")
            print(f"Ready for search: {result['ready_for_search']}")
            print(f"Input round: {result['input_round']}")
            print(f"Accumulated text length: {len(result['accumulated_text'].split())} words")
            
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"Quality score: {analysis.get('quality_score', 'N/A')}")
                print(f"Reasoning: {analysis.get('reasoning', 'N/A')}")
            
            print(f"\nGuidance response:")
            print(f"{result['guidance_response']}")
            
            # Validate expectations
            if test_case['expected_ready'] != result['ready_for_search']:
                print(f"⚠️ WARNING: Expected ready_for_search={test_case['expected_ready']}, got {result['ready_for_search']}")
            else:
                print(f"✅ Expected outcome achieved")
                
        except Exception as e:
            print(f"❌ Error processing input: {str(e)}")
            import traceback
            traceback.print_exc()

@pytest.mark.asyncio
@patch('app.services.conversation_guide_service.genai.Client')
async def test_cumulative_input(mock_genai_client):
    """Test the cumulative input functionality"""
    
    # Mock the Gemini client
    mock_client = MagicMock()
    mock_genai_client.return_value = mock_client
    
    def mock_generate_content(model, contents):
        prompt = contents[0] if contents else ""
        mock_response = MagicMock()
        
        if "I feel bad today" in prompt and "criticized my work" in prompt:
            # Combined text should have higher quality
            mock_response.text = '{"quality_score": 0.7, "reasoning": "The text provides personal perspective with emotional context and situational details."}'
        elif "I feel bad today" in prompt:
            # Short text should have low quality
            mock_response.text = '{"quality_score": 0.2, "reasoning": "The text is too brief and lacks emotional detail."}'
        else:
            # Guidance response
            mock_response.text = "I can sense you're going through something difficult. Can you tell me more about what's making you feel this way?"
        
        return mock_response
    
    mock_client.models.generate_content.side_effect = mock_generate_content
    
    from app.services.conversation_guide_service import ConversationGuideService
    guide_service = ConversationGuideService()
    
    print(f"\n{'='*60}")
    print("Testing Cumulative Input Functionality")
    print('='*60)
    
    # First input - should need more detail
    result1 = await guide_service.process_user_input(
        user_text="I feel bad today.",
        current_accumulated_text="",
        current_round=0
    )
    
    print(f"\nRound 1:")
    print(f"Input: 'I feel bad today.'")
    print(f"Ready for search: {result1['ready_for_search']}")
    print(f"Accumulated text: '{result1['accumulated_text']}'")
    
    # Second input - building on the first
    result2 = await guide_service.process_user_input(
        user_text="My boss criticized my work in front of everyone and I felt humiliated and angry.",
        current_accumulated_text=result1['accumulated_text'],
        current_round=result1['input_round']
    )
    
    print(f"\nRound 2:")
    print(f"Input: 'My boss criticized my work in front of everyone and I felt humiliated and angry.'")
    print(f"Ready for search: {result2['ready_for_search']}")
    print(f"Accumulated text: '{result2['accumulated_text']}'")
    print(f"Quality score: {result2['analysis'].get('quality_score', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(test_conversation_guide_service())
    asyncio.run(test_cumulative_input())