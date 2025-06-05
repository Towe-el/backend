# app/tests/test_conversation_guide.py

import sys
import os
import asyncio
import pytest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.services.conversation_guide_service import ConversationGuideService

@pytest.mark.asyncio
async def test_conversation_guide():
    guide_service = ConversationGuideService()
    
    # 测试用例列表
    test_inputs = [
        # 陈述性句子，缺乏情感表达
        "I went to the store today and bought some groceries.",
        
        # 简单情感表达
        "I feel sad.",
        
        # 较详细的情感表达
        "I feel really frustrated and angry because my friend didn't understand my feelings when I tried to explain my situation.",
        
        # 中性描述
        "The weather is nice today and I finished my work.",
        
        # 混合情感表达
        "I'm happy about getting the job, but I'm also nervous about starting in a new environment."
    ]

    for input_text in test_inputs:
        print("\n" + "="*50)
        print(f"Testing input: {input_text}")
        print("="*50)
        
        try:
            result = await guide_service.process_user_input(input_text)
            
            # 打印分析结果
            print("\nAnalysis Results:")
            print(f"Sentence count: {result['analysis']['sentence_count']}")
            
            emotion_analysis = result['analysis']['emotion_analysis']
            print(f"\nEmotion Analysis:")
            print(f"Has emotion content: {emotion_analysis['has_emotion_content']}")
            print(f"Emotion intensity: {emotion_analysis['emotion_intensity']:.2f}")
            print(f"Confidence: {emotion_analysis['confidence']:.2f}")
            
            if emotion_analysis['analysis_details']:
                print("\nDetailed Analysis:")
                structure = emotion_analysis['analysis_details']['structure_analysis']
                print(f"Personal pronoun ratio: {structure['personal_pronoun_ratio']:.2f}")
                print(f"Emotion word ratio: {structure['emotion_word_ratio']:.2f}")
                
            print("\nNeeds more detail:", result['needs_more_input'])
            print("\nGuidance suggestion:", result['analysis']['guidance_suggestion'])
            print("\nAI Response:", result['guide_response'])
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_conversation_guide())