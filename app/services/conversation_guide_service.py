import os
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from google import genai
from google.genai import types
from .search_service import (
    _get_vertex_embedding_service,
    split_into_sentences_service
)
import json
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        # Pre-defined emotion benchmarks and their embeddings
        self.emotion_benchmarks = self._initialize_emotion_benchmarks()
        # Personal pronouns list
        self.personal_pronouns = {
            'i', 'me', 'my', 'mine', 'myself',
            'we', 'us', 'our', 'ours', 'ourselves'
        }
        # Emotion word roots
        self.emotion_word_roots = {
            'happy': ['happy', 'joy', 'delighted', 'excited', 'pleased'],
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'down'],
            'angry': ['angry', 'furious', 'mad', 'irritated', 'annoyed'],
            'afraid': ['afraid', 'scared', 'fearful', 'anxious', 'worried'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
            'disgusted': ['disgusted', 'repulsed', 'revolted'],
            'neutral': ['okay', 'fine', 'normal', 'alright']
        }
        # Intensity indicators
        self.intensity_indicators = {
            'very', 'really', 'extremely', 'deeply', 'strongly',
            'so', 'too', 'absolutely', 'completely', 'totally',
            'always', 'never', 'cant', 'cannot', 'couldnt',
            'must', 'have to', 'need to'
        }

    def _initialize_emotion_benchmarks(self) -> Dict[str, List[float]]:
        """Initialize emotion benchmarks and their embeddings"""
        benchmark_sentences = {
            'high_emotion': [
                "I feel absolutely devastated and heartbroken",
                "I am incredibly happy and overjoyed",
                "I am extremely angry and frustrated",
                "I am terribly anxious and worried"
            ],
            'neutral': [
                "The weather is okay today",
                "I went to the store",
                "Things are fine",
                "It's a normal day"
            ],
            'personal_experience': [
                "I feel that",
                "This makes me",
                "I experienced",
                "In my opinion"
            ]
        }

        benchmarks = {}
        for category, sentences in benchmark_sentences.items():
            embeddings = [_get_vertex_embedding_service(sent) for sent in sentences]
            if embeddings and all(emb for emb in embeddings):
                benchmarks[category] = np.mean(embeddings, axis=0).tolist()

        return benchmarks

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _analyze_sentence_structure(self, text: str) -> Dict[str, float]:
        """Analyze sentence structure features"""
        words = text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return {
                'personal_pronoun_ratio': 0.0,
                'emotion_word_ratio': 0.0,
                'intensity_indicator_ratio': 0.0,
                'avg_word_length': 0.0
            }

        # Calculate personal pronoun ratio
        personal_pronoun_count = sum(1 for word in words if word in self.personal_pronouns)
        
        # Calculate emotion word ratio
        emotion_word_count = 0
        for emotion_words in self.emotion_word_roots.values():
            emotion_word_count += sum(1 for word in words if any(ew in word for ew in emotion_words))

        # Calculate intensity indicator ratio
        intensity_indicator_count = sum(1 for word in words if word in self.intensity_indicators)
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / total_words

        return {
            'personal_pronoun_ratio': personal_pronoun_count / total_words,
            'emotion_word_ratio': emotion_word_count / total_words,
            'intensity_indicator_ratio': intensity_indicator_count / total_words,
            'avg_word_length': avg_word_length
        }

    def analyze_emotion_content(self, text: str) -> Dict:
        """Main analysis function, return emotion content analysis result"""
        # Get text embedding
        text_embedding = _get_vertex_embedding_service(text)
        if not text_embedding:
            return {
                'has_emotion_content': False,
                'emotion_intensity': 0.0,
                'confidence': 0.0,
                'analysis_details': None
            }

        # Calculate similarity with each benchmark
        similarities = {
            category: self._calculate_cosine_similarity(text_embedding, benchmark_embedding)
            for category, benchmark_embedding in self.emotion_benchmarks.items()
        }

        # Analyze sentence structure
        structure_analysis = self._analyze_sentence_structure(text)

        # Overall score - adjust weights and calculation method
        emotion_score = (
            similarities.get('high_emotion', 0) * 0.3 +  # Lower the weight of high emotion similarity
            (1 - similarities.get('neutral', 0)) * 0.2 +  # Lower the weight of non-neutral degree
            structure_analysis['personal_pronoun_ratio'] * 0.2 +  # Increase the weight of personal pronouns
            structure_analysis['emotion_word_ratio'] * 0.2 +  # Increase the weight of emotion words
            structure_analysis['intensity_indicator_ratio'] * 0.1  # Add weight to intensity indicators
        )

        # Adjust emotion intensity calculation
        base_intensity = min(1.0, max(0.0, emotion_score * 1.5))  # Increase emotion intensity
        
        # Adjust confidence calculation
        confidence = min(1.0, max(0.0, 
            (similarities.get('high_emotion', 0) * 0.3 + 
             structure_analysis['emotion_word_ratio'] * 0.3 +
             structure_analysis['personal_pronoun_ratio'] * 0.2 +
             structure_analysis['intensity_indicator_ratio'] * 0.2) * 1.5
        ))

        # Lower the threshold for emotion content determination
        has_emotion = base_intensity > 0.2  # From 0.3 to 0.2

        return {
            'has_emotion_content': has_emotion,
            'emotion_intensity': base_intensity,
            'confidence': confidence,
            'analysis_details': {
                'similarities': similarities,
                'structure_analysis': structure_analysis,
                'needs_more_detail': base_intensity < 0.2 or confidence < 0.3
            }
        }

    def get_emotion_guidance(self, analysis_result: Dict) -> str:
        """Based on the analysis result, generate a guidance suggestion"""
        if not analysis_result['has_emotion_content']:
            if analysis_result['emotion_intensity'] < 0.2:
                return "Could you share more about how you're feeling?"
            else:
                return "I sense some emotions in your words. Could you tell me more about that?"
        
        if analysis_result['confidence'] < 0.3:
            return "What specific situation made you feel this way?"
        
        if analysis_result['analysis_details']['structure_analysis']['personal_pronoun_ratio'] < 0.1:
            return "How does this situation affect you personally?"

        return "Could you elaborate more on these feelings?"

class ConversationGuideService:
    def __init__(self):
        # Initialize Gemini model
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "europe-west1")
        )
        self.model = "gemini-2.0-flash-001"
        self.context_manager = ContextManager()
        self.input_analyzer = InputAnalyzer()
        self.prompt_generator = PromptGenerator()

    async def process_user_input(self, user_input: str) -> Dict:
        try:
            # 1. analyze user input
            analysis = self.input_analyzer.analyze_input(user_input)
            
            # 2. get relevant context
            context = self.context_manager.get_relevant_context()
            
            # 3. generate guide prompt
            guide_response = await self._generate_ai_guide(
                user_input, 
                analysis, 
                context
            )

            # 4. generate concise response
            concise_response = self.generate_ai_response(
                user_input,
                analysis["emotion_analysis"]
            )
            
            # 5. update context
            self.context_manager.update_context(user_input, guide_response)
            
            return {
                "guide_response": guide_response,
                "concise_response": concise_response,
                "analysis": analysis,
                "needs_more_input": analysis["needs_more_detail"]
            }
        except Exception as e:
            print(f"Error in process_user_input: {str(e)}")
            return {
                "guide_response": "I apologize, but I'm having trouble processing your input right now. Could you try expressing your thoughts in a different way?",
                "concise_response": {
                    "response": "I understand how you feel.",
                    "guidance_suggestion": ""
                },
                "analysis": {
                    "emotion_analysis": {
                        "has_emotion_content": False,
                        "emotion_intensity": 0.0,
                        "confidence": 0.0,
                        "analysis_details": {"needs_more_detail": True}
                    },
                    "needs_more_detail": True,
                    "guidance_suggestion": "Could you try expressing your thoughts in a different way?"
                },
                "needs_more_input": True
            }

    async def _generate_ai_guide(
        self, 
        user_input: str, 
        analysis: Dict, 
        context: List[Dict]
    ) -> str:
        # build prompt template
        prompt = self.prompt_generator.create_prompt(
            user_input, 
            analysis, 
            context
        )
        
        try:
            # Use Gemini to generate response
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "You are an empathetic AI guide helping users express their emotions and experiences more accurately.",
                    prompt
                ]
            )
            return response.text
        except Exception as e:
            print(f"Error generating AI guide response: {e}")
            return "I apologize, but I'm having trouble providing a response right now. Could you try expressing your thoughts in a different way?"

    def generate_ai_response(self, user_input: str, emotion_analysis: Dict) -> Dict[str, Any]:
        """Generate AI response"""
        try:
            # Build prompt
            json_format = '''
{
    "response": "Your single sentence response here",
    "guidance_suggestion": "Optional example sentence for user to express feelings"
}
'''
            prompt = f'''You are a supportive AI assistant. The user said: "{user_input}"

Based on emotion analysis:
- Emotion Intensity: {emotion_analysis['emotion_intensity']}
- Confidence: {emotion_analysis['confidence']}

Your task is to provide a brief, empathetic response with these rules:
1. Keep your main response to a single sentence that shows understanding and support
2. If needed, provide ONE example sentence that the user could use to express their feelings more clearly
3. Be direct and concise
4. Do not give advice unless specifically asked
5. Focus on acknowledging emotions

Format your response EXACTLY as the following JSON (no additional quotes, backticks, or markdown):
{json_format}'''

            # Get AI response using the correct model
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "You are a supportive AI assistant focused on providing brief, empathetic responses.",
                    prompt
                ]
            )
            response_text = response.text.strip()

            # Clean up the response text
            # Remove any markdown code block indicators
            response_text = response_text.replace('```json', '').replace('```', '')
            # Remove any leading/trailing whitespace and quotes
            response_text = response_text.strip('`\'" \n')
            
            logger.info(f"Cleaned response text: {response_text}")

            # Parse JSON response
            try:
                response_dict = json.loads(response_text)
                # Ensure the response has the required fields
                if not isinstance(response_dict, dict) or 'response' not in response_dict:
                    raise json.JSONDecodeError("Invalid response format", response_text, 0)
                    
                return {
                    'response': response_dict.get('response', '').strip(),
                    'guidance_suggestion': response_dict.get('guidance_suggestion', '').strip()
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {response_text}")
                # Try to extract response and guidance_suggestion using regex
                import re
                try:
                    response_match = re.search(r'"response":\s*"([^"]+)"', response_text)
                    guidance_match = re.search(r'"guidance_suggestion":\s*"([^"]+)"', response_text)
                    
                    return {
                        'response': response_match.group(1) if response_match else "I understand how you feel.",
                        'guidance_suggestion': guidance_match.group(1) if guidance_match else ""
                    }
                except Exception as regex_error:
                    logger.error(f"Regex extraction failed: {str(regex_error)}")
                    return {
                        'response': "I understand how you feel.",
                        'guidance_suggestion': ""
                    }

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return {
                'response': "I understand how you feel.",
                'guidance_suggestion': ""
            }

class InputAnalyzer:
    def __init__(self):
        self.emotion_detector = EmotionDetector()

    def analyze_input(self, text: str) -> Dict:
        # Get basic analysis
        sentences = split_into_sentences_service(text)
        
        # Perform emotion analysis
        emotion_analysis = self.emotion_detector.analyze_emotion_content(text)
        
        # Get guidance suggestion
        guidance_suggestion = self.emotion_detector.get_emotion_guidance(emotion_analysis)
        
        # Merge analysis results
        return {
            "sentence_count": len(sentences),
            "emotion_analysis": emotion_analysis,
            "needs_more_detail": (
                len(sentences) < 3 or 
                emotion_analysis['analysis_details']['needs_more_detail']
            ),
            "guidance_suggestion": guidance_suggestion
        }

class PromptGenerator:
    def create_prompt(
        self, 
        user_input: str, 
        analysis: Dict, 
        context: List[Dict]
    ) -> str:
        # Build prompt based on analysis
        emotion_analysis = analysis["emotion_analysis"]
        has_emotion = emotion_analysis["has_emotion_content"]
        intensity = emotion_analysis["emotion_intensity"]
        needs_detail = analysis["needs_more_detail"]
        
        prompt = f"""As an empathetic AI guide, analyze the following user input:
"{user_input}"

Based on the analysis:
- Emotional content: {"Present" if has_emotion else "Limited"}
- Emotional intensity: {intensity:.2f}
- Needs more detail: {"Yes" if needs_detail else "No"}

Please provide a response that:
1. Acknowledges their current emotional state
2. Shows empathy and understanding
3. {"Encourages deeper emotional expression" if needs_detail else "Explores the implications of their feelings"}
4. Offers gentle guidance for reflection

Keep the response natural, supportive, and focused on their emotional experience."""

        return prompt

class ContextManager:
    def __init__(self):
        self.conversation_history = []
        self.max_history = 5

    def update_context(self, user_input: str, ai_response: str):
        self.conversation_history.append({
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": datetime.now()
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_relevant_context(self) -> List[Dict]:
        return self.conversation_history