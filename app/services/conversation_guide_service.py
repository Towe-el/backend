import os
import numpy as np
from typing import List, Dict
from datetime import datetime
from google import genai
from .search_service import (
    _get_vertex_embedding_service,
    split_into_sentences_service
)

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

        # Overall score - adjust weights to be more lenient
        emotion_score = (
            similarities.get('high_emotion', 0) * 0.25 +  # Reduced weight of high emotion similarity
            (1 - similarities.get('neutral', 0)) * 0.25 +  # Increased weight of non-neutral degree
            structure_analysis['personal_pronoun_ratio'] * 0.2 +  # Keep personal pronouns weight
            structure_analysis['emotion_word_ratio'] * 0.2 +  # Keep emotion words weight
            structure_analysis['intensity_indicator_ratio'] * 0.1  # Keep intensity indicators weight
        )

        # Adjust emotion intensity calculation - more lenient scaling
        base_intensity = min(1.0, max(0.0, emotion_score * 2.0))  # Increased multiplier from 1.5 to 2.0
        
        # Adjust confidence calculation - more lenient
        confidence = min(1.0, max(0.0, 
            (similarities.get('high_emotion', 0) * 0.25 + 
             structure_analysis['emotion_word_ratio'] * 0.35 +  # Increased weight for emotion words
             structure_analysis['personal_pronoun_ratio'] * 0.25 +  # Increased weight for personal pronouns
             structure_analysis['intensity_indicator_ratio'] * 0.15) * 2.0  # Increased multiplier
        ))

        # Lower thresholds for emotion content determination
        has_emotion = base_intensity > 0.15  # Lowered from 0.2

        # More lenient needs_more_detail threshold
        needs_more_detail = base_intensity < 0.15 or confidence < 0.25  # Lowered from 0.2/0.3

        return {
            'has_emotion_content': has_emotion,
            'emotion_intensity': base_intensity,
            'confidence': confidence,
            'analysis_details': {
                'similarities': similarities,
                'structure_analysis': structure_analysis,
                'needs_more_detail': needs_more_detail
            }
        }

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
            
            # 4. update context
            self.context_manager.update_context(user_input, guide_response)
            
            return {
                "guidance_response": guide_response,
                "analysis": analysis,
                "needs_more_input": analysis["needs_more_detail"]
            }
        except Exception as e:
            print(f"Error in process_user_input: {str(e)}")
            return {
                "guidance_response": "I apologize, but I'm having trouble processing your input right now. Could you try expressing your thoughts in a different way?",
                "analysis": {
                    "emotion_analysis": {
                        "has_emotion_content": False,
                        "emotion_intensity": 0.0,
                        "confidence": 0.0
                    },
                    "needs_more_detail": True
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

class InputAnalyzer:
    def __init__(self):
        self.emotion_detector = EmotionDetector()

    def analyze_input(self, text: str) -> Dict:
        # Get basic analysis
        sentences = split_into_sentences_service(text)
        
        # Perform emotion analysis
        emotion_analysis = self.emotion_detector.analyze_emotion_content(text)
        
        # More lenient criteria for needs_more_detail
        needs_more_detail = (
            len(sentences) < 2 or  # Reduced from 3 to 2
            emotion_analysis['analysis_details']['needs_more_detail']
        )
        
        # Merge analysis results
        return {
            "sentence_count": len(sentences),
            "emotion_analysis": emotion_analysis,
            "needs_more_detail": needs_more_detail
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
1. Acknowledges their current emotional state in 1 or 2 sentences
2. Shows empathy and understanding
3. {'''If they need more detail, provide:
   a) 2-3 specific follow-up questions from this list:
      - "What happened that made you feel this way?"
      - "When did you start feeling like this?"
      - "How does this feeling show up in your body?"
      - "What thoughts come up with this feeling?"
      - "Did something specific trigger this emotion?"
      - "How long have you been feeling this way?"
   b) A set of emotion words they might relate to, like:
      - For sadness: "disappointed", "lonely", "hopeless", "hurt"
      - For anger: "frustrated", "irritated", "furious", "resentful"
      - For anxiety: "nervous", "overwhelmed", "worried", "tense"
      Choose words that best match their expressed emotion.''' if needs_detail else "Acknowledge that they've expressed themselves clearly"}

Keep the response concise and supportive. If suggesting words, present them as options to consider, not as declarations about their state."""

        return prompt

class ContextManager:
    def __init__(self):
        self.conversation_history = []
        self.max_history = 5

    def update_context(self, user_input: str, guidance_response: str):
        self.conversation_history.append({
            "user_input": user_input,
            "guidance_response": guidance_response,
            "timestamp": datetime.now()
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_relevant_context(self) -> List[Dict]:
        return self.conversation_history