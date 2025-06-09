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

        # Overall score - 调整权重使其更严格
        emotion_score = (
            similarities.get('high_emotion', 0) * 0.3 +  # 增加高情感相似度权重
            (1 - similarities.get('neutral', 0)) * 0.2 +  # 减少非中性权重
            structure_analysis['personal_pronoun_ratio'] * 0.2 +  
            structure_analysis['emotion_word_ratio'] * 0.2 +  
            structure_analysis['intensity_indicator_ratio'] * 0.1  
        )

        # 更严格的情感强度计算
        base_intensity = min(1.0, max(0.0, emotion_score * 2.0))
        
        # 更严格的置信度计算
        confidence = min(1.0, max(0.0, 
            (similarities.get('high_emotion', 0) * 0.3 + 
             structure_analysis['emotion_word_ratio'] * 0.3 +  
             structure_analysis['personal_pronoun_ratio'] * 0.2 +  
             structure_analysis['intensity_indicator_ratio'] * 0.2) * 1.5
        ))

        # 更严格的情感内容判断
        has_emotion = base_intensity > 0.25

        # 更严格的needs_more_detail阈值
        needs_more_detail = base_intensity < 0.35 or confidence < 0.3

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
        
        # 添加临时输入存储（会话级别，刷新后释放）
        self.accumulated_input = []  # 存储用户在当前会话中的所有输入
        self.session_start_time = datetime.now()

    def clear_accumulated_input(self):
        """清空累积的用户输入（用于开始新的会话）"""
        print(f"Clearing accumulated input. Previous inputs: {self.accumulated_input}")
        self.accumulated_input = []
        self.session_start_time = datetime.now()

    def get_accumulated_text(self) -> str:
        """获取累积的所有用户输入文本"""
        return " ".join(self.accumulated_input)

    async def process_user_input(self, user_input: str) -> Dict:
        try:
            # 添加当前输入到累积存储中
            self.accumulated_input.append(user_input.strip())
            
            # 调试日志：打印累积的输入
            print(f"Session start time: {self.session_start_time}")
            print(f"Current input: '{user_input}'")
            print(f"Accumulated inputs: {self.accumulated_input}")
            print(f"Combined accumulated text: '{self.get_accumulated_text()}'")
            
            # 1. 分析累积的用户输入（而不是单独分析当前输入）
            accumulated_text = self.get_accumulated_text()
            analysis = self.input_analyzer.analyze_input(accumulated_text)
            
            print(f"Analysis result: {analysis}")
            
            # 2. get relevant context
            context = self.context_manager.get_relevant_context()
            
            # 3. 根据质量检测结果决定是否生成引导
            guidance_response = None
            if analysis["needs_more_detail"]:
                # 只有在需要更多细节时才生成引导响应
                guidance_response = await self._generate_ai_guide(
                    user_input,  # 当前输入
                    accumulated_text,  # 累积输入
                    analysis, 
                    context,
                    len(self.accumulated_input)  # 输入轮次
                )
            else:
                # 质量检测通过，不需要引导
                print("Quality check passed - no guidance needed")
                guidance_response = "Your input is comprehensive and ready for analysis. You can now proceed to search for similar experiences and get detailed insights."
            
            # 4. update context (只有在需要更多输入时才不更新正式的对话历史)
            if not analysis["needs_more_detail"]:
                # 质量检测通过，将累积的内容作为完整输入记录到对话历史
                self.context_manager.update_context(accumulated_text, guidance_response)
                # 注意：这里不清空累积输入，等待用户主动触发搜索后再清空
                print(f"Quality check passed. Ready for search. Accumulated input: '{accumulated_text}'")
            
            return {
                "guidance_response": guidance_response,
                "analysis": analysis,
                "needs_more_input": analysis["needs_more_detail"],
                "accumulated_text": accumulated_text,  # 返回累积文本供前端参考
                "input_round": len(self.accumulated_input) if analysis["needs_more_detail"] else 0,
                "ready_for_search": not analysis["needs_more_detail"]  # 新增：表示是否准备好搜索
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
                "needs_more_input": True,
                "accumulated_text": self.get_accumulated_text(),
                "input_round": len(self.accumulated_input),
                "ready_for_search": False
            }

    async def _generate_ai_guide(
        self, 
        current_input: str,
        accumulated_text: str,
        analysis: Dict, 
        context: List[Dict],
        input_round: int
    ) -> str:
        # build prompt template
        prompt = self.prompt_generator.create_prompt(
            current_input,
            accumulated_text,
            analysis, 
            context,
            input_round
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
        
        # 更严格的综合质量判断
        word_count = len(text.split())
        
        # 基本要求：至少3句话且30个词
        basic_length_sufficient = len(sentences) >= 3 and word_count >= 30
        
        # 情感分析要求
        emotion_sufficient = not emotion_analysis['analysis_details']['needs_more_detail']
        
        # 综合判断：两个条件都要满足
        needs_more_detail = not (basic_length_sufficient and emotion_sufficient)
        
        print(f"Quality analysis details:")
        print(f"  - sentences: {len(sentences)} (need >=3)")
        print(f"  - words: {word_count} (need >=30)")
        print(f"  - basic_length_sufficient: {basic_length_sufficient}")
        print(f"  - emotion_sufficient: {emotion_sufficient}")
        print(f"  - final needs_more_detail: {needs_more_detail}")
        
        # Merge analysis results
        return {
            "sentence_count": len(sentences),
            "emotion_analysis": emotion_analysis,
            "needs_more_detail": needs_more_detail
        }

class PromptGenerator:
    def create_prompt(
        self, 
        current_input: str,
        accumulated_text: str,
        analysis: Dict, 
        context: List[Dict],
        input_round: int
    ) -> str:
        # Build prompt based on analysis
        emotion_analysis = analysis["emotion_analysis"]
        has_emotion = emotion_analysis["has_emotion_content"]
        intensity = emotion_analysis["emotion_intensity"]
        needs_detail = analysis["needs_more_detail"]
        
        # 根据输入轮次调整引导策略
        if input_round == 1:
            # 第一次输入，常规引导
            guidance_strategy = "This is the user's first input in this session."
        else:
            # 后续输入，需要递进式引导
            guidance_strategy = f"This is the user's {input_round} input in this session. They have previously shared: '{accumulated_text.replace(current_input.strip(), '').strip()}'. Focus on what's still missing rather than repeating previous guidance."

        prompt = f"""As an empathetic AI guide, analyze the user's input in this session:

Current input: "{current_input}"
All accumulated input: "{accumulated_text}"

Analysis context:
- Emotional content: {"Present" if has_emotion else "Limited"}
- Emotional intensity: {intensity:.2f}
- Needs more detail: {"Yes" if needs_detail else "No"}
- {guidance_strategy}

Please provide a response that:
1. Acknowledges their current sharing (1-2 sentences)
2. Shows empathy and understanding

{"3. Since this is not their first input, focus on what specific aspects are still missing. Avoid repeating guidance for information they've already provided. Instead, identify what gaps remain and guide them towards those specific areas." if input_round > 1 else "3. "}
{'''If they need more detail, provide targeted follow-up questions. Choose 2-3 specific questions that address what's missing:
   
   For incomplete emotional expression:
   - "What happened that made you feel this way?"
   - "When did you start feeling like this?"
   - "How does this feeling show up in your body?"
   - "What thoughts come up with this feeling?"
   - "Did something specific trigger this emotion?"
   - "How long have you been feeling this way?"
   
   For insufficient context:
   - "Can you tell me more about the situation?"
   - "What led up to this moment?"
   - "How is this affecting your daily life?"
   
   Also provide emotion words they might relate to:
   - For sadness: "disappointed", "lonely", "hopeless", "hurt", "discouraged"
   - For anger: "frustrated", "irritated", "furious", "resentful", "betrayed"
   - For anxiety: "nervous", "overwhelmed", "worried", "tense", "uncertain"
   - For joy: "excited", "content", "grateful", "proud", "fulfilled"
   
   Choose words that best match their expressed emotion.''' if needs_detail else "Acknowledge that they've expressed themselves clearly and their input is sufficient."}

{"Focus on building upon what they've already shared rather than starting over. Be progressive and specific about what additional information would be most helpful." if input_round > 1 else ""}

Keep the response concise, supportive, and tailored to their current level of sharing."""

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