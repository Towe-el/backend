import os
import numpy as np
import collections.abc
from typing import List, Dict, Any
from datetime import datetime
from google import genai
from .search_service import (
    _get_vertex_embedding_service,
    split_into_sentences_service
)
import json

# This threshold determines if the input has enough substance to be searchable.
QUALITY_THRESHOLD = 0.6

def _convert_numpy_types(data: Any) -> Any:
    """
    Recursively traverses a data structure and converts NumPy types to native Python types
    to ensure MongoDB compatibility.
    """
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, collections.abc.Mapping):
        return {key: _convert_numpy_types(value) for key, value in data.items()}
    if isinstance(data, collections.abc.Iterable) and not isinstance(data, str):
        return [_convert_numpy_types(item) for item in data]
    return data

class ConversationGuideService:
    """
    Analyzes user input quality using a powerful language model to determine
    if it's detailed enough for a meaningful search.
    """

    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "europe-west1")
        )
        self.model = "gemini-2.0-flash-001"
        print("ConversationGuideService initialized (Model-Based Quality Analysis).")

    async def process_user_input(
        self,
        user_text: str,
        current_accumulated_text: str = "",
        current_round: int = 0
    ) -> Dict[str, Any]:
        """
        Analyzes user input quality via a language model and provides guidance.
        """
        print(f"Processing input round {current_round + 1}...")
        
        # 1. Combine new user input with previous text.
        new_accumulated_text = f"{current_accumulated_text} {user_text}".strip()
        word_count = len(new_accumulated_text.split())

        # 2. Analyze quality using the new model-based method.
        quality_analysis = await self._analyze_quality_with_llm(new_accumulated_text)
        
        # 3. Decide if the input is ready for search (must pass LLM and word count checks).
        ready_for_search = (
            quality_analysis.get("quality_score", 0.0) >= QUALITY_THRESHOLD
            and word_count > 30
        )
        
        # 4. Generate guidance if needed.
        guidance_response = None
        if not ready_for_search:
            guidance_response = await self._generate_guidance(
                analysis_reasoning=quality_analysis.get("reasoning", "The text is too brief."),
                text_history=new_accumulated_text
            )
        else:
            guidance_response = "Your input is comprehensive and ready for analysis."

        # 5. Return the complete, updated state.
        final_result = {
            "guidance_response": guidance_response,
            "accumulated_text": new_accumulated_text,
            "input_round": current_round + 1,
            "ready_for_search": ready_for_search,
            "analysis": quality_analysis,
        }
        
        return _convert_numpy_types(final_result)

    async def _analyze_quality_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Uses a llm to perform a sophisticated analysis of the text's quality.
        """
        if not text or len(text.split()) < 10:
             return {"quality_score": 0.0, "reasoning": "The text is too short to analyze."}

        prompt = f"""
        Analyze the following user-provided text to determine its suitability for a deep emotional analysis.
        Evaluate the text based on four criteria: Emotional Depth, Contextual Completeness, Personal Perspective, and Clarity of Expression.

        TEXT TO ANALYZE:
        ---
        {text}
        ---

        Your task is to return a JSON object with two keys:
        1. "quality_score": A single float between 0.0 (very poor) and 1.0 (excellent), representing the overall quality for emotional analysis.
        2. "reasoning": A brief, one-sentence explanation for your score, highlighting the main strength or weakness.

        CRITERIA FOR SCORING:
        - **Emotional Depth (Weight: 40%)**: Does the text describe nuanced feelings, or just surface-level facts? (e.g., "I feel a hollow ache in my chest" vs. "I am sad").
        - **Contextual Completeness (Weight: 30%)**: Does the text provide background (who, what, where, when) to understand the source of the emotions?
        - **Personal Perspective (Weight: 20%)**: Is the text a personal, subjective account ("I felt," "it seemed to me"), or an objective report?
        - **Clarity of Expression (Weight: 10%)**: Is the language specific and clear, or vague and ambiguous?

        Example output for a good text:
        {{
          "quality_score": 0.85,
          "reasoning": "The text provides a clear personal perspective with deep emotional descriptions and sufficient context."
        }}

        Example output for a poor text:
        {{
          "quality_score": 0.2,
          "reasoning": "The text is a brief, factual statement lacking personal feelings and contextual details."
        }}

        Now, analyze the provided text and return ONLY the JSON object.
        """
        try:
            response = self.client.models.generate_content(model=self.model, contents=[prompt])
            # Clean and parse the JSON response
            response_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            analysis_result = json.loads(response_text)
            return analysis_result
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"Error parsing LLM quality analysis response: {e}")
            return {"quality_score": 0.0, "reasoning": "Failed to analyze text quality due to a technical issue."}
    
    async def _generate_guidance(self, analysis_reasoning: str, text_history: str) -> str:
        """
        Generates contextual guidance based on the LLM's reasoning.
        """
        prompt = f"""
        You are an empathetic AI guide helping users express their emotions and experiences more accurately.
        The user has provided the following text so far:
        "{text_history}"

        Refer to the analysis to understand why the user needs to provide more detail: "{analysis_reasoning}"

        Your approach should be:
        1. Acknowledge what they've shared with empathy (1-2 sentences)
        2. Act as a caring friend who wants to understand better

        If they need more detail, help them explore through:

        **For vague or ambiguous emotions** (like "off", "bad", "weird", "not right"):
        - Gently explore what that feeling means to them specifically
        - Offer 1-2 interpretive possibilities as questions: "When you say [their word], do you mean more like [option A], or [option B], or perhaps [option C]?"
        - Example: "When you say you feel 'off', I'm wondering if that's more like feeling disconnected from yourself, or feeling like something doesn't quite fit, or maybe feeling physically uncomfortable?"

        **For incomplete emotional context**:
        - Ask about the situation: "What was happening when this feeling started?"
        - Explore the trigger: "Was there a particular moment or interaction that brought this on?"
        - Inquire about the experience: "How does this show up for you - in your thoughts, your body, or your behavior?"

        **For emotions that need more depth**:
        - "You mentioned feeling [emotion] - can you help me understand what that looked like in this situation?"
        - "When you felt [emotion], what thoughts were going through your mind?"
        - "How long have you been carrying this feeling with you?"

        Keep your response:
        - Conversational and warm, like talking to a trusted friend
        - Focused on understanding rather than diagnosing
        - Limited to 1-2 gentle questions or clarifications
        - Avoid listing emotion words unless offering them as interpretive possibilities

        Remember: You're helping them discover their own emotional landscape, not prescribing how they should feel.
        """
        try:
            response = self.client.models.generate_content(model=self.model, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            print(f"Error generating AI guidance: {e}")
            return "I'm having a little trouble understanding. Could you try rephrasing or adding a bit more detail?"
