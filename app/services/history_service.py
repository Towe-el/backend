import os
import json
from typing import Optional
from google import genai


class HistoryService:
    """
    Service for generating concise titles for user input history.
    When input is ready for search, generates a brief summary title.
    """

    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "europe-west1")
        )
        self.model = "gemini-2.0-flash-001"
        print("HistoryService initialized for title generation.")

    async def generate_title(self, accumulated_text: str) -> Optional[str]:
        """
        Generate a concise title (≤10 words) for the user's accumulated text.
        
        Args:
            accumulated_text: The user's complete input text
            
        Returns:
            A brief title string, or None if generation fails
        """
        if not accumulated_text or not accumulated_text.strip():
            return None

        prompt = f"""
        Based on the following user input, generate a very concise title that captures the main emotional theme or situation. The title should be:
        - Maximum 10 words
        - Capture the essence of the user's emotional state or main concern
        - Use simple, clear language
        - First person perspective

        USER INPUT:
        ---
        {accumulated_text.strip()}
        ---

        Return ONLY the title, nothing else. Examples of good titles:
        - "Feeling overwhelmed by work pressure"
        - "Struggling with friendship conflict"
        - "Anxiety about upcoming life changes"
        - "Processing grief after loss"

        Generate the title now:
        """

        try:
            response = self.client.models.generate_content(
                model=self.model, 
                contents=[prompt]
            )
            
            # Clean the response
            title = response.text.strip()
            
            # Remove quotes if present
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            if title.startswith("'") and title.endswith("'"):
                title = title[1:-1]
            
            # Ensure it's not too long (backup check)
            words = title.split()
            if len(words) > 10:
                title = " ".join(words[:10])
            
            return title
            
        except Exception as e:
            print(f"Error generating title: {e}")
            return "User conversation session"  # Fallback title 