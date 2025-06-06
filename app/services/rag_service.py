from typing import List, Dict, Tuple
from collections import Counter
from google import genai
import os

class RAGProcessor:
    def __init__(self):
        # Initialize Gemini model
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "europe-west1")
        )
        self.model = "gemini-2.0-flash-001"

    def process_search_results(self, search_results: List[Dict]) -> Dict:
        """Process search results to get emotion statistics and generate empathetic responses"""
        # 1. Calculate emotion label statistics
        emotion_stats = self._calculate_emotion_stats(search_results)
        if not emotion_stats:
            return {
                "emotion_stats": [],
                "example_responses": {}
            }

        # 2. Generate example responses based on similar experiences
        example_responses = self._generate_example_responses(search_results, emotion_stats)

        return {
            "emotion_stats": emotion_stats,
            "example_responses": example_responses
        }

    def _calculate_emotion_stats(self, search_results: List[Dict]) -> List[Dict]:
        """Calculate statistics for emotion labels"""
        # Count emotion labels, excluding 'neutral' and 'unclear'
        emotion_counter = Counter()
        for result in search_results:
            label = result.get("emotion_label")
            if label and label not in ["neutral", "unclear"]:
                emotion_counter[label] += 1

        if not emotion_counter:
            return []

        # Get top 3 emotions
        top_emotions = emotion_counter.most_common(3)
        total_count = sum(count for _, count in top_emotions)

        # Calculate percentages
        emotion_stats = [
            {
                "label": label,
                "count": count,
                "percentage": round(count / total_count * 100, 2)
            }
            for label, count in top_emotions
        ]

        return emotion_stats

    def _generate_example_responses(self, search_results: List[Dict], emotion_stats: List[Dict]) -> Dict:
        """Generate example responses for each top emotion"""
        example_responses = {}
        
        for emotion in emotion_stats:
            # Filter texts by current emotion
            relevant_texts = [
                result["text"] 
                for result in search_results 
                if result.get("emotion_label") == emotion["label"]
            ]
            
            if not relevant_texts:
                continue

            # Generate empathetic response based on similar experiences
            prompt = f"""Based on these similar experiences from other people who felt {emotion["label"]}:

{relevant_texts[:3]}  # Using up to 3 examples to keep context manageable

Create a single sentence that:
1. Expresses a similar feeling or experience
2. Uses natural, first-person perspective
3. Focuses on the emotional aspect of {emotion["label"]}
4. Is relatable and authentic
5. Avoids any advice or suggestions

The sentence should help readers feel that others understand and share their emotions."""

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt]
                )
                example_responses[emotion["label"]] = response.text.strip()
            except Exception as e:
                print(f"Error generating example response for {emotion['label']}: {e}")
                example_responses[emotion["label"]] = ""

        return example_responses 