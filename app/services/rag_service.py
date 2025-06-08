from typing import List, Dict, Tuple
from collections import Counter
from google import genai
import os
import ast # Use ast for safe literal evaluation
import numpy as np

# Valence-Arousal (VA) model, meticulously re-calibrated based on UI/UX and a High/Medium/Low intensity principle.
# V: Valence (Pleasure), A: Arousal (Activation)
EMOTION_VA_MAP = {
    # Top-Right Quadrant (V+, A+): Positive, High Arousal
    "excitement": (0.8, 0.8), "joy": (0.8, 0.6), "love": (0.9, 0.5),
    "amusement": (0.6, 0.5), "surprise": (0.4, 0.7), "desire": (0.6, 0.6),
    "pride": (0.7, 0.5), "optimism": (0.7, 0.4), "curiosity": (0.5, 0.5),

    # Bottom-Right Quadrant (V+, A-): Positive, Low Arousal
    "gratitude": (0.8, -0.3), "caring": (0.7, -0.4), "approval": (0.6, -0.2),
    "admiration": (0.7, -0.5), "relief": (0.5, -0.5), "realization": (0.3, -0.2),

    # Top-Left Quadrant (V-, A+): Negative, High Arousal
    "anger": (-0.7, 0.8), "fear": (-0.6, 0.8), "annoyance": (-0.5, 0.6),
    "disgust": (-0.8, 0.5), "nervousness": (-0.4, 0.7),

    # Bottom-Left Quadrant (V-, A-): Negative, Low Arousal
    "sadness": (-0.8, -0.6), "grief": (-0.9, -0.7), "disappointment": (-0.7, -0.4),
    "remorse": (-0.6, -0.3), "embarrassment": (-0.5, -0.4), "disapproval": (-0.6, -0.2),
    "confusion": (-0.3, -0.1),

    # Center
    "neutral": (0.0, 0.0)
}
EMOTION_LABELS = list(EMOTION_VA_MAP.keys())

class RAGProcessor:
    def __init__(self):
        # Initialize Gemini model
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "europe-west1")
        )
        self.model = "gemini-2.0-flash-001"

    def _get_query_emotion_profile(self, text: str) -> np.ndarray | None:
        """
        Analyzes the user's input text to determine its emotional center in VA space.
        This provides a ground truth for filtering search results.
        """
        try:
            prompt = f"""
            Analyze every potential emotion in the following text. From the provided list of 28 emotions, return a JSON object where keys are the emotion labels and values are confidence scores (a float between 0.0 and 1.0), representing how likely that emotion is present.
            You must include ALL 28 emotions from the list in your response.
            Emotion List: {', '.join(EMOTION_LABELS)}
            Text to analyze: "{text}"
            Your response must be ONLY the JSON object.
            """
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt]
            )
            if not response or not response.text:
                return None

            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            emotion_scores = ast.literal_eval(response_text)
            if not isinstance(emotion_scores, dict):
                return None

            weighted_v, weighted_a, total_score = 0.0, 0.0, 0.0
            for label, score in emotion_scores.items():
                if label in EMOTION_VA_MAP and isinstance(score, (int, float)) and score > 0:
                    v, a = EMOTION_VA_MAP[label]
                    weighted_v += v * score
                    weighted_a += a * score
                    total_score += score
            
            if total_score == 0:
                return None
            
            return np.array([weighted_v / total_score, weighted_a / total_score])
        except Exception as e:
            print(f"Error getting query emotion profile: {e}")
            return None

    def process_search_results(self, search_results: List[Dict], user_input: str) -> Dict:
        """Process search results to get emotion statistics and generate empathetic responses"""
        try:
            if not isinstance(search_results, list) or not search_results:
                return {
                    "emotion_stats": [],
                    "example_responses": {}
                }

            # 1. extract and preprocess search results
            processed_results = []
            for result in search_results:
                if isinstance(result, dict):
                    processed_result = {
                        "text": str(result.get("text", "")),
                        "emotion_label": str(result.get("emotion_label", "")),
                        "score": float(result.get("score", 0.0))
                    }
                    processed_results.append(processed_result)

            if not processed_results:
                return {
                    "emotion_stats": [],
                    "example_responses": {}
                }

            # 2. calculate emotion statistics
            emotion_stats = self._calculate_emotion_stats(processed_results, user_input)
            if not emotion_stats:
                return {
                    "emotion_stats": [],
                    "example_responses": {}
                }

            # 3. generate example responses
            example_responses = self._generate_example_responses(processed_results, emotion_stats)

            return {
                "emotion_stats": emotion_stats,
                "example_responses": example_responses
            }

        except Exception as e:
            print(f"Error in process_search_results: {e}")
            return {
                "emotion_stats": [],
                "example_responses": {}
            }

    def _extract_primary_emotion(self, emotion_label) -> str:
        """Extract the primary emotion from a complex emotion label structure"""
        try:
            if isinstance(emotion_label, str):
                # Try to parse if it's a string representation of a list
                try:
                    emotion_label = ast.literal_eval(emotion_label)
                except (ValueError, SyntaxError):
                    # If it's a simple string, return it directly.
                    return emotion_label.strip()
                
            if isinstance(emotion_label, list) and emotion_label:
                # Assuming the list is already somewhat structured, e.g. from GoEmotions
                # We can try to find the one with the highest count, excluding neutral.
                valid_emotions = [
                    item for item in emotion_label 
                    if isinstance(item, dict) and 
                    item.get('tag', '').lower() not in ['neutral', 'unclear', 'none', ''] and
                    item.get('cnt', 0) > 0
                ]
                
                if valid_emotions:
                    # Sort by count descending
                    valid_emotions.sort(key=lambda x: (-x.get('cnt', 0), x.get('tag', '')))
                    return valid_emotions[0]['tag']
                
            return "" # Return empty if no valid emotion is found
        except Exception as e:
            print(f"Error extracting primary emotion: {e}")
            return ""

    def _analyze_emotion_relevance(self, user_input: str, emotion: str, context_texts: List[str]) -> float:
        """Analyze how relevant an emotion is to the user's input"""
        try:
            prompt = f"""Given this user's experience and the emotion "{emotion}", rate how relevant this emotion is on a scale of 0-10.

User's experience:
{user_input}

Consider:
1. Direct expression of {emotion}
2. Situations typically associated with {emotion}
3. Words and phrases indicating {emotion}
4. The overall emotional tone
5. The context and consequences described

Return only a number from 0-10, where:
0 = Not relevant at all
5 = Moderately relevant
10 = Extremely relevant"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt]
            )
            
            try:
                score = float(response.text.strip())
                return min(max(score, 0), 10)  # Ensure score is between 0 and 10
            except ValueError:
                print(f"Error parsing relevance score for {emotion}")
                return 0
            
        except Exception as e:
            print(f"Error analyzing emotion relevance: {e}")
            return 0

    def _calculate_emotion_stats(self, search_results: List[Dict], user_input: str) -> List[Dict]:
        """
        Calculates emotion stats using a "Quadrant Guard" followed by distance filtering.
        1. Determines the user query's emotional center and its quadrant.
        2. Aggregates all emotions from search results.
        3. "Quadrant Guard": Instantly discards any emotion not in the same or adjacent quadrant as the query's center.
        4. Calculates distance for the remaining candidates and filters the closest ones.
        """
        # 1. Get the user query's emotional profile.
        query_emotion_center = self._get_query_emotion_profile(user_input)
        if query_emotion_center is None:
            print("Could not determine query emotion profile. Proceeding without filtering.")
        else:
            print(f"Query Emotion Center (VA): {query_emotion_center}")

        # 2. Aggregate all emotions.
        emotion_data = {}
        total_cnt_sum = 0
        for result in search_results:
            raw_emotion_label = result.get("emotion_label")
            if not raw_emotion_label:
                continue
            try:
                if isinstance(raw_emotion_label, str):
                    parsed_labels = ast.literal_eval(raw_emotion_label)
                elif isinstance(raw_emotion_label, list):
                    parsed_labels = raw_emotion_label
                else:
                    continue
                if not isinstance(parsed_labels, list):
                    continue
                for label_item in parsed_labels:
                    if isinstance(label_item, dict):
                        tag = label_item.get('tag')
                        cnt = label_item.get('cnt', 0)
                        if not tag or cnt == 0:
                            continue
                        if tag not in emotion_data:
                            emotion_data[tag] = {'score': 0.0, 'count': 0, 'docs': 0}
                        emotion_data[tag]['score'] += result.get("score", 0.0)
                        emotion_data[tag]['count'] += cnt
                        emotion_data[tag]['docs'] += 1
                        total_cnt_sum += cnt
            except (ValueError, SyntaxError):
                tag = str(raw_emotion_label).strip()
                if tag:
                    if tag not in emotion_data:
                        emotion_data[tag] = {'score': 0.0, 'count': 0, 'docs': 0}
                    emotion_data[tag]['score'] += result.get("score", 0.0)
                    emotion_data[tag]['count'] += 1
                    emotion_data[tag]['docs'] += 1
                    total_cnt_sum += 1

        # 3. "Quadrant Guard" and Distance Filtering.
        if query_emotion_center is not None:
            filtered_emotion_data = {}
            query_v, query_a = query_emotion_center
            
            for tag, data in emotion_data.items():
                if tag in EMOTION_VA_MAP:
                    tag_v, tag_a = EMOTION_VA_MAP[tag]
                    
                    # Quadrant Guard: Check if signs match. We allow for some flexibility,
                    # e.g., a highly negative emotion (V<0) can be adjacent to neutral (V=0),
                    # so we check if the product of valences is non-positive (v1*v2 <= 0), not strictly positive.
                    # This filters out emotions in the completely opposite quadrant.
                    is_opposite_valence = query_v * tag_v < -0.01 # Check for clearly opposite signs
                    is_opposite_arousal = query_a * tag_a < -0.01

                    if is_opposite_valence and is_opposite_arousal:
                         print(f"Quadrant Guard: Filtering out '{tag}' (opposite quadrant).")
                         continue

                    # Distance Filtering for remaining candidates
                    tag_vector = np.array(EMOTION_VA_MAP[tag])
                    distance = np.linalg.norm(tag_vector - query_emotion_center)
                    DISTANCE_THRESHOLD = 1.0 # Stricter threshold now
                    if distance <= DISTANCE_THRESHOLD:
                        filtered_emotion_data[tag] = data
                    else:
                        print(f"Distance Filter: Filtering out '{tag}' (distance: {distance:.2f} > {DISTANCE_THRESHOLD})")
                else: # If emotion is not in our map, keep it by default
                    filtered_emotion_data[tag] = data
            emotion_data = filtered_emotion_data

        if not emotion_data:
            return []

        # 4. Calculate final weighted scores for the relevant emotions.
        final_scores = []
        for tag, data in emotion_data.items():
            avg_similarity = data['score'] / data['docs'] if data['docs'] > 0 else 0
            cnt_weight = data['count'] / total_cnt_sum if total_cnt_sum > 0 else 0
            final_score = avg_similarity * 0.7 + cnt_weight * 0.3
            final_scores.append({
                "label": tag,
                "score": final_score,
                "count": data['count']
            })

        # 5. Sort by final score and take top 3.
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        top_emotions = final_scores[:3]

        # 6. Calculate percentage based on the COUNT of the top emotions.
        total_top_count = sum(e["count"] for e in top_emotions)
        emotion_stats = []
        for emotion in top_emotions:
            percentage = (emotion["count"] / total_top_count) * 100 if total_top_count > 0 else 0
            emotion_stats.append({
                "label": emotion["label"],
                "count": emotion["count"],
                "percentage": percentage
            })

        return emotion_stats

    def _generate_example_responses(self, search_results: List[Dict], emotion_stats: List[Dict]) -> Dict:
        """Generate example responses for each emotion"""
        try:
            example_responses = {}
            
            # Generate responses for each emotion label in the corrected stats
            for emotion_info in emotion_stats:
                emotion = emotion_info.get("label")
                if not emotion:
                    continue

                # Find relevant texts for this emotion from the original results
                relevant_texts = []
                for result in search_results:
                    primary_emotion = self._extract_primary_emotion(result.get("emotion_label"))
                    if primary_emotion == emotion:
                        relevant_texts.append(str(result.get("text", "")))
                
                # Limit to top 5 relevant texts to keep the prompt concise
                relevant_texts = relevant_texts[:5]

                # Generate a prompt to create an empathetic response
                prompt = f"""
Given that a user is expressing feelings related to "{emotion}", and has shared these experiences:
- "{', '.join(relevant_texts)}"

Please provide a short, empathetic, and encouraging response. The response should:
- Acknowledge the user's feelings without being repetitive.
- Offer gentle encouragement or a supportive perspective.
- Be phrased as if you are speaking directly to the user.
- Do NOT ask questions.
- Keep it to 1-2 sentences.

Example for "sadness": "It sounds like you're going through a really tough time. Remember to be kind to yourself as you navigate these feelings."

Your response:
"""
                try:
                    # Call Gemini to generate the response
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[prompt]
                    )
                    example_responses[emotion] = response.text.strip()
                except Exception as e:
                    print(f"Error generating Gemini response for {emotion}: {e}")
                    example_responses[emotion] = f"It's valid to feel {emotion}. Acknowledging this is a brave step."

            return example_responses

        except Exception as e:
            print(f"Error in _generate_example_responses: {e}")
            return {} 