from typing import List, Dict, Tuple
from collections import Counter
from google import genai
import os
import ast # Use ast for safe literal evaluation
import numpy as np

# Official emotion definitions, to be provided to the user.
EMOTION_DEFINITIONS = {
    "admiration": "Admiration is the feeling of finding something impressive or worthy of respect.",
    "amusement": "Amusement is the feeling of finding something funny or being entertained.",
    "anger": "Anger is a strong feeling of displeasure or antagonism.",
    "annoyance": "Annoyance is a feeling of mild anger or irritation.",
    "approval": "Approval is the feeling of having or expressing a favorable opinion.",
    "caring": "Caring is the act of displaying kindness and concern for others.",
    "confusion": "Confusion is a state of uncertainty or a lack of understanding.",
    "curiosity": "Curiosity is a strong desire to know or learn something.",
    "desire": "Desire is a strong feeling of wanting something or wishing for something to happen.",
    "disappointment": "Disappointment is a feeling of sadness or displeasure caused by the nonfulfillment of one's hopes or expectations.",
    "disapproval": "Disapproval is the feeling of having or expressing an unfavorable opinion.",
    "disgust": "Disgust is a feeling of revulsion or strong disapproval aroused by something unpleasant or offensive.",
    "embarrassment": "Embarrassment is a feeling of self-consciousness, shame, or awkwardness.",
    "excitement": "Excitement is a feeling of great enthusiasm and eagerness.",
    "fear": "Fear is the feeling of being afraid or worried.",
    "gratitude": "Gratitude is a feeling of thankfulness and appreciation.",
    "grief": "Grief is intense sorrow, especially when caused by someone's death.",
    "joy": "Joy is a feeling of great pleasure and happiness.",
    "love": "Love is a strong positive emotion of regard and affection.",
    "nervousness": "Nervousness is a feeling of apprehension, worry, or anxiety.",
    "optimism": "Optimism is a feeling of hopefulness and confidence about the future or the success of something.",
    "pride": "Pride is a feeling of pleasure or satisfaction due to one's own achievements or the achievements of those one is closely associated with.",
    "realization": "Realization is the act of becoming aware of something.",
    "relief": "Relief is a feeling of reassurance and relaxation following release from anxiety or distress.",
    "remorse": "Remorse is a feeling of regret or guilt.",
    "sadness": "Sadness is a feeling of emotional pain or sorrow.",
    "surprise": "Surprise is a feeling of being astonished or startled by something unexpected.",
    "neutral": "Neutral is a state of lacking strong emotional content."
}

# Valence-Arousal (VA) model, meticulously re-calibrated based on UI/UX and a High/Medium/Low intensity principle.
# V: Valence (Pleasure), A: Arousal (Activation)
EMOTION_VA_MAP = {
    # Top-Right Quadrant (V+, A+): Positive, High Arousal
    "excitement": (0.8, 0.8), "joy": (0.8, 0.6), "love": (0.9, 0.5),
    "amusement": (0.6, 0.5), "surprise": (0.4, 0.7), "desire": (0.6, 0.6),
    "pride": (0.7, 0.5), "curiosity": (0.5, 0.5),

    # Bottom-Right Quadrant (V+, A-): Positive, Low Arousal
    "gratitude": (0.8, -0.3), "caring": (0.7, -0.4), "approval": (0.6, -0.3),
    "admiration": (0.7, -0.5), "relief": (0.5, -0.5), "realization": (0.3, -0.2),
    "optimism": (0.7, -0.4),

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
        """
        Processes search results to generate a rich, analytical, and supportive response.
        """
        # 1. Calculate the top 3 relevant emotions based on the search results.
        top_emotions_stats = self._calculate_emotion_stats(search_results, user_input)
        if not top_emotions_stats:
            return { "enriched_emotion_stats": [], "summary_report": "" }

        # 2. For each of the top emotions, generate enriched details.
        enriched_emotion_stats = []
        for emotion_stat in top_emotions_stats:
            enriched_data = self._generate_enriched_emotion_data(
                emotion_stat, 
                user_input, 
                search_results
            )
            enriched_emotion_stats.append(enriched_data)

        # 3. Generate a final summary report based on all enriched data.
        summary_report = self._generate_summary_report(enriched_emotion_stats, user_input)

        return {
            "enriched_emotion_stats": enriched_emotion_stats,
            "summary_report": summary_report
        }

    def _generate_enriched_emotion_data(self, emotion_stat: Dict, user_input: str, search_results: List[Dict]) -> Dict:
        """
        Generates the detailed analysis for a single emotion.
        """
        label = emotion_stat["label"]
        
        # 1. Get Definition
        definition = EMOTION_DEFINITIONS.get(label, "No definition available.")

        # 2. Get Quote (formerly example_response)
        quote = self._generate_quote(label, search_results)
        
        # 3. Get Analysis
        analysis = self._generate_analysis(label, definition, user_input)
        
        # 4. Check for high-intensity negative emotions to add advice
        query_emotion_center = self._get_query_emotion_profile(user_input)
        if query_emotion_center is not None:
            is_negative = query_emotion_center[0] < -0.1
            intensity = np.linalg.norm(query_emotion_center)
            if is_negative and intensity > 0.75: # High intensity negative emotion
                analysis += " It's important to acknowledge the weight of these feelings. If they persist or feel overwhelming, speaking with a mental health professional can provide valuable support and guidance."

        return {
            "label": label,
            "count": emotion_stat["count"],
            "percentage": emotion_stat["percentage"],
            "definition": definition,
            "quote": quote,
            "analysis": analysis
        }

    def _generate_quote(self, emotion: str, search_results: List[Dict]) -> str:
        """
        Generates a short, empathetic quote for a single emotion.
        """
        try:
            relevant_texts = [
                str(r.get("text", "")) for r in search_results 
                if self._extract_primary_emotion(r.get("emotion_label")) == emotion
            ][:5]

            prompt = f"""
            Given that a user is expressing feelings related to "{emotion}", and has shared experiences like: "{', '.join(relevant_texts)}".
            Please provide a short, empathetic, and encouraging response (1-2 sentences).
            Acknowledge the feeling, offer gentle support, and speak directly to the user. Do not ask questions.
            Example for "sadness": "It sounds like you're going through a really tough time. Remember to be kind to yourself as you navigate these feelings."
            """
            response = self.client.models.generate_content(model=self.model, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            print(f"Error generating quote for {emotion}: {e}")
            return f"It's valid to feel {emotion}. Acknowledging this is a brave step."

    def _generate_analysis(self, label: str, definition: str, user_input: str) -> str:
        """
        Generates a counselor-style analysis for a single emotion.
        """
        try:
            prompt = f"""
            As a psychological counselor, analyze why the user's text might contain the emotion '{label}'.
            The definition of '{label}' is: "{definition}".
            User's text: "{user_input}"
            
            Your task:
            1. Write an analysis of 4-5 sentences.
            2. Start by validating the user's feelings.
            3. Connect specific phrases or situations from the user's text to the definition of the emotion.
            4. Explain how these elements contribute to the feeling of '{label}'.
            5. Maintain a supportive and professional tone.
            
            Example for 'sadness': "It's completely understandable that you're feeling a sense of sadness. You mentioned being late, being singled out by the teacher, and the heavy feeling of 'everyone's eyes' on you. These experiences align with the core of sadness, which is often rooted in feelings of loss—in this case, a loss of control and a sense of belonging in that moment. The rainy day you described can also mirror and amplify this internal state of emotional pain and sorrow."
            """
            response = self.client.models.generate_content(model=self.model, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            print(f"Error generating analysis for {label}: {e}")
            return "Analyzing this feeling is the first step toward understanding it better."

    def _generate_summary_report(self, enriched_stats: List[Dict], user_input: str) -> str:
        """
        Generates a final, comprehensive summary report for the user.
        """
        try:
            analysis_summary = "\n".join([f"- For '{s['label']}': {s['analysis']}" for s in enriched_stats])
            
            # Determine overall sentiment for tailoring suggestions
            query_emotion_center = self._get_query_emotion_profile(user_input)
            sentiment_type = "negative" if query_emotion_center is not None and query_emotion_center[0] < 0 else "positive"

            prompt = f"""
            You are an empathetic AI assistant. Your task is to write a summary report for a user based on a detailed emotional analysis of their text.

            Here is the user's original text:
            "{user_input}"

            Here is a summary of the key emotions detected and their psychological analysis:
            {analysis_summary}

            Based on all this information, please write a comprehensive summary report that includes:
            1.  **Opening**: Start with a warm, validating opening that summarizes the core emotional state in 1-2 sentences.
            2.  **Deeper Insight**: Briefly synthesize the individual analyses. Explain how the key emotions (like {', '.join(s['label'] for s in enriched_stats)}) are interconnected in the user's experience.
            3.  **Actionable Suggestions**: Provide 2-3 concrete, actionable suggestions tailored to their emotional state.
                - If the user's sentiment is primarily '{sentiment_type}', suggest ways to soothe or process these feelings (e.g., mindfulness, journaling, talking to a friend).
                - If the sentiment is positive, suggest ways to cherish, extend, or build upon these feelings.
            4.  **Closing**: End with a hopeful and encouraging closing statement.

            Please make the report detailed, supportive, and easy to understand.
            """
            response = self.client.models.generate_content(model=self.model, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return "We have analyzed your feelings and want to remind you that your emotional well-being is important. Taking time to understand your feelings is a positive step."

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
