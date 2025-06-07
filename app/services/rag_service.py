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

    def process_search_results(self, search_results: List[Dict], user_input: str) -> Dict:
        """Process search results to get emotion statistics and generate empathetic responses"""
        try:
            # Debug: Print search results structure
            print("Search results type:", type(search_results))
            if search_results:
                print("First result type:", type(search_results[0]))
                print("First result keys:", search_results[0].keys())
                print("First result emotion_label type:", 
                      type(search_results[0].get("emotion_label")) if search_results[0].get("emotion_label") else "None")

            # Ensure search_results is a list of dictionaries
            if not isinstance(search_results, list):
                print("Error: search_results is not a list")
                return {
                    "emotion_stats": [],
                    "example_responses": {}
                }

            # Convert any non-dict items to dict
            processed_results = []
            for result in search_results:
                if isinstance(result, dict):
                    processed_result = {
                        "text": str(result.get("text", "")),
                        "emotion_label": str(result.get("emotion_label", "")),
                        "score": float(result.get("score", 0.0))
                    }
                    processed_results.append(processed_result)
                else:
                    print(f"Skipping non-dict result: {type(result)}")

            if not processed_results:
                print("No valid results after processing")
                return {
                    "emotion_stats": [],
                    "example_responses": {}
                }

            # 1. Calculate emotion label statistics with user input context
            emotion_stats = self._calculate_emotion_stats(processed_results, user_input)
            if not emotion_stats:
                print("No emotion stats generated")
                return {
                    "emotion_stats": [],
                    "example_responses": {}
                }

            # 2. Generate example responses based on similar experiences
            example_responses = self._generate_example_responses(processed_results, emotion_stats)

            result = {
                "emotion_stats": emotion_stats,
                "example_responses": example_responses
            }

            # Debug: Print final result structure
            print("Final result structure:")
            print("- emotion_stats length:", len(result["emotion_stats"]))
            print("- example_responses keys:", list(result["example_responses"].keys()))

            return result

        except Exception as e:
            print(f"Error in process_search_results: {str(e)}")
            print("Exception type:", type(e))
            import traceback
            print("Traceback:", traceback.format_exc())
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
                    import ast
                    emotion_label = ast.literal_eval(emotion_label)
                except:
                    return emotion_label.strip()
                
            if isinstance(emotion_label, list):
                # Sort by count and exclude neutral
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
                
            return ""
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
        """Calculate statistics for emotion labels from search results"""
        try:
            print("Calculating stats for results:", len(search_results))

            # Initialize emotion counter
            emotion_counts = {}
            emotion_scores = {}  # Store relevance scores

            # Count emotions and collect texts
            for result in search_results:
                emotion = self._extract_primary_emotion(result.get("emotion_label"))
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if not emotion_counts:
                print("No valid emotions found in results")
                return []

            # Get emotions with top 3 or more counts
            count_threshold = sorted(emotion_counts.values(), reverse=True)[min(2, len(emotion_counts)-1)]
            candidate_emotions = [
                emotion for emotion, count in emotion_counts.items()
                if count >= count_threshold
            ]

            print(f"Candidate emotions with counts >= {count_threshold}: {candidate_emotions}")

            # If we have more than 3 candidates, analyze relevance
            if len(candidate_emotions) > 3:
                # Calculate relevance scores for all candidates
                for emotion in candidate_emotions:
                    relevance_score = self._analyze_emotion_relevance(user_input, emotion, [
                        result.get("text", "") for result in search_results
                        if self._extract_primary_emotion(result.get("emotion_label")) == emotion
                    ])
                    emotion_scores[emotion] = relevance_score

                print("Emotion relevance scores:", emotion_scores)
                
                # Sort by relevance score and count
                candidate_emotions.sort(
                    key=lambda x: (-emotion_scores.get(x, 0), -emotion_counts[x], x)
                )
                candidate_emotions = candidate_emotions[:3]

            # Calculate percentages for final emotions
            total_count = sum(emotion_counts[emotion] for emotion in candidate_emotions)
            
            emotion_stats = []
            for emotion in candidate_emotions:
                count = emotion_counts[emotion]
                percentage = (count / total_count) * 100 if total_count > 0 else 0
                stat = {
                    "label": str(emotion),
                    "count": int(count),
                    "percentage": float(percentage)
                }
                if emotion in emotion_scores:
                    stat["relevance_score"] = float(emotion_scores[emotion])
                emotion_stats.append(stat)

            print("Generated emotion stats:", emotion_stats)
            return emotion_stats

        except Exception as e:
            print(f"Error in _calculate_emotion_stats: {str(e)}")
            print("Exception type:", type(e))
            import traceback
            print("Traceback:", traceback.format_exc())
            return []

    def _generate_example_responses(self, search_results: List[Dict], emotion_stats: List[Dict]) -> Dict:
        """Generate example responses based on similar experiences"""
        try:
            print("Generating responses for results:", len(search_results))
            print("Using emotion stats:", emotion_stats)

            example_responses = {}
            
            # Get the labels from emotion stats
            top_emotions = [stat["label"] for stat in emotion_stats]
            
            # Extract all relevant texts for context
            all_texts = [str(result.get("text", "")).strip() for result in search_results if result.get("text")]
            context_texts = all_texts[:20]  # Use top 20 texts for context
            
            # Group responses by emotion
            for emotion in top_emotions:
                # Generate empathetic response based on similar experiences
                prompt = f"""Based on these experiences where people felt {emotion}, create a single sentence that expresses this emotion deeply and personally.

Context:
Original experience: {context_texts[0]}
Similar experiences: {' | '.join(context_texts[1:3])}

Expression styles {{
- Express through thoughts: "I can't stop thinking about...", "My mind keeps replaying..."
- Express through social impact: "The way others reacted made me feel...", "I feel so distant from everyone..."
- Express through environment: "The world feels different now...", "Everything around me seems..."
- Express through pure emotion: "This {emotion} feels like...", "I'm overwhelmed by..."
- Express through metaphor: "It's as if...", "Like a..."
- Express through action: "When I...", "As I..."
- Express through memory: "I keep remembering...", "The thought of..."
- Express through sensation: "Inside, I feel...", "There's this feeling..."
}}

Requirements:
- Create a natural, first-person sentence that captures the essence of {emotion}
- Make it specific and vivid
- Focus on the emotional experience
- Be authentic and relatable
- Avoid explaining or giving advice
- Use one of the expression styles"""

                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[prompt]
                    )
                    example_responses[emotion] = str(response.text.strip())
                except Exception as e:
                    print(f"Error generating example response for {emotion}: {e}")
                    # Fallback to using a similar experience if generation fails
                    matching_results = [
                        result for result in search_results
                        if self._extract_primary_emotion(result.get("emotion_label")) == emotion
                    ]
                    if matching_results:
                        example_responses[emotion] = str(matching_results[0].get("text", "")).strip()
                    else:
                        example_responses[emotion] = f"I deeply feel {emotion} in this situation."

            print("Generated responses:", example_responses)
            return example_responses

        except Exception as e:
            print(f"Error in _generate_example_responses: {str(e)}")
            print("Exception type:", type(e))
            import traceback
            print("Traceback:", traceback.format_exc())
            return {} 