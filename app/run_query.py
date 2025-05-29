import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
import re
import sys

# Google Cloud Vertex AI setup
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
CRED_PATH  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LOCATION   = "us-central1"
MODEL_NAME = "text-embedding-005"

if not PROJECT_ID:
    sys.exit("Please set the GOOGLE_CLOUD_PROJECT environment variable.")
if CRED_PATH and not os.path.exists(CRED_PATH):
    # GOOGLE_APPLICATION_CREDENTIALS can be optional if running in a GCP environment with service account attached
    print(f"Warning: Credential file specified by GOOGLE_APPLICATION_CREDENTIALS not found: {CRED_PATH}")

# Initialize Vertex AI SDK
try:
    print(f"Initializing Vertex AI SDK with Project ID: {PROJECT_ID}, Location: {LOCATION}...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    print("Vertex AI SDK initialized successfully.")
    print(f"Loading text embedding model: {MODEL_NAME}...")
    text_embedding_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    print("Text embedding model loaded successfully.")
except Exception as e:
    sys.exit(f"Failed to initialize Vertex AI SDK or load model: {e}")

# Placeholder for your sentence tokenizer
def split_into_sentences(paragraph: str) -> list[str]:
    """
    Splits a paragraph into sentences.
    This is a basic implementation. Consider using a more robust library like NLTK or spaCy.
    """
    if not paragraph:
        return []
    sentences = re.split(r'([.!?])\s*', paragraph)
    result = []
    current_sentence = ""
    for part in sentences:
        if not part:
            continue
        current_sentence += part
        if part in ".!?":
            result.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence.strip():
        result.append(current_sentence.strip())
    return [s for s in result if s]


def get_embedding(text: str) -> list[float]:
    """
    Generates an embedding for the given text using Google Vertex AI text-embedding-005 API.
    Aims for 256 dimensions as per model loading.
    """
    if not text.strip():
        print("Warning: Empty text provided to get_embedding. Returning empty list.")
        return []
    try:
        # print(f"Generating embedding for: '{text[:50]}...'") # Optional: for verbose logging
        # The model is already loaded globally as text_embedding_model
        embeddings = text_embedding_model.get_embeddings([text], output_dimensionality=256)
        if embeddings and embeddings[0].values:
            return embeddings[0].values
        else:
            print(f"Warning: Received no embedding values for text: '{text[:50]}...'")
            return []
    except Exception as e:
        print(f"Error calling Vertex AI embedding API for text '{text[:50]}...': {e}")
        return []


def get_weighted_average_embedding(texts: list[str], weights: list[float] = None) -> list[float]:
    """
    Calculates the weighted average of embeddings for a list of texts.
    If weights are not provided, a simple average is computed.
    """
    if not texts:
        print("No texts provided for embedding.")
        return []

    embeddings_list = [] # Renamed from embeddings to avoid confusion with the import
    for text_item in texts: # Renamed from text to avoid confusion
        emb = get_embedding(text_item)
        if emb:
            embeddings_list.append(emb)
        else:
            print(f"Warning: Could not generate embedding for text: '{text_item[:50]}...'")
    
    if not embeddings_list:
        print("No embeddings were generated.")
        return []

    embeddings_np = np.array(embeddings_list)

    if weights:
        if len(weights) != len(embeddings_np):
            print("Warning: Number of weights does not match the number of texts. Using simple average.")
            weights_np_calc = None # Fallback to simple average, renamed to avoid confusion
        else:
            weights_np_calc = np.array(weights)
            if np.sum(weights_np_calc) == 0:
                 print("Warning: Sum of weights is zero. Using simple average.")
                 weights_np_calc = None
            else:
                weights_np_calc = weights_np_calc / np.sum(weights_np_calc)
    else:
        weights_np_calc = None

    if weights_np_calc is not None:
        weighted_avg = np.sum(embeddings_np * weights_np_calc[:, np.newaxis], axis=0)
    else:
        weighted_avg = np.mean(embeddings_np, axis=0)
    
    return weighted_avg.tolist()

def vector_search(query_vector: list[float], collection, limit: int = 3):
    """
    Performs a vector search on the specified collection.
    """
    if not query_vector:
        print("Query vector is empty. Cannot perform search.")
        return []
        
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_vector,
                "path": "vector",
                "numCandidates": 100,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "emotion_label": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    try:
        results = collection.aggregate(pipeline)
        return list(results)
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []

if __name__ == "__main__":
    client = None
    try:
        mongo_url = os.environ.get("MONGODB_URL")
        if not mongo_url:
            print("Error: MONGODB_URL environment variable not set.")
            exit(1)
            
        client = MongoClient(mongo_url, server_api=ServerApi('1'))
        db = client.GoEmotion
        collection = db.vectorizedText
        print("Successfully connected to MongoDB.")

        user_paragraph = input("Enter a paragraph for semantic search (or type 'exit' to quit): ")

        while user_paragraph.lower() != 'exit':
            if not user_paragraph.strip():
                print("Input is empty. Please try again or type 'exit'.")
                user_paragraph = input("Enter a paragraph for semantic search (or type 'exit' to quit): ")
                continue

            sentences = split_into_sentences(user_paragraph)
            print(f"\nSentences ({len(sentences)}): {sentences}")

            if not sentences:
                print("Could not split the input into sentences.")
            else:
                print("\nCalculating average query vector...")
                avg_query_vector = get_weighted_average_embedding(sentences)

                if not avg_query_vector:
                    print("Could not generate a query vector.")
                else:
                    print(f"\nPerforming vector search with vector of dimension {len(avg_query_vector)}...")
                    search_results = vector_search(avg_query_vector, collection, limit=3)

                    print("\nTop 3 matching documents:")
                    if search_results:
                        for i, doc in enumerate(search_results):
                            print(f"Result {i+1}:")
                            print(f"  ID: {doc.get('_id')}")
                            print(f"  Text: {doc.get('text')}")
                            print(f"  Emotion Label: {doc.get('emotion_label')}")
                            print(f"  Match Score: {doc.get('score')}")
                            print("-" * 20)
                    else:
                        print("No matching documents found.")
            
            user_paragraph = input("\nEnter another paragraph (or type 'exit' to quit): ")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if client:
            client.close()
            print("\nConnection closed.")
