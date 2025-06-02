import os
import sys
import re
import numpy as np
from typing import List, Optional

from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Google Cloud Vertex AI setup
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

# --- Constants and Global Initializations ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-005"

MONGO_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DATABASE", "GoEmotion") # Default to GoEmotion if not set
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "vectorizedText") # Default if not set

# --- Global variables for initialized clients/models ---
db_collection_service = None
text_embedding_model_service = None

print("Initializing Search Service...")

# Validate core environment variables at module load time
if not PROJECT_ID:
    print("FATAL ERROR in search_service: GOOGLE_CLOUD_PROJECT env var not set.")
    # In a real app, might raise an exception or have a better config management
if not MONGO_URI:
    print("FATAL ERROR in search_service: MONGODB_URI env var not set.")

if CRED_PATH and not os.path.exists(CRED_PATH):
    print(f"Warning in search_service: Credential file specified by GOOGLE_APPLICATION_CREDENTIALS not found.")
    print(f"Attempted to find it at absolute path: {os.path.abspath(CRED_PATH)}")
    print(f"Current working directory is: {os.getcwd()}")

# Initialize Vertex AI SDK and Model (runs once when module is imported)
try:
    if PROJECT_ID: # Proceed only if Project ID is set
        print(f"search_service: Initializing Vertex AI SDK (Project: {PROJECT_ID}, Location: {LOCATION})...")
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        print("search_service: Vertex AI SDK initialized.")
        
        print(f"search_service: Loading text embedding model: {MODEL_NAME} (target 256-dim)...")
        try:
            text_embedding_model_service = TextEmbeddingModel.from_pretrained(MODEL_NAME, output_dimensionality=256)
            print("search_service: Text embedding model loaded (specified 256-dim).")
        except Exception as e_dim:
            print(f"search_service: Failed to load model with output_dimensionality=256: {e_dim}. Retrying without...")
            text_embedding_model_service = TextEmbeddingModel.from_pretrained(MODEL_NAME)
            print("search_service: Text embedding model loaded (default dim). WARNING: Verify it is 256.")
    else:
        text_embedding_model_service = None # Ensure it's None if not initialized
        print("search_service: Vertex AI not initialized due to missing PROJECT_ID.")
except Exception as e_vertex:
    print(f"FATAL ERROR in search_service: Failed to initialize Vertex AI or load model: {e_vertex}")
    text_embedding_model_service = None # Ensure it's None on failure

# Connect to MongoDB (runs once when module is imported)
try:
    if MONGO_URI: # Proceed only if MONGO_URI is set
        print(f"search_service: Connecting to MongoDB...")
        mongo_client_service = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        mongo_client_service.admin.command('ping') # Verify connection
        print("search_service: MongoDB connection successful.")
        db = mongo_client_service[DB_NAME]
        db_collection_service = db[COLLECTION_NAME]
    else:
        db_collection_service = None # Ensure it's None if not initialized
        print("search_service: MongoDB not connected due to missing MONGO_URI.")
except Exception as e_mongo:
    print(f"FATAL ERROR in search_service: Could not connect to MongoDB: {e_mongo}")
    db_collection_service = None # Ensure it's None on failure

# Adjusted conditional checks for service initialization status
if text_embedding_model_service is not None and db_collection_service is not None:
    print("Search Service initialized successfully with Vertex AI and MongoDB.")
elif text_embedding_model_service is None:
    print("Search Service initialized WITH ERRORS: Vertex AI model NOT available.")
elif db_collection_service is None:
    print("Search Service initialized WITH ERRORS: MongoDB collection NOT available.")
else: # Should not be reached if the above cover all None cases, but good for safety
    print("Search Service initialized WITH UNKNOWN ERRORS: Check component statuses.")

# --- Helper Functions ---
def split_into_sentences_service(paragraph: str) -> list[str]:
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

def _get_vertex_embedding_service(text: str) -> list[float]:
    global text_embedding_model_service
    if text_embedding_model_service is None:
        print("Error in _get_vertex_embedding_service: Text embedding model not available.")
        return [] 
        
    if not text.strip():
        return [] 
    try:
        embeddings_response = text_embedding_model_service.get_embeddings([text], output_dimensionality=256)
        if embeddings_response and embeddings_response[0].values:
            raw_embedding = embeddings_response[0].values
            if len(raw_embedding) != 256:
                print(f"Warning in _get_vertex_embedding_service: Embedding dim is {len(raw_embedding)}, not 256! For text: '{text[:30]}...'")
            return raw_embedding
        return []
    except Exception as e:
        print(f"Error in _get_vertex_embedding_service calling Vertex AI for text '{text[:50]}...': {e}")
        return []

def _get_weighted_average_embedding_service(texts: list[str], weights: Optional[list[float]] = None) -> list[float]:
    if not texts:
        return []
    embeddings_list = []
    for text_item in texts:
        emb = _get_vertex_embedding_service(text_item)
        if emb:
            embeddings_list.append(emb)
    
    if not embeddings_list:
        return []

    embeddings_np = np.array(embeddings_list)
    # Basic check for consistent dimensions if multiple embeddings were generated
    if embeddings_np.ndim > 1 and embeddings_np.shape[0] > 1:
        first_dim_shape = embeddings_np[0].shape
        for i in range(1, embeddings_np.shape[0]):
            if embeddings_np[i].shape != first_dim_shape:
                print(f"Error in _get_weighted_average_embedding_service: Inconsistent embedding dimensions found.")
                return [] # Cannot average inconsistent dimensions
    elif embeddings_np.ndim == 0 or (embeddings_np.ndim ==1 and len(texts) > 1) :
        print(f"Error in _get_weighted_average_embedding_service: Unexpected embedding array structure.")
        return []

    weights_np_calc = None
    if weights:
        if len(weights) != len(embeddings_np):
            print("Warning in _get_weighted_average_embedding_service: Mismatched weights and texts. Using simple average.")
        else:
            weights_np_calc = np.array(weights)
            if np.sum(weights_np_calc) == 0:
                weights_np_calc = None 
            else:
                weights_np_calc = weights_np_calc / np.sum(weights_np_calc)
    
    if weights_np_calc is not None and embeddings_np.size > 0:
        # Ensure embeddings_np is treated as a 2D array for broadcasting with weights
        if embeddings_np.ndim == 1: # Single embedding, but weights provided (unusual but handle)
             if len(weights_np_calc) == 1:
                 weighted_avg = embeddings_np * weights_np_calc[0] # Element-wise if weights_np_calc is scalar
             else: # Should not happen if len(weights) == len(embeddings_np) check passes with single embedding
                 print("Error: Weight logic error for single embedding.")
                 return []
        else:
            weighted_avg = np.sum(embeddings_np * weights_np_calc[:, np.newaxis], axis=0)
    elif embeddings_np.size > 0:
        weighted_avg = np.mean(embeddings_np, axis=0)
    else:
        return []
        
    return weighted_avg.tolist()

# --- Main Search Function ---
async def perform_semantic_search(query_text: str, top_n: int = 3) -> List[dict]:
    global db_collection_service
    if db_collection_service is None:
        print("Error in perform_semantic_search: MongoDB collection not available.")
        return [] 
    if text_embedding_model_service is None:
        print("Error in perform_semantic_search: Text embedding model not available.")
        return []

    if not query_text.strip():
        return []

    sentences = split_into_sentences_service(query_text)
    if not sentences:
        return []

    avg_query_vector = _get_weighted_average_embedding_service(sentences, weights=None)
    if not avg_query_vector or len(avg_query_vector) != 256:
        print(f"Error in perform_semantic_search: Failed to generate valid 256-dim query vector. Dim: {len(avg_query_vector) if avg_query_vector else 'None'}")
        return []
        
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "queryVector": avg_query_vector,
                "path": "vector",       
                "numCandidates": 100,   
                "limit": top_n          
            }
        },
        {
            "$project": {
                "_id": {"$toString": "$_id"}, 
                "text": 1,
                "emotion_label": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    try:
        results = list(db_collection_service.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Error in perform_semantic_search during DB aggregation: {e}")
        return [] 