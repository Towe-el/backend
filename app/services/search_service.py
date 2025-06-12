import os
import sys
import re
import numpy as np
from typing import List, Optional, Dict
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.database import sync_db, sync_client

# Google Cloud Vertex AI setup
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.api_core import exceptions as google_exceptions
from google.api_core.retry import Retry
from .rag_service import RAGProcessor

# --- Constants and Global Initializations ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LOCATION = "europe-west1"
MODEL_NAME = "text-embedding-005"

# --- Global variables for initialized clients/models ---
db_collection_service = None
text_embedding_model_service = None
rag_processor = RAGProcessor()

print("Initializing Search Service...")

# Validate core environment variables at module load time
if not PROJECT_ID:
    print("FATAL ERROR in search_service: GOOGLE_CLOUD_PROJECT env var not set.")

if CRED_PATH and not os.path.exists(CRED_PATH):
    print(f"Warning in search_service: Credential file specified by GOOGLE_APPLICATION_CREDENTIALS not found.")
    print(f"Attempted to find it at absolute path: {os.path.abspath(CRED_PATH)}")
    print(f"Current working directory is: {os.getcwd()}")

# Initialize Vertex AI SDK and Model (runs once when module is imported)
try:
    if PROJECT_ID:
        print(f"search_service: Initializing Vertex AI SDK (Project: {PROJECT_ID}, Location: {LOCATION})...")
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        print("search_service: Vertex AI SDK initialized.")
        
        print(f"search_service: Loading text embedding model: {MODEL_NAME}...")
        try:
            text_embedding_model_service = TextEmbeddingModel.from_pretrained(MODEL_NAME)
            print("search_service: Text embedding model loaded.")
        except Exception as e_dim:
            print(f"search_service: Failed to load model: {e_dim}")
            text_embedding_model_service = TextEmbeddingModel.from_pretrained(MODEL_NAME)
            print("search_service: Text embedding model loaded (retry).")
    else:
        text_embedding_model_service = None
        print("search_service: Vertex AI not initialized due to missing PROJECT_ID.")
except Exception as e_vertex:
    print(f"FATAL ERROR in search_service: Failed to initialize Vertex AI or load model: {e_vertex}")
    text_embedding_model_service = None

# Connect to MongoDB
try:
    from app.database import COLLECTION_NAME
    print(f"search_service: MongoDB connecting...")
    # Test the connection
    sync_client.admin.command('ping')
    print("search_service: MongoDB connection successful.")
    db_collection_service = sync_db[COLLECTION_NAME]
except Exception as e_mongo:
    print(f"FATAL ERROR in search_service: Could not connect to MongoDB: {e_mongo}")
    db_collection_service = None

# Adjusted conditional checks for service initialization status
if text_embedding_model_service is not None and db_collection_service is not None:
    print("Search Service initialized successfully with Vertex AI and MongoDB.")
elif text_embedding_model_service is None:
    print("Search Service initialized WITH ERRORS: Vertex AI model NOT available.")
elif db_collection_service is None:
    print("Search Service initialized WITH ERRORS: MongoDB collection NOT available.")
else:
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

def _get_vertex_embedding_service_core(text: str) -> list[float]:
    """Core function to get text embedding from Vertex AI."""
    global text_embedding_model_service
    if text_embedding_model_service is None:
        print("Error in _get_vertex_embedding_service: Text embedding model not available.")
        raise ValueError("Text embedding model not available.")
        
    if not text.strip():
        return [] 
    try:
        embeddings_response = text_embedding_model_service.get_embeddings([text], output_dimensionality=256)
        if embeddings_response and embeddings_response[0].values:
            raw_embedding = embeddings_response[0].values
            if len(raw_embedding) != 256:
                print(f"Warning: Embedding dim is {len(raw_embedding)}, not 256! For text: '{text[:30]}...'")
            return raw_embedding
        return []
    except Exception as e:
        print(f"Error in _get_vertex_embedding_service_core calling Vertex AI for text '{text[:50]}...': {e}")
        raise e

@lru_cache(maxsize=1024)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        ConnectionError
    )),
    reraise=True
)
def _get_vertex_embedding_service(text: str) -> list[float]:
    """Gets a text embedding from Vertex AI with caching and retries."""
    if not text.strip():
        return []
    try:
        return _get_vertex_embedding_service_core(text)
    except (google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            ConnectionError) as e:
        print(f"Failed to get embedding for '{text[:50]}...' after all retries: {e}")
        raise e
    except ValueError as e:
        print(f"Error in _get_vertex_embedding_service: {e}")
        raise e
    except Exception as e:
        print(f"Unexpected error while getting embedding: {e}")
        raise e

def _get_weighted_average_embedding_service(texts: list[str], weights: Optional[list[float]] = None) -> list[float]:
    """Calculate weighted average of embeddings for multiple texts."""
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
    
    if weights:
        if len(weights) != len(embeddings_np):
            print("Warning: Mismatched weights and texts. Using simple average.")
            weights_np = np.ones(len(embeddings_np)) / len(embeddings_np)
        else:
            weights_np = np.array(weights)
            if np.sum(weights_np) == 0:
                weights_np = np.ones(len(embeddings_np)) / len(embeddings_np)
            else:
                weights_np = weights_np / np.sum(weights_np)
    else:
        weights_np = np.ones(len(embeddings_np)) / len(embeddings_np)

    try:
        weighted_avg = np.average(embeddings_np, axis=0, weights=weights_np)
        return weighted_avg.tolist()
    except Exception as e:
        print(f"Error calculating weighted average: {e}")
        return []

# --- Main Search Function ---
def perform_semantic_search(query_text: str, top_n: int = 20) -> Dict:
    """
    Performs a direct vector search on the database without any pre-filtering.
    All emotion analysis and interpretation is handled by the RAG post-processing service.
    """
    global db_collection_service
    if db_collection_service is None:
        print("Error in perform_semantic_search: MongoDB collection not available.")
        return {
            "results": [],
            "rag_analysis": None
        }
    if text_embedding_model_service is None:
        print("Error in perform_semantic_search: Text embedding model not available.")
        return {
            "results": [],
            "rag_analysis": None
        }

    if not query_text.strip():
        return {
            "results": [],
            "rag_analysis": None
        }

    try:
        # Step 1: Get text vector for the user query.
        sentences = split_into_sentences_service(query_text)
        if not sentences:
            return {
                "results": [],
                "rag_analysis": None
            }

        avg_query_vector = _get_weighted_average_embedding_service(sentences, weights=None)
        if not avg_query_vector or len(avg_query_vector) != 256:
            print(f"Error in perform_semantic_search: Failed to generate valid 256-dim query vector. Dim: {len(avg_query_vector) if avg_query_vector else 'None'}")
            return {
                "results": [],
                "rag_analysis": None
            }
        
        # Step 2: Build a simple vector search pipeline without pre-filtering.
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_search_index",
                    "path": "vector",
                    "queryVector": avg_query_vector,
                    "numCandidates": 200,
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
        
        # Step 3: Execute the search.
        results = list(db_collection_service.aggregate(pipeline))
        print(f"Direct vector search returned {len(results)} results")
        
        # Step 4: Process RAG analysis on the raw search results.
        rag_analysis = None
        if results:
            try:
                rag_analysis = rag_processor.process_search_results(results, query_text)
            except Exception as e:
                print(f"Error in RAG processing: {e}")
        
        return {
            "results": results,
            "rag_analysis": rag_analysis
        }
        
    except Exception as e:
        print(f"Error in perform_semantic_search: {e}")
        return {
            "results": [],
            "rag_analysis": None
        } 