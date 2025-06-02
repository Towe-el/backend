from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from app.services.search_service import perform_semantic_search

router = APIRouter(
    prefix="/search",
    tags=["search"]
)

class SearchQuery(BaseModel):
    """
    Model for search query input
    """
    text: str

class SearchResultItem(BaseModel):
    """
    Model for a single search result item
    """
    id: Optional[str]  # MongoDB ObjectId as string
    text: Optional[str]
    emotion_label: Optional[list] # Assuming this is the structure
    score: Optional[float]

class SearchResponse(BaseModel):
    """
    Model for the search API response
    """
    results: List[SearchResultItem]
    message: Optional[str] = None # For any additional info, like "No results found"

@router.post("/", response_model=SearchResponse)
async def search_emotions(query: SearchQuery):
    """
    Search for similar texts in the database using semantic search.
    (History saving is currently disabled)
    """
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        # Perform semantic search using the new service
        search_results_raw = await perform_semantic_search(query.text, top_n=10)
        
        if not search_results_raw and isinstance(search_results_raw, list): # Check if it's an empty list from service error
            return SearchResponse(results=[], message="No matching documents found or an error occurred during search.")

        # Convert raw results to Pydantic models for typed response
        # This ensures the response conforms to the SearchResultItem schema
        pydantic_results = [
            SearchResultItem(
                id=doc.get("_id"), 
                text=doc.get("text"), 
                emotion_label=doc.get("emotion_label"), 
                score=doc.get("score")
            ) for doc in search_results_raw
        ]
            
        return SearchResponse(results=pydantic_results)

    except HTTPException as http_exc: # Re-raise HTTPExceptions from underlying services if any
        raise http_exc
    except Exception as e:
        # Log the exception for server-side review
        print(f"Unexpected error in /search endpoint: {str(e)} - Query: {query.text}") 
        # Consider logging traceback: import traceback; traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during the search."
        ) 