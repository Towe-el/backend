from fastapi import APIRouter, HTTPException
from typing import List
from models import SearchHistory, SearchResult
from services import vector_search, save_search_history
from pydantic import BaseModel

router = APIRouter(
    prefix="/search",
    tags=["search"]
)

class SearchQuery(BaseModel):
    """
    Model for search query
    """
    text: str

@router.post("/", response_model=SearchHistory)
async def search_emotions(query: SearchQuery):
    """
    Search for similar texts in GoEmotion database and save the search history
    
    The query text should be provided in the request body, not as a URL parameter.
    This allows for longer text input and better security.
    """
    try:
        # Perform vector search
        search_results = await vector_search(query.text)
        
        # Extract texts from results
        similar_texts = [result.text for result in search_results]
        
        # Save search history
        history = await save_search_history(query.text, similar_texts)
        
        return history
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during the search: {str(e)}"
        ) 