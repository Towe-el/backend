from fastapi import APIRouter, HTTPException
from typing import List
from models import SearchHistory, SearchResult
from app.vector_index import semantic_search
from app.services.services import save_search_history
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
    Search for similar texts in GoEmotion database using semantic search
    """
    try:
        # 执行语义搜索
        search_results = await semantic_search(query.text)
        
        # 提取文本
        similar_texts = [result["text"] for result in search_results]
        
        # 保存搜索历史
        history = await save_search_history(query.text, similar_texts)
        
        return history
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during the search: {str(e)}"
        ) 