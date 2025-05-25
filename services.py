from typing import List
from datetime import datetime
from database import raw_data_collection, emotion_history_collection
from models import SearchHistory, SearchResult

async def vector_search(query_text: str, limit: int = 3) -> List[SearchResult]:
    """
    Perform vector search in GoEmotion rawData collection
    """
    # MongoDB Atlas Vector Search aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "default",  # your vector search index name
                "path": "text",
                "queryVector": query_text,
                "numCandidates": 100,
                "limit": limit
            }
        },
        {
            "$project": {
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = []
    async for doc in raw_data_collection.aggregate(pipeline):
        results.append(SearchResult(
            text=doc["text"],
            similarity_score=doc["score"]
        ))
    
    return results

async def save_search_history(query_text: str, similar_texts: List[str]) -> SearchHistory:
    """
    Save search results to emotionHistory database
    """
    search_history = SearchHistory(
        query_text=query_text,
        similar_texts=similar_texts,
        timestamp=datetime.utcnow()
    )
    
    result = await emotion_history_collection.insert_one(
        search_history.model_dump(by_alias=True, exclude=["id"])
    )
    
    created_history = await emotion_history_collection.find_one(
        {"_id": result.inserted_id}
    )
    
    return SearchHistory(**created_history) 