from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel
from app.services.search_service import perform_semantic_search
from app.services.conversation_guide_service import ConversationGuideService

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

class EmotionAnalysis(BaseModel):
    """
    Model for emotion analysis results
    """
    has_emotion_content: bool
    emotion_intensity: float
    confidence: float
    needs_more_detail: bool
    guidance_suggestion: Optional[str]

class SearchResponse(BaseModel):
    """
    Model for search response
    """
    results: List[SearchResultItem]
    message: Optional[str]
    emotion_analysis: Optional[EmotionAnalysis]
    ai_response: Optional[str]
    concise_response: Optional[Dict[str, str]]

@router.post("/", response_model=SearchResponse)
async def search_emotions(query: SearchQuery):
    """
    Search for similar texts in the database and get AI analysis.
    """
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        # Initialize conversation guide service
        guide_service = ConversationGuideService()
        
        # Perform semantic search
        search_results_raw = await perform_semantic_search(query.text, top_n=10)
        
        # Get AI analysis and response
        guide_result = await guide_service.process_user_input(query.text)
        
        # Convert raw results to Pydantic models
        pydantic_results = [
            SearchResultItem(
                id=doc.get("_id"), 
                text=doc.get("text"), 
                emotion_label=doc.get("emotion_label"), 
                score=doc.get("score")
            ) for doc in search_results_raw
        ]
        
        # Create emotion analysis response
        emotion_analysis = None
        if guide_result and "analysis" in guide_result:
            analysis = guide_result["analysis"]
            emotion_analysis = EmotionAnalysis(
                has_emotion_content=analysis["emotion_analysis"]["has_emotion_content"],
                emotion_intensity=analysis["emotion_analysis"]["emotion_intensity"],
                confidence=analysis["emotion_analysis"]["confidence"],
                needs_more_detail=analysis.get("needs_more_detail", False),
                guidance_suggestion=analysis.get("guidance_suggestion")
            )
            
        # Prepare the combined response
        message = "No matching documents found." if not search_results_raw else None
        return SearchResponse(
            results=pydantic_results,
            message=message,
            emotion_analysis=emotion_analysis,
            ai_response=guide_result.get("guide_response") if guide_result else None,
            concise_response=guide_result.get("concise_response") if guide_result else None
        )

    except Exception as e:
        print(f"Unexpected error in /search endpoint: {str(e)} - Query: {query.text}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during the search."
        ) 