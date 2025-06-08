from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel
import asyncio
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

class EmotionStat(BaseModel):
    label: str
    count: int
    percentage: float

class EnrichedEmotionStat(BaseModel):
    """
    Model for a single enriched emotion stat, including definition, quote, and analysis.
    """
    label: str
    count: int
    percentage: float
    definition: str
    quote: str
    analysis: str

class RAGAnalysis(BaseModel):
    """
    The main model for the entire RAG analysis package.
    """
    enriched_emotion_stats: List[EnrichedEmotionStat]
    summary_report: str

class SearchResponse(BaseModel):
    """
    Model for search response
    """
    results: List[SearchResultItem]
    message: Optional[str]
    emotion_analysis: Optional[EmotionAnalysis]
    guidance_response: Optional[str]
    rag_analysis: Optional[RAGAnalysis]

@router.post("/", response_model=SearchResponse)
async def search_emotions(query: SearchQuery):
    """
    Search for similar texts in the database and get AI analysis.
    Use asyncio.to_thread to run sync logic in thread pool, keep API asynchronous
    """
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        # Initialize conversation guide service
        guide_service = ConversationGuideService()
        
        # run sync search logic in thread pool using asyncio.to_thread
        search_result = await asyncio.to_thread(perform_semantic_search, query.text, 30)
        search_results_raw = search_result["results"]
        rag_analysis_data = search_result.get("rag_analysis")
        
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
            )
            
        # Prepare the combined response
        message = "No matching documents found." if not search_results_raw else None
        return SearchResponse(
            results=pydantic_results,
            message=message,
            emotion_analysis=emotion_analysis,
            guidance_response=guide_result.get("guidance_response") if guide_result else None,
            rag_analysis=RAGAnalysis(**rag_analysis_data) if rag_analysis_data else None
        )
        
    except Exception as e:
        print(f"Unexpected error in /search endpoint: {str(e)} - Query: {query.text}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during the search."
        ) 