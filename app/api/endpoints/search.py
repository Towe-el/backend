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

# global conversation guide service instance, keep session state across requests
global_guide_service = ConversationGuideService()

class SearchQuery(BaseModel):
    """
    Model for search query input
    """
    text: str
    execute_search: bool = False  # whether to execute search, default is False

class SearchResultItem(BaseModel):
    """
    Single search result model
    """
    id: Optional[str]
    text: Optional[str]
    emotion_label: Optional[str]
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
    Single enriched emotion statistic
    """
    label: str
    count: int
    percentage: float
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
    accumulated_text: Optional[str] = None
    input_round: Optional[int] = None
    ready_for_search: Optional[bool] = None

@router.post("/", response_model=SearchResponse)
async def search_emotions(query: SearchQuery):
    """
    Search for similar texts in the database and get AI analysis.
    Two-step process: 
    1. First call (execute_search=False): Quality check and guidance
    2. Second call (execute_search=True): Execute search and RAG when ready
    """
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        # use global service instance to maintain session state
        guide_service = global_guide_service
        
        # if execute_search is True, directly perform search (skip quality check)
        if query.execute_search:
            print(f"Executing search for accumulated text")
            
            # get current accumulated text
            accumulated_text = guide_service.get_accumulated_text()
            if not accumulated_text:
                raise HTTPException(status_code=400, detail="No accumulated text found. Please provide input first.")
            
            print(f"Searching with accumulated text: '{accumulated_text}'")
            
            # perform search and RAG analysis
            search_result = await asyncio.to_thread(perform_semantic_search, accumulated_text, 30)
            search_results_raw = search_result["results"]
            rag_analysis_data = search_result.get("rag_analysis")
            
            # convert raw results to Pydantic model, handle emotion_label type conversion
            pydantic_results = []
            for doc in search_results_raw:
                emotion_label = doc.get("emotion_label")
                if isinstance(emotion_label, list):
                    emotion_label_str = str(emotion_label)
                else:
                    emotion_label_str = str(emotion_label) if emotion_label else ""
                
                pydantic_results.append(SearchResultItem(
                    id=doc.get("_id"), 
                    text=doc.get("text"), 
                    emotion_label=emotion_label_str,
                    score=doc.get("score")
                ))
            
            # clear accumulated input after search
            guide_service.clear_accumulated_input()
            
            message = "No matching documents found." if not search_results_raw else None
            return SearchResponse(
                results=pydantic_results,
                message=message,
                emotion_analysis=None,
                guidance_response="Search completed successfully. Here are the results based on your emotional experience.",
                rag_analysis=RAGAnalysis(**rag_analysis_data) if rag_analysis_data else None,
                accumulated_text=accumulated_text,
                input_round=0,
                ready_for_search=False
            )
        
        print(f"Processing conversation guide analysis for: '{query.text}'")
        guide_result = await guide_service.process_user_input(query.text)
        
        # create emotion analysis response
        emotion_analysis = None
        if guide_result and "analysis" in guide_result:
            analysis = guide_result["analysis"]
            emotion_analysis = EmotionAnalysis(
                has_emotion_content=analysis["emotion_analysis"]["has_emotion_content"],
                emotion_intensity=analysis["emotion_analysis"]["emotion_intensity"],
                confidence=analysis["emotion_analysis"]["confidence"],
                needs_more_detail=analysis.get("needs_more_detail", False),
            )
        
        # get status information
        needs_more_detail = guide_result.get("needs_more_input", True) if guide_result else True
        accumulated_text = guide_result.get("accumulated_text", "") if guide_result else ""
        input_round = guide_result.get("input_round", 0) if guide_result else 0
        ready_for_search = guide_result.get("ready_for_search", False) if guide_result else False
        
        # debug quality check logic
        print(f"Quality check details:")
        print(f"  - needs_more_detail: {needs_more_detail}")
        print(f"  - ready_for_search: {ready_for_search}")
        print(f"  - accumulated_text: '{accumulated_text}'")
        print(f"  - input_round: {input_round}")
        if guide_result and "analysis" in guide_result:
            print(f"  - sentence_count: {guide_result['analysis']['sentence_count']}")
            print(f"  - emotion_intensity: {guide_result['analysis']['emotion_analysis']['emotion_intensity']}")
        
        # return quality check result, do not execute search
        if ready_for_search:
            message = "Your input quality is sufficient. Click the search button to find similar emotional experiences and get detailed analysis."
        else:
            message = "Please provide more detailed information about your emotional experience."
            
        return SearchResponse(
            results=[],
            message=message,
            emotion_analysis=emotion_analysis,
            guidance_response=guide_result.get("guidance_response") if guide_result else None,
            rag_analysis=None,
            accumulated_text=accumulated_text,
            input_round=input_round,
            ready_for_search=ready_for_search
        )
        
    except Exception as e:
        print(f"Unexpected error in /search endpoint: {str(e)} - Query: {query.text}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during the search."
        )

@router.post("/clear-session", response_model=Dict[str, str])
async def clear_session():
    """
    clear accumulated input
    """
    try:
        global_guide_service.clear_accumulated_input()
        return {"message": "Session cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@router.get("/session-status", response_model=Dict)
async def get_session_status():
    """
    get current session status
    """
    try:
        return {
            "accumulated_inputs": global_guide_service.accumulated_input,
            "accumulated_text": global_guide_service.get_accumulated_text(),
            "session_start_time": global_guide_service.session_start_time.isoformat(),
            "input_count": len(global_guide_service.accumulated_input)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session status: {str(e)}")

@router.post("/execute", response_model=SearchResponse)
async def execute_search():
    """
    Execute search and RAG analysis for the accumulated text.
    This endpoint is called when user clicks the search button.
    """
    try:
        guide_service = global_guide_service
        
        # get current accumulated text
        accumulated_text = guide_service.get_accumulated_text()
        if not accumulated_text:
            raise HTTPException(status_code=400, detail="No accumulated text found. Please provide input first.")
        
        print(f"Executing search for accumulated text: '{accumulated_text}'")
        
        # perform search and RAG analysis
        search_result = await asyncio.to_thread(perform_semantic_search, accumulated_text, 30)
        search_results_raw = search_result["results"]
        rag_analysis_data = search_result.get("rag_analysis")
        
        # convert raw results to Pydantic model, handle emotion_label type conversion
        pydantic_results = []
        for doc in search_results_raw:
            emotion_label = doc.get("emotion_label")
            if isinstance(emotion_label, list):
                emotion_label_str = str(emotion_label)
            else:
                emotion_label_str = str(emotion_label) if emotion_label else ""
            
            pydantic_results.append(SearchResultItem(
                id=doc.get("_id"), 
                text=doc.get("text"), 
                emotion_label=emotion_label_str,
                score=doc.get("score")
            ))
        
        # clear accumulated input after search
        guide_service.clear_accumulated_input()
        
        message = "No matching documents found." if not search_results_raw else None
        return SearchResponse(
            results=pydantic_results,
            message=message,
            emotion_analysis=None,
            guidance_response="Search completed successfully. Here are the results based on your emotional experience.",
            rag_analysis=RAGAnalysis(**rag_analysis_data) if rag_analysis_data else None,
            accumulated_text=accumulated_text,
            input_round=0,
            ready_for_search=False
        )
        
    except Exception as e:
        print(f"Unexpected error in /search/execute endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during search execution."
        ) 