from fastapi import APIRouter, HTTPException, Header, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import asyncio
from datetime import datetime, timedelta
from app.services.search_service import perform_semantic_search
from app.services.conversation_guide_service import ConversationGuideService
from app.services.session_service import SessionService
from app.services.history_service import HistoryService

router = APIRouter(
    prefix="/search",
    tags=["search"]
)

# Instantiate all services
# These are now mostly stateless or manage state externally (like SessionService)
session_service = SessionService()
guide_service = ConversationGuideService()
history_service = HistoryService()

class TextQuery(BaseModel):
    text: str

class BaseResponse(BaseModel):
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

class SearchResponse(BaseResponse):
    results: List[Dict] = []
    rag_analysis: Optional[Dict] = None
    title: Optional[str] = None
    message: Optional[str] = "Search completed successfully."

class GuidanceResponse(BaseResponse):
    guidance_response: str
    accumulated_text: str
    input_round: int
    ready_for_search: bool
    analysis: Dict[str, Any]
    message: Optional[str] = "Input processed."

class SearchQuery(BaseModel):
    """
    Model for search query input - simplified to only contain text
    """
    text: str

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

def _auto_clear_old_sessions():
    """
    Automatically clear sessions older than 30 minutes
    """
    try:
        if hasattr(guide_service, 'session_start_time') and guide_service.session_start_time:
            time_diff = datetime.now() - guide_service.session_start_time
            if time_diff > timedelta(minutes=30):
                print("Auto-clearing old session (>30 minutes)")
                guide_service.clear_accumulated_input()
    except Exception as e:
        print(f"Error in auto session cleanup: {e}")

@router.post("/", response_model=GuidanceResponse, summary="Process user's text input")
async def process_text_input(query: TextQuery, request: Request, session_id: Optional[str] = Header(None)):
    """
    Processes user's text input. Manages session state in the database.
    - If session_id is not provided in the header, a new session is created.
    - The session_id from the response MUST be stored by the client and sent
      back in the header for subsequent requests.
    """
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Debug: Print all headers
    print(f"[process_text_input] All headers: {dict(request.headers)}")
    print(f"[process_text_input] Received session_id from header: {session_id}")
    print(f"[process_text_input] User text: '{query.text}'")

    try:
        session = await session_service.get_or_create_session(session_id)
        print(f"[process_text_input] Got session {session['_id']} with input_round={session.get('input_round', 0)}")
        
        guide_result = await guide_service.process_user_input(
            user_text=query.text,
            current_accumulated_text=session.get("accumulated_text", ""),
            current_round=session.get("input_round", 0)
        )
        
        print(f"[process_text_input] Guide result - new input_round: {guide_result.get('input_round', 'N/A')}")
        print(f"[process_text_input] Guide result - accumulated_text length: {len(guide_result.get('accumulated_text', ''))}")
        
        await session_service.update_session(session["_id"], guide_result)
        print(f"[process_text_input] Updated session {session['_id']}")
        
        return GuidanceResponse(session_id=session["_id"], **guide_result)
        
    except Exception as e:
        print(f"Error in /search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during analysis.")

@router.post("/execute", response_model=SearchResponse, summary="Execute search and RAG analysis")
async def execute_search(session_id: str = Header(..., description="The session ID is required to execute a search.")):
    """
    Executes search using the accumulated text from the session.
    Requires a valid session_id in the header.
    """
    try:
        session = await session_service.get_session(session_id)
        if not session or not session.get("accumulated_text"):
            raise HTTPException(status_code=404, detail="Session not found or is empty.")
        
        accumulated_text = session["accumulated_text"]
        print(f"Executing search for session {session_id} with text: '{accumulated_text}'")
        
        # Execute search
        search_result = await asyncio.to_thread(perform_semantic_search, accumulated_text, 30)
        
        # Generate title for history
        title = None
        if accumulated_text.strip():
            print(f"[execute_search] Generating title for session {session_id}...")
            title = await history_service.generate_title(accumulated_text)
            print(f"[execute_search] Generated title: '{title}'")
        
        return SearchResponse(
            session_id=session_id,
            results=search_result.get("results", []),
            rag_analysis=search_result.get("rag_analysis"),
            title=title,
        )
        
    except Exception as e:
        print(f"Error in /search/execute endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during search execution.")

@router.get("/session-status", response_model=Dict)
async def get_session_status():
    """
    Get current session status for debugging purposes
    """
    try:
        # Auto-clear old sessions
        _auto_clear_old_sessions()
        
        return {
            "accumulated_inputs": guide_service.accumulated_input,
            "accumulated_text": guide_service.get_accumulated_text(),
            "session_start_time": guide_service.session_start_time.isoformat() if guide_service.session_start_time else None,
            "input_count": len(guide_service.accumulated_input)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session status: {str(e)}")

@router.get("/session", response_model=SessionResponse, summary="Create a new session")
async def create_new_session():
    """
    Creates a new session and returns the session_id.
    This endpoint allows the frontend to obtain a session_id before making any POST requests,
    avoiding the issue of session_id binding order.
    """
    try:
        session_id = await session_service.create_session()
        return SessionResponse(
            session_id=session_id,
            message="New session created successfully"
        )
    except Exception as e:
        print(f"Error in /search/session endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create new session.") 