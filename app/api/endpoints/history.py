from fastapi import APIRouter, HTTPException, Path
from typing import Optional
from pydantic import BaseModel
from app.services.session_service import SessionService
from app.services.history_service import HistoryService

router = APIRouter(
    prefix="/history",
    tags=["history"]
)

# Instantiate services
session_service = SessionService()
history_service = HistoryService()

class TitleResponse(BaseModel):
    session_id: str
    title: Optional[str]
    ready_for_search: bool
    message: str

@router.get("/title/{session_id}", response_model=TitleResponse, summary="Generate title for session history")
async def get_session_title(
    session_id: str = Path(..., description="The session ID to generate title for")
):
    """
    Generate a concise title for the session's accumulated text.
    Only generates a title if the session is ready for search (ready_for_search = true).
    
    Returns:
    - title: Generated title (max 10 words) if ready_for_search is true, otherwise None
    - ready_for_search: Boolean indicating if the session is ready for search
    - message: Status message
    """
    try:
        # Get the session data
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        accumulated_text = session.get("accumulated_text", "")
        ready_for_search = session.get("ready_for_search", False)
        
        print(f"[get_session_title] Session {session_id} - ready_for_search: {ready_for_search}")
        print(f"[get_session_title] Accumulated text length: {len(accumulated_text)}")
        
        title = None
        message = "Session not ready for title generation."
        
        # Only generate title if ready for search and has content
        if ready_for_search and accumulated_text.strip():
            print(f"[get_session_title] Generating title for session {session_id}...")
            title = await history_service.generate_title(accumulated_text)
            print(f"[get_session_title] Generated title: '{title}'")
            message = "Title generated successfully." if title else "Failed to generate title."
        elif not accumulated_text.strip():
            message = "No content available for title generation."
        
        return TitleResponse(
            session_id=session_id,
            title=title,
            ready_for_search=ready_for_search,
            message=message
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in /history/title/{session_id} endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating title.") 