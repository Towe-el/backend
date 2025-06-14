import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from app.database import async_db

class SessionService:
    """
    Manages user sessions by storing them in a MongoDB collection.
    This service is designed for a stateless server environment like Cloud Run.
    """

    def __init__(self):
        """
        Initializes the service and gets a reference to the 'sessions' collection.
        """
        if async_db is None:
            print("Warning: async_db is None. Database operations will fail.")
            self.db = None
            self.collection = None
        else:
            self.db = async_db
            self.collection = self.db["sessions"]
            print("SessionService initialized. MongoDB collection: sessions")

    async def create_session(self) -> str:
        """
        Creates a new, empty session in the database.

        Returns:
            The unique session ID for the newly created session.
        """
        if self.collection is None:
            raise RuntimeError("Database not available")
            
        session_id = str(uuid.uuid4())
        session_data = {
            "_id": session_id,
            "accumulated_text": "",
            "input_round": 0,
            "ready_for_search": False,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        await self.collection.insert_one(session_data)
        print(f"Created new session with ID: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a session from the database.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The session data as a dictionary, or None if not found.
        """
        if not session_id or self.collection is None:
            print(f"[get_session] Early return: session_id={session_id}, collection={self.collection}")
            return None
        
        print(f"[get_session] Looking for session: {session_id}")
        try:
            session = await self.collection.find_one({"_id": session_id})
            if session:
                print(f"[get_session] Found session {session_id} with input_round={session.get('input_round', 0)}")
            else:
                print(f"[get_session] Session {session_id} NOT FOUND in database")
            return session
        except Exception as e:
            print(f"[get_session] Database error looking for session {session_id}: {e}")
            return None

    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Updates a session with new data.

        Args:
            session_id: The ID of the session to update.
            update_data: A dictionary containing the fields to update.

        Returns:
            True if the update was successful, False otherwise.
        """
        if self.collection is None:
            print(f"[update_session] Database collection is None")
            return False
            
        update_data["updated_at"] = datetime.now(timezone.utc)
        print(f"[update_session] Updating session {session_id} with data: {update_data}")
        
        try:
            result = await self.collection.update_one(
                {"_id": session_id},
                {"$set": update_data}
            )
            print(f"[update_session] Update result - matched: {result.matched_count}, modified: {result.modified_count}")
            
            if result.modified_count == 0 and result.matched_count > 0:
                print(f"Session {session_id} was looked up but not modified.")
            elif result.modified_count > 0:
                print(f"Successfully updated session {session_id}.")
            elif result.matched_count == 0:
                print(f"[update_session] WARNING: Session {session_id} not found for update!")
                
            return result.modified_count > 0 or result.matched_count > 0
        except Exception as e:
            print(f"[update_session] Database error updating session {session_id}: {e}")
            return False
    
    async def clear_accumulated_text(self, session_id: str) -> bool:
        """
        Clears the accumulated text and resets the state for a new conversation
        within the same session.

        Args:
            session_id: The ID of the session to clear.

        Returns:
            True if the session was cleared, False otherwise.
        """
        print(f"Clearing accumulated text for session {session_id}")
        return await self.update_session(
            session_id,
            {
                "accumulated_text": "",
                "input_round": 0,
                "ready_for_search": False,
            }
        )
    


    async def get_or_create_session(self, session_id: Optional[str]) -> Dict[str, Any]:
        """
        A convenient helper to either get an existing session or create a new one.

        Args:
            session_id: The session ID from the client, which might be None.

        Returns:
            A session data dictionary (either existing or newly created).
        """
        print(f"[get_or_create_session] Called with session_id: {session_id}")
        session = None
        if session_id:
            session = await self.get_session(session_id)
        
        if not session:
            print(f"[get_or_create_session] Creating new session because existing session not found")
            new_session_id = await self.create_session()
            session = await self.get_session(new_session_id)
        else:
            print(f"[get_or_create_session] Using existing session {session_id}")

        return session 