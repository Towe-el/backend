import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from app.database import db

class SessionService:
    """
    Manages user sessions by storing them in a MongoDB collection.
    This service is designed for a stateless server environment like Cloud Run.
    """

    def __init__(self):
        """
        Initializes the service and gets a reference to the 'sessions' collection.
        """
        self.db = db
        self.collection = self.db["sessions"]
        print("SessionService initialized. MongoDB collection: sessions")

    async def create_session(self) -> str:
        """
        Creates a new, empty session in the database.

        Returns:
            The unique session ID for the newly created session.
        """
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
        if not session_id:
            return None
        return await self.collection.find_one({"_id": session_id})

    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Updates a session with new data.

        Args:
            session_id: The ID of the session to update.
            update_data: A dictionary containing the fields to update.

        Returns:
            True if the update was successful, False otherwise.
        """
        update_data["updated_at"] = datetime.now(timezone.utc)
        result = await self.collection.update_one(
            {"_id": session_id},
            {"$set": update_data}
        )
        if result.modified_count == 0 and result.matched_count > 0:
            print(f"Session {session_id} was looked up but not modified.")
        elif result.modified_count > 0:
            print(f"Successfully updated session {session_id}.")
        return result.modified_count > 0 or result.matched_count > 0
    
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
        session = None
        if session_id:
            session = await self.get_session(session_id)
        
        if not session:
            new_session_id = await self.create_session()
            session = await self.get_session(new_session_id)

        return session 