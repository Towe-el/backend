from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

# Represents an ObjectId field in the database
PyObjectId = Annotated[str, BeforeValidator(str)]

class SearchResult(BaseModel):
    """
    Model for a single search result from GoEmotions database
    """
    text: str
    similarity_score: float

class SearchHistory(BaseModel):
    """
    Model for storing search history
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    query_text: str
    similar_texts: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "query_text": "I feel happy today",
                "similar_texts": ["feeling joyful", "so happy", "great day"],
                "timestamp": "2024-03-15T10:30:00"
            }
        } 