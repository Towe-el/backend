import os
from typing import Optional, List
import datetime

from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator

from typing_extensions import Annotated

from bson import ObjectId
import motor.motor_asyncio
from pymongo import ReturnDocument

from app.api.endpoints import search

app = FastAPI(
    title="Toweel backend API",
    summary="Test vector search.",
)
client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.GoEmotion
vectorized_text_collection = db.get_collection("vectorizedText")

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]

@app.get("/health")
async def health_check():
    try:
        # Check MongoDB connection
        await db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.datetime.now(datetime.UTC)
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy: database connection failed"
        )

# Include the search router
app.include_router(search.router)
