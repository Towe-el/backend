import os
from typing import Optional, List
import datetime

from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator

from typing_extensions import Annotated

from bson import ObjectId
import motor.motor_asyncio
from pymongo import ReturnDocument

from app.api.endpoints import search
from app.services.search_service import text_embedding_model_service

app = FastAPI(
    title="Toweel backend API",
    summary="Test vector search.",
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all sources, should be set to specific sources in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URI"])
db = client.GoEmotion
vectorized_text_collection = db.get_collection("vectorizedText")

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]

@app.get("/health")
async def health_check():
    status_details = {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.UTC),
        "services": {
            "database": "unknown",
            "vertex_ai": "unknown"
        }
    }
    
    # Check MongoDB connection
    try:
        await db.command("ping")
        status_details["services"]["database"] = "connected"
    except Exception as e:
        status_details["services"]["database"] = "disconnected"
        status_details["status"] = "unhealthy"
    
    # Check Vertex AI status
    if text_embedding_model_service is not None:
        status_details["services"]["vertex_ai"] = "connected"
    else:
        status_details["services"]["vertex_ai"] = "disconnected"
        # Don't mark as unhealthy if only Vertex AI is disconnected
        # since the service might still be partially functional
    
    if status_details["services"]["database"] == "disconnected":
        raise HTTPException(
            status_code=503,
            detail=status_details
        )
    
    return status_details

# Include the search router
app.include_router(search.router)
