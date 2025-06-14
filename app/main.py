import os
import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import search
from app.api.endpoints import debug
from app.api.endpoints import history
from app.services.search_service import text_embedding_model_service
from app.database import async_db, async_health_check

app = FastAPI(
    title="Toweel backend API",
    summary="Toweel backend API",
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://toweel-frontend.web.app", "https://toweel-frontend.firebaseapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get database and collection instances from the centralized database module
db = async_db
vectorized_text_collection = db.get_collection("vectorizedText")

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
    
    # Check MongoDB connection using centralized health check
    if await async_health_check():
        status_details["services"]["database"] = "connected"
    else:
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

# Include the routers
app.include_router(search.router)
app.include_router(debug.router)
app.include_router(history.router)
