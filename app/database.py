import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# MongoDB configuration
MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DATABASE", "GoEmotion") 
COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION", "vectorizedText")

# Async MongoDB connection (for FastAPI)
async_client = AsyncIOMotorClient(MONGODB_URI)
async_db = async_client[DB_NAME]

# Sync MongoDB connection (for other services)
sync_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
sync_db = sync_client[DB_NAME]

# Simple health check functions
async def async_health_check() -> bool:
    try:
        await async_db.command("ping")
        return True
    except:
        return False

def sync_health_check() -> bool:
    try:
        sync_client.admin.command('ping')
        return True
    except:
        return False

