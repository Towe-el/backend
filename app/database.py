import os
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection
client = AsyncIOMotorClient(os.environ["MONGODB_URI"])

# Database
db = client.GoEmotion