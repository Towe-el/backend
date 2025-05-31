import os
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection
client = AsyncIOMotorClient(os.environ["MONGODB_URL"])

# Database
db = client.GoEmotion

# Collections
raw_data_collection = db.get_collection("vectorizedText")
# emotion_history_collection = db.get_collection("emotionHistory")