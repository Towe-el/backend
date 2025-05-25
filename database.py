import os
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection
client = AsyncIOMotorClient(os.environ["MONGODB_URL"])

# Database
db = client.GoEmotion  # 使用同一个数据库

# Collections
raw_data_collection = db.get_collection("rawData")  # 被搜索的集合
emotion_history_collection = db.get_collection("emotionHistory")  # 存储搜索历史的集合 