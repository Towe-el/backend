from typing import List
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from app.database import raw_data_collection, emotion_history_collection
from models import SearchHistory, SearchResult

# 加载文本转向量模型
model = SentenceTransformer('all-MiniLM-L6-v2')

async def text_to_vector(text: str) -> List[float]:
    """
    Convert text to vector using sentence transformer
    """
    # 生成向量并转换为列表
    embedding = model.encode(text)
    return embedding.tolist()

async def vector_search(query_text: str, limit: int = 3) -> List[SearchResult]:
    """
    Perform vector search in GoEmotion rawData collection
    """
    # 将查询文本转换为向量
    query_vector = await text_to_vector(query_text)
    
    # MongoDB Atlas Vector Search aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "emotion_vector_index",  # Atlas中创建的索引名称
                "path": "embedding",              # 存储向量的字段名
                "queryVector": query_vector,
                "numCandidates": 100,            # 内部候选项数量
                "limit": limit,                  # 返回结果数量
                "similarity": {
                    "cosine": {}                 # 使用余弦相似度
                }
            }
        },
        {
            "$project": {
                "text": 1,                      # 原始文本
                "score": {"$meta": "vectorSearchScore"}  # 相似度分数
            }
        }
    ]
    
    results = []
    async for doc in raw_data_collection.aggregate(pipeline):
        results.append(SearchResult(
            text=doc["text"],
            similarity_score=doc["score"]
        ))
    
    return results

async def save_search_history(query_text: str, similar_texts: List[str]) -> SearchHistory:
    """
    Save search results to emotionHistory database
    """
    search_history = SearchHistory(
        query_text=query_text,
        similar_texts=similar_texts,
        timestamp=datetime.utcnow()
    )
    
    result = await emotion_history_collection.insert_one(
        search_history.model_dump(by_alias=True, exclude=["id"])
    )
    
    created_history = await emotion_history_collection.find_one(
        {"_id": result.inserted_id}
    )
    
    return SearchHistory(**created_history) 