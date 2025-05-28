import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载文本转向量模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 连接到 MongoDB
client = MongoClient(os.environ["MONGODB_URL"], server_api=ServerApi('1'))
db = client.GoEmotion
collection = db.rawData

# 测试查询文本
query_text = "I feel bad because I don't have job yet."

# 将查询文本转换为向量
query_vector = model.encode(query_text)

print("\nDebug Information:")
print(f"Query vector dimensions: {len(query_vector)}")

try:
    # 获取所有文档的文本
    print("\nProcessing documents...")
    documents = list(collection.find({}, {"text": 1, "emotion_label": 1}))
    
    if not documents:
        print("No documents found in the collection.")
        exit()
    
    print(f"Processing {len(documents)} documents...")
    
    # 将所有文档文本转换为向量
    texts = [doc["text"] for doc in documents]
    doc_vectors = model.encode(texts)
    
    # 计算余弦相似度
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
    )
    
    # 获取最相似的3个文档的索引
    top_k = 3
    most_similar_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print("\nSearch results for query:", query_text)
    print("-" * 50)
    
    # 打印结果
    for idx in most_similar_indices:
        similarity_score = similarities[idx]
        doc = documents[idx]
        print(f"Text: {doc['text']}")
        print(f"Emotion: {doc['emotion_label']}")
        print(f"Similarity Score: {similarity_score:.4f}")
        print("-" * 50)

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    client.close()