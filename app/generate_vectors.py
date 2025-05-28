import os
from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# 加载文本转向量模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 连接到 MongoDB
client = MongoClient(os.environ["MONGODB_URL"], server_api=ServerApi('1'))
db = client.GoEmotion
collection = db.rawData

try:
    # 获取没有vector字段的文档
    documents = list(collection.find(
        {"vector": {"$exists": False}}, 
        {"text": 1}
    ))
    
    if not documents:
        print("All documents already have vectors!")
        exit()
    
    print(f"\nProcessing {len(documents)} documents...")
    
    # 批处理大小
    BATCH_SIZE = 100
    
    # 准备批量更新操作
    updates = []
    
    # 使用tqdm显示进度条
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Generating vectors"):
        batch = documents[i:i + BATCH_SIZE]
        
        # 获取这批文档的文本
        texts = [doc["text"] for doc in batch]
        
        # 生成向量
        vectors = model.encode(texts)
        
        # 准备更新操作
        for doc, vector in zip(batch, vectors):
            updates.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"vector": vector.tolist()}}
            ))
        
        # 如果累积了足够的更新操作，执行批量更新
        if len(updates) >= BATCH_SIZE:
            result = collection.bulk_write(updates)
            print(f"\nUpdated {result.modified_count} documents")
            updates = []
            # 短暂暂停，避免过度占用数据库资源
            time.sleep(0.1)
    
    # 处理剩余的更新操作
    if updates:
        result = collection.bulk_write(updates)
        print(f"\nUpdated final {result.modified_count} documents")
    
    print("\nVector generation completed!")
    
    # 验证更新结果
    docs_with_vectors = collection.count_documents({"vector": {"$exists": True}})
    total_docs = collection.count_documents({})
    print(f"\nDocuments with vectors: {docs_with_vectors}/{total_docs}")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    client.close() 