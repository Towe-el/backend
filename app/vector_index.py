import os
import time
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi

# Connect to MongoDB
client = MongoClient(os.environ["MONGODB_URL"], server_api=ServerApi('1'))
db = client.GoEmotion
collection = db.vectorizedText

# Define the vector search index
index_definition = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "vector": {
                "type": "knnVector",
                "dimensions": 256,
                "similarity": "cosine"
            },
            "text": {
                "type": "string"
            }
        }
    }
}

# index name
index_name = "vector_index"

try:
    # list the existing indexes
    print("\nExisting search indexes before operation:")
    indexes = list(collection.list_search_indexes())
    for index in indexes:
        print(f"- {index['name']}")
    
    # if the index exists, drop it first
    if any(index['name'] == index_name for index in indexes):
        print(f"\nDropping existing index '{index_name}'...")
        collection.drop_search_index(index_name)
        print(f"Index '{index_name}' dropped successfully!")
        # add a short delay to wait for the deletion operation to complete
        time.sleep(15)

    # create the index model
    print("\nCreating new search index...")
    search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name
    )
    
    # create the index using the model
    result = collection.create_search_index(model=search_index_model)
    print(f"Vector search index '{result}' created successfully!")

    # wait for the index creation to complete
    time.sleep(10)

    # verify the final index status
    print("\nFinal search indexes:")
    indexes = list(collection.list_search_indexes())
    for index in indexes:
        print(f"- {index['name']}")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    client.close()
