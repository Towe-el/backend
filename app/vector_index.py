import time
from pymongo.operations import SearchIndexModel
from app.database import sync_client, sync_db, COLLECTION_NAME

# MongoDB connection
client = sync_client
db = sync_db
collection = db[COLLECTION_NAME]

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
            },
            "emotion_label": {
                "type": "document",
                "fields": {
                    "tag": {
                        "type": "token"
                    },
                    "cnt": {
                        "type": "number"
                    }
                }
            }
        }
    }
}

# index name
index_name = "vector_search_index"

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
    print(f"Error creating search index: {e}")
    raise

finally:
    client.close()
