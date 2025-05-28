import json
import os
import sys
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import time
from tqdm import tqdm

def check_credentials():
    """Check Google Cloud authentication configuration"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not project_id:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable is not set")
        return False
        
    if not credentials_path:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        return False
        
    if not os.path.exists(credentials_path):
        print(f"Error: Can't find the authentication file: {credentials_path}")
        return False
    
    return True

# check the authentication configuration
if not check_credentials():
    sys.exit(1)

# initialize Google Cloud
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = "us-central1"

try:
    print("Initializing Google Cloud...")
    aiplatform.init(project=project_id, location=location)
except Exception as e:
    print(f"Initialize Google Cloud failed: {str(e)}")
    print("Please check your authentication configuration and network connection")
    sys.exit(1)

# load the processed data
print("Loading processed data...")
try:
    with open('processed_emotions_with_counts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error:Can't find processed_emotions_with_counts.json file")
    print("Please ensure that you run the data processing script to generate this file first")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error:Can't parse JSON file")
    print("Please ensure that the file format is correct")
    sys.exit(1)

# initialize the text embedding model
print("Initializing text embedding model...")
try:
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
except Exception as e:
    print(f"Initialize model failed: {str(e)}")
    sys.exit(1)

def get_embedding(text):
    """Get the embedding of the text"""
    try:
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        print(f"Get embedding failed: {text[:50]}...")
        print(f"Error information: {str(e)}")
        return None

# batch size
BATCH_SIZE = 5  # Vertex AI has QPS limit, so we use small batch size
print(f"\nStart processing {len(data)} documents, batch size is {BATCH_SIZE}...")

# process all documents
processed_count = 0
error_count = 0
processed_data = []

try:
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i + BATCH_SIZE]
        
        for doc in batch:
            try:
                # get the vector representation of the text
                vector = get_embedding(doc['text'])
                
                if vector is not None:
                    # add the vector to the document
                    doc['vector'] = vector
                    processed_data.append(doc)
                    processed_count += 1
                else:
                    error_count += 1
                
            except Exception as e:
                print(f"\nError processing document: {str(e)}")
                error_count += 1
            
        # add a short delay to comply with API limit
        time.sleep(1)
        
        # save checkpoint every 100 documents
        if processed_count > 0 and processed_count % 100 == 0:
            print(f"\nSave checkpoint, processed {processed_count} documents...")
            with open('vectorized_emotions_vertex.json', 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

except KeyboardInterrupt:
    print("\nUser interrupted. Saving current progress...")
except Exception as e:
    print(f"\nError: {str(e)}")
finally:
    # save the final result
    print("\nSave final result...")
    with open('vectorized_emotions_vertex.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

# print the processing statistics
print("\nProcessing completed!")
print(f"Successfully processed: {processed_count} documents")
print(f"Failed to process: {error_count} documents")

# show an example document (including vector)
if processed_data:
    print("\nExample document structure:")
    sample = processed_data[0].copy()
    sample['vector'] = sample['vector'][:5] + ['...']  # only show the first 5 vector values
    print(json.dumps(sample, ensure_ascii=False, indent=2)) 