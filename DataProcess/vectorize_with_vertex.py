import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from tqdm import tqdm

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
CRED_PATH  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LOCATION   = "us-central1"
MODEL_NAME = "text-embedding-005"

if not PROJECT_ID or not CRED_PATH:
    sys.exit("Please export GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS before running.")
if not os.path.exists(CRED_PATH):
    sys.exit(f"Credential file not found: {CRED_PATH}")


print("Initialising Vertex AI client …")
aiplatform.init(project=PROJECT_ID, location=LOCATION)
try:
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
except Exception as e:
    sys.exit(f"Failed to load embedding model: {e}")


INPUT_FILE  = "processed_emotions_with_counts.json"
OUTPUT_FILE = "vectorized_emotions.json"

print(f"Loading {INPUT_FILE} …")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)
except Exception as exc:
    sys.exit(f"Failed to read {INPUT_FILE}: {exc}")

total_docs = len(documents)
print(f"Loaded {total_docs} documents.\n")


# ------------------------------------------
BATCH_SIZE   = 250   # maximum allowed by Vertex AI for text-embedding models
MAX_WORKERS  = 20    # 20 × 250 = 5000 texts
CHECKPOINT_EVERY = 1000

def embed_batch(batch):
    """Call Vertex AI once for a list[str] -> list[list[float]]."""
    texts = [doc["text"] for doc in batch]
    try:
        vec_objs = model.get_embeddings(texts, output_dimensionality=256)
        return [v.values for v in vec_objs]
    except Exception as err:
        # propagate—outer wrapper will count error & retry / skip
        raise RuntimeError(f"Vertex AI batch embed failed: {err}")

# Split documents into chunks of BATCH_SIZE
def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

processed, failed = 0, 0
out_docs = []
print(f"Vectorising with batch={BATCH_SIZE}, concurrency={MAX_WORKERS} …")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_to_batch = {
        pool.submit(embed_batch, batch): batch for batch in chunkify(documents, BATCH_SIZE)
    }

    for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch)):
        batch = future_to_batch[future]
        try:
            vectors = future.result()
            for doc, vec in zip(batch, vectors):
                doc["vector"] = vec
                out_docs.append(doc)
                processed += 1
        except Exception as exc:
            # fall back to single‑item retry to salvage as much as possible
            failed += len(batch)
            print(f"[warn] batch of {len(batch)} failed → {exc}")

        # checkpoint regularly
        if processed % CHECKPOINT_EVERY == 0 and processed != 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
                json.dump(out_docs, fp, ensure_ascii=False, indent=2)
            print(f"checkpoint written - {processed} vectors so far")


print("Writing final output …")
with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
    json.dump(out_docs, fp, ensure_ascii=False, indent=2)
print(f"Done! processed={processed}, failed={failed}")

if out_docs:
    demo = out_docs[0].copy()
    demo["vector"] = demo["vector"][:5] + ["…"]
    print("\nExample document:")
    print(json.dumps(demo, ensure_ascii=False, indent=2))
