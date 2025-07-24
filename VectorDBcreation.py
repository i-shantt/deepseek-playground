import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Get path to script's folder
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load FAISS index and documents
index = faiss.read_index(os.path.join(base_dir, "disaster_index.faiss"))
with open(os.path.join(base_dir, "disaster_documents.json"), "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load the same embedding model used before
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function: Retrieve top-k similar documents given a query
def retrieve_similar_examples(query_text, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(documents[i])
    return results

#Example usage
query = "What should we do during a flash flood?"
retrieved = retrieve_similar_examples(query, top_k=3)

print("\nTop 3 similar examples:\n")
for i, text in enumerate(retrieved, 1):
    print(f"{i}. {text}\n")
