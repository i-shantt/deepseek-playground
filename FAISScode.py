import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Get path to this script's folder
output_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
file_path = os.path.join(output_dir, "sample1_data_1.jsonl")
documents = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        documents.append(data["output"])  # Or combine instruction/output if needed

# Make sure we have data
if not documents:
    raise ValueError("No documents found! Check the JSON keys or file content.")

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index + docs
faiss_path = os.path.join(output_dir, "disaster_index.faiss")
json_path = os.path.join(output_dir, "disaster_documents.json")

faiss.write_index(index, faiss_path)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"Saved FAISS index to: {faiss_path}")
print(f"Saved JSON documents to: {json_path}")
