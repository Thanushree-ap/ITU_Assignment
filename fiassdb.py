from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Initialize model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # Output size of MiniLM
index = faiss.IndexFlatL2(dimension)

# Step 2: Global lists to store metadata
metadata = []     # Stores document name for each chunk
all_chunks = []   # Stores the actual text chunks

# Step 3: Function to add parsed document chunks to index
def add_chunks_to_index(doc_name, chunks):
    if not chunks:
        print(f"No chunks to add for: {doc_name}")
        return

    embeddings = model.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)

    all_chunks.extend(chunks)
    metadata.extend([doc_name] * len(chunks))

    print(f" Added {len(chunks)} chunks from '{doc_name}'")

# Step 4: Function to search
def search_query(query, top_k=3, filter_doc=None):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k * 5)

    print(f"\nTop {top_k} results for: '{query}'\n")
    count = 0

    for idx in indices[0]:
        if idx >= len(metadata) or idx >= len(all_chunks):
            continue
        if filter_doc and metadata[idx] != filter_doc:
            continue

        print(f"Result {count+1} (from '{metadata[idx]}'):\n{all_chunks[idx]}")
        print("-" * 80)
        count += 1
        if count >= top_k:
            break

    if count == 0:
        print("No matching chunks found.")

# Step 5: Add your real chunks to the FAISS index
# Assumes variable `chunks` was already created earlier from a DOCX or PDF
add_chunks_to_index("ITU_assignmet_word_format.docx",chunks)

# Step 6: Ask your query
query = input("Enter your question: ")
search_query(query, top_k=3)
