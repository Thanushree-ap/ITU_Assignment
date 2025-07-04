from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings_from_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    embedding_dim = embeddings.shape[1]

    print(" First Embedding Vector:\n", embeddings[0])
    print("\n Embedding Dimension:", embedding_dim)
    return embeddings

generate_embeddings_from_chunks(chunks)