# Upload, Read, Chunk, and Print Word Document Content (NO NLTK version)
from google.colab import files
from docx import Document
from transformers import AutoTokenizer

# Step 1: Upload DOCX file
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Step 2: Extract clean text
def get_clean_text(file_path):
    doc = Document(file_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return " ".join(paragraphs)

text = get_clean_text(file_name)

# Step 3: Define chunking function using tokenizer only (no nltk)
def split_text_to_chunks(text, chunk_size=512, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text.strip())

    return chunks

# Step 4: Create and print chunks
chunks = split_text_to_chunks(text, chunk_size=512)

print(f"\n Total Chunks Created: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print("\n" + "-"*80 + "\n")

