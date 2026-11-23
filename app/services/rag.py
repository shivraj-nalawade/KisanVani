import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

print("Loading Multilingual Embedding Model... (Will download on first run)")
# This model aligns Hindi, Marathi, and English text into the same vector space
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

DOCS_DIR = "data/documents"
DB_DIR = "data/vector_db"
INDEX_FILE = f"{DB_DIR}/faiss_index.bin"
CHUNKS_FILE = f"{DB_DIR}/chunks.pkl"

def build_vector_db():
    """Reads PDFs, creates embeddings, and saves the FAISS index."""
    print("Building Knowledge Base...")
    chunks = []
    
    # 1. Read all PDFs in the documents folder
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            
            # 2. Simple chunking (split by paragraphs or fixed length)
            words = text.split()
            chunk_size = 100 # words per chunk
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i+chunk_size])
                # Append source name for LLM citation
                chunks.append(f" {chunk_text}")

    if not chunks:
        print("No PDFs found in data/documents/")
        return False

    # 3. Create embeddings
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # 4. Build and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
        
    print("Knowledge Base built successfully!")
    return True

def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Searches the database for the most relevant document chunks."""
    if not os.path.exists(INDEX_FILE):
        build_vector_db()

    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    # Embed the user's query
    query_vector = embedder.encode([query], convert_to_numpy=True)
    
    # Search FAISS
    distances, indices = index.search(query_vector, top_k)
    
    # Compile the retrieved evidence
    evidence = []
    for idx in indices[0]:
        if idx < len(chunks):
            evidence.append(chunks[idx])
            
    return "\n\n".join(evidence)

# Auto-build on startup if missing
if not os.path.exists(INDEX_FILE) and os.listdir(DOCS_DIR):
    build_vector_db()