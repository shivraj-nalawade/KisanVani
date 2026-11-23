import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

print("Loading Multilingual Embedding Model... (Will download on first run)")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=10.0)

DOCS_DIR = "data/documents"
DB_DIR = "data/vector_db"
INDEX_FILE = f"{DB_DIR}/faiss_index.bin"
CHUNKS_FILE = f"{DB_DIR}/chunks.pkl"

def build_vector_db():
    """Reads PDFs, creates overlapping embeddings, and saves the FAISS index."""
    print("Building Knowledge Base...")
    os.makedirs(DB_DIR, exist_ok=True)
    chunks = []
    
    # 1. Read all PDFs in the documents folder
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            
            # 2. Overlapping chunking (150 words per chunk, 30-word overlap)
            words = text.split()
            chunk_size = 150
            overlap = 30
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk_text = " ".join(words[i:i + chunk_size])
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

def _translate_query_for_search(query: str) -> str:
    """Normalizes and translates non-English ASR queries into clean English for FAISS search."""
    if query.isascii():
        return query

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate Indian speech-to-text queries into English for searching Indian agricultural PDFs. "
                        "ASR often splits compound words with extra spaces or slight phonetic typos—mentally rejoin split "
                        "syllables before translating. Use standard Indian agricultural terminology (never Western/American terms). "
                        "Output the clean English question followed by 3-5 key English search terms on the same line."
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_completion_tokens=150,
            reasoning_effort="low",
            stream=False
        )
        translated = (completion.choices[0].message.content or "").strip()
        if translated:
            print(f"[RAG SEARCH QUERY (Translated)]: {translated}")
            return translated
    except Exception as e:
        print(f"[RAG TRANSLATION WARNING]: {e}")

    return query

def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Searches the database for the most relevant document chunks."""
    if not os.path.exists(INDEX_FILE):
        build_vector_db()

    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    # Normalize Indic queries into English for accurate matching against English PDFs
    search_query = _translate_query_for_search(query)

    # Embed the search query
    query_vector = embedder.encode([search_query], convert_to_numpy=True)
    
    # Search FAISS
    distances, indices = index.search(query_vector, top_k)
    
    # Compile the retrieved evidence
    evidence = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            evidence.append(chunks[idx])
            
    return "\n\n".join(evidence)

# Auto-build on startup if missing
if not os.path.exists(INDEX_FILE) and os.path.exists(DOCS_DIR) and os.listdir(DOCS_DIR):
    build_vector_db()