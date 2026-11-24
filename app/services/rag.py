import os
import fitz  # PyMuPDF
import pickle
from groq import Groq
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=10.0)

DOCS_DIR = "data/documents"
DB_DIR = "data/vector_db"
INDEX_FILE = f"{DB_DIR}/tfidf_index.pkl"


def build_vector_db():
    """Reads PDFs, creates overlapping chunks, and saves the TF-IDF search index."""
    print("Building Knowledge Base...")
    os.makedirs(DB_DIR, exist_ok=True)
    chunks = []

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"

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

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, f)

    print(f"Knowledge Base built successfully with {len(chunks)} chunks!")
    return True


def _translate_query_for_search(query: str) -> str:
    """Normalizes and translates non-English ASR queries into clean English for search."""
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
                        "syllables before translating. Use standard Indian agricultural terminology. "
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

    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)

    vectorizer = data["vectorizer"]
    matrix = data["matrix"]
    chunks = data["chunks"]

    search_query = _translate_query_for_search(query)
    query_vec = vectorizer.transform([search_query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    top_indices = scores.argsort()[::-1][:top_k]
    evidence = [chunks[idx] for idx in top_indices if scores[idx] > 0.01]

    return "\n\n".join(evidence)


if not os.path.exists(INDEX_FILE) and os.path.exists(DOCS_DIR) and os.listdir(DOCS_DIR):
    build_vector_db()