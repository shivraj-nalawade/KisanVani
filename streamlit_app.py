import os
import tempfile
import streamlit as st
from app.services.asr import transcribe_audio
from app.services.rag import search_knowledge_base, build_vector_db, DOCS_DIR
from app.services.llm import generate_response
from app.services.tts import text_to_speech

st.set_page_config(
    page_title="KisanVani - Multilingual Agri Assistant",
    page_icon="🌾",
    layout="centered"
)

OUTPUT_AUDIO_DIR = os.path.join("data", "output_audio")
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

LANG_OPTIONS = {
    "Auto-Detect": None,
    "Marathi (मराठी)": "mr",
    "Hindi (हिन्दी)": "hi",
    "English": "en",
}

def detect_text_language(text: str) -> str:
    """Detects English vs Marathi/Hindi Devanagari for typed text queries."""
    if text.isascii():
        return "en"
    # Check for Marathi-specific Devanagari characters (ळ, ऱ, ॅ, ॉ) or common Marathi markers
    marathi_markers = ("ळ", "ऱ", "आहे", "आहेत", "कसे", "काय", "साठी", "पिकाला", "शेती")
    if any(marker in text for marker in marathi_markers):
        return "mr"
    return "hi"

def run_agri_pipeline(query_text: str, lang_code: str) -> tuple[str, str]:
    """Runs RAG retrieval, Groq LLM generation, and gTTS audio synthesis."""
    context = search_knowledge_base(query_text)
    answer = generate_response(query_text, context)
    audio_filename = text_to_speech(answer, lang=lang_code)
    audio_path = os.path.join(OUTPUT_AUDIO_DIR, audio_filename)
    return answer, audio_path

# --- Sidebar: Knowledge Base & Settings ---
with st.sidebar:
    st.header("🌾 KisanVani Portal")
    st.caption("Multilingual Voice & Text Agricultural Assistant")

    selected_lang_label = st.selectbox(
        "Response Language",
        options=list(LANG_OPTIONS.keys()),
        index=0,
        help="Auto-Detect matches your spoken or typed language automatically."
    )
    forced_lang = LANG_OPTIONS[selected_lang_label]

    st.divider()
    st.subheader("📚 Indexed Knowledge Base")
    if os.path.exists(DOCS_DIR):
        pdfs = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        if pdfs:
            for pdf in pdfs:
                st.markdown(f"- `{pdf}`")
        else:
            st.warning("No PDFs found in `data/documents/`.")

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Re-indexing PDFs into FAISS..."):
            success = build_vector_db()
        if success:
            st.success("Knowledge Base rebuilt!")
        else:
            st.error("No PDFs available to index.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
st.title("🌾 KisanVani Agricultural Assistant")
st.write("Ask farming, pest control, or PM-KISAN questions by **speaking** or **typing** in Marathi, Hindi, or English.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"], format="audio/mp3")

# --- Voice Input Section ---
audio_recording = st.audio_input("🎙️ Record a Voice Question (Marathi / Hindi / English)")

if audio_recording is not None:
    # Prevent reprocessing the same recording on Streamlit reruns
    audio_bytes = audio_recording.getvalue()
    audio_hash = hash(audio_bytes)

    if st.session_state.get("last_audio_hash") != audio_hash:
        st.session_state["last_audio_hash"] = audio_hash

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(audio_bytes)
            tmp_wav_path = tmp_wav.name

        try:
            with st.spinner("Transcribing your voice note..."):
                transcript, detected_lang = transcribe_audio(tmp_wav_path)
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

        if transcript:
            reply_lang = forced_lang or detected_lang
            user_display = f"🎙️ *{transcript}*"
            st.session_state.messages.append({"role": "user", "content": user_display})

            with st.chat_message("user"):
                st.markdown(user_display)

            with st.chat_message("assistant"):
                with st.spinner("Searching agricultural guides & generating voice reply..."):
                    answer, audio_path = run_agri_pipeline(transcript, reply_lang)
                st.markdown(answer)
                st.audio(audio_path, format="audio/mp3", autoplay=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "audio_path": audio_path
            })
        else:
            st.error("Could not understand the audio. Please try speaking clearly again.")

# --- Text Input Section ---
typed_query = st.chat_input("Type your question in Marathi, Hindi, or English...")

if typed_query:
    reply_lang = forced_lang or detect_text_language(typed_query)
    st.session_state.messages.append({"role": "user", "content": typed_query})

    with st.chat_message("user"):
        st.markdown(typed_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching agricultural guides & generating answer..."):
            answer, audio_path = run_agri_pipeline(typed_query, reply_lang)
        st.markdown(answer)
        st.audio(audio_path, format="audio/mp3", autoplay=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "audio_path": audio_path
    })