import os
from groq import Groq
from gtts.lang import tts_langs
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Dynamically fetch supported ISO codes and language names directly from gTTS
SUPPORTED_TTS_LANGS = tts_langs()
LANG_NAME_TO_CODE = {name.lower(): code for code, name in SUPPORTED_TTS_LANGS.items()}

# Domain-free orthography prompt to anchor Devanagari and Latin scripts
ORTHOGRAPHY_PROMPT = "नमस्कार, आपले स्वागत आहे. नमस्ते, आपका स्वागत है. Hello, welcome."

def _resolve_tts_code(detected: str) -> str:
    """Dynamically maps Whisper's language output to a valid gTTS language code."""
    clean = (detected or "en").lower().strip()
    if clean in SUPPORTED_TTS_LANGS:
        return clean
    return LANG_NAME_TO_CODE.get(clean, "en")

def _verify_indic_script(transcript: str, detected_lang: str) -> tuple[str, str]:
    """Corrects short Marathi/Hindi speech that Whisper mistakenly transcribed in Gujarati script."""
    if detected_lang != "gu" or not transcript:
        return transcript, detected_lang

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Speech-to-text sometimes transcribes short Marathi or Hindi audio phonetically "
                        "using Gujarati script. Analyze the phonetics and grammar of the input text:\n"
                        "- If it is spoken Marathi written in Gujarati script, convert it to accurate Marathi Devanagari and output: mr|<marathi_text>\n"
                        "- If it is spoken Hindi written in Gujarati script, convert it to accurate Hindi Devanagari and output: hi|<hindi_text>\n"
                        "- If it is genuine Gujarati, keep it unchanged and output: gu|<original_text>\n"
                        "Output ONLY in the format code|text with no explanation."
                    )
                },
                {"role": "user", "content": transcript}
            ],
            temperature=0.0,
            max_completion_tokens=200,
            reasoning_effort="low",
            stream=False
        )
        raw = (completion.choices[0].message.content or "").strip()
        if "|" in raw:
            code, corrected_text = raw.split("|", 1)
            code = code.strip().lower()
            if code in SUPPORTED_TTS_LANGS and corrected_text.strip():
                return corrected_text.strip(), code
    except Exception as e:
        print(f"[SCRIPT VERIFY WARNING]: {e}")

    return transcript, detected_lang

def transcribe_audio(file_path: str) -> tuple[str, str]:
    print(f"Transcribing audio with Groq Whisper from {file_path}...")
    
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                prompt=ORTHOGRAPHY_PROMPT,
            )
        
        transcript = (transcription.text or "").strip()
        raw_lang = getattr(transcription, "language", None) or "english"
        detected_lang = _resolve_tts_code(raw_lang)
        
        # Ensure Marathi/Hindi isn't misclassified as Gujarati
        transcript, detected_lang = _verify_indic_script(transcript, detected_lang)
        
        print(f"\n[ASR RESULT] Language: {detected_lang} | Transcript: {transcript}\n")
        return transcript, detected_lang

    except Exception as e:
        print(f"\n[ASR ERROR]: {str(e)}\n")
        return "", "en"