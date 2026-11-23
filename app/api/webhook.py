import os
import time
from dotenv import load_dotenv
from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from twilio.rest import Client
from app.services.audio import download_and_convert_audio
from app.services.asr import transcribe_audio
from app.services.rag import search_knowledge_base
from app.services.llm import generate_response
from app.services.tts import text_to_speech

load_dotenv()

router = APIRouter()

TEMP_AUDIO_DIR = os.path.join("data", "temp_audio")
OUTPUT_AUDIO_DIR = os.path.join("data", "output_audio")


def _detect_text_language(text: str) -> str:
    """Detects English vs Marathi/Hindi Devanagari for typed WhatsApp messages."""
    if text.isascii():
        return "en"
    marathi_markers = ("ळ", "ऱ", "आहे", "आहेत", "कसे", "काय", "साठी", "पिकाला", "शेती")
    if any(marker in text for marker in marathi_markers):
        return "mr"
    return "hi"


def _cleanup_old_files(directory: str, max_age_seconds: int = 600):
    """Deletes temporary audio files older than max_age_seconds to save disk space."""
    if not os.path.exists(directory):
        return
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath) and (now - os.path.getmtime(filepath)) > max_age_seconds:
                os.remove(filepath)
        except OSError:
            pass


def send_whatsapp_reply(to_number: str, text: str, audio_filename: str, bot_number: str):
    print(f"Sending final reply to WhatsApp from {bot_number}...")
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    ngrok_url = os.getenv("NGROK_URL", "").strip().rstrip("/")

    client = Client(account_sid, auth_token)
    media_url = f"{ngrok_url}/audio/{audio_filename}"

    message = client.messages.create(
        from_=bot_number,
        to=to_number,
        body=text,
        media_url=[media_url]
    )
    print(f"Message sent! SID: {message.sid}")


def process_voice_note(media_url: str, sender: str, bot_number: str):
    """Background task to handle incoming WhatsApp voice notes."""
    wav_path = None
    try:
        # 1. Download and convert
        wav_path = download_and_convert_audio(media_url, sender)

        # 2. Transcribe and auto-detect language
        transcript, detected_lang = transcribe_audio(wav_path)
        if not transcript:
            return

        # 3. Retrieve relevant context from FAISS
        context = search_knowledge_base(transcript)

        # 4. Generate grounded response
        final_answer = generate_response(transcript, context)

        # 5. Synthesize spoken audio in detected language
        audio_filename = text_to_speech(final_answer, lang=detected_lang)

        # 6. Send text + audio back via Twilio
        send_whatsapp_reply(sender, final_answer, audio_filename, bot_number)

    except Exception as e:
        print(f"[VOICE PIPELINE ERROR]: {e}")

    finally:
        # Remove processed input WAV immediately and sweep expired audio files
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
        _cleanup_old_files(TEMP_AUDIO_DIR, max_age_seconds=300)
        _cleanup_old_files(OUTPUT_AUDIO_DIR, max_age_seconds=600)


def process_text_message(text_body: str, sender: str, bot_number: str):
    """Background task to handle typed WhatsApp text messages."""
    try:
        detected_lang = _detect_text_language(text_body)
        print(f"[TEXT QUERY] Language: {detected_lang} | Text: {text_body}")

        # 1. Retrieve relevant context from FAISS
        context = search_knowledge_base(text_body)

        # 2. Generate grounded response
        final_answer = generate_response(text_body, context)

        # 3. Synthesize spoken audio so farmers can also listen to typed replies
        audio_filename = text_to_speech(final_answer, lang=detected_lang)

        # 4. Send text + audio back via Twilio
        send_whatsapp_reply(sender, final_answer, audio_filename, bot_number)

    except Exception as e:
        print(f"[TEXT PIPELINE ERROR]: {e}")

    finally:
        _cleanup_old_files(OUTPUT_AUDIO_DIR, max_age_seconds=600)


@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    form_data = await request.form()

    sender = form_data.get("From")
    bot_number = form_data.get("To")
    media_url = form_data.get("MediaUrl0")
    text_body = (form_data.get("Body") or "").strip()

    if media_url:
        print(f"\n--- New Voice Note from {sender} to {bot_number} ---")
        background_tasks.add_task(process_voice_note, media_url, sender, bot_number)
    elif text_body:
        print(f"\n--- New Text Message from {sender} to {bot_number} ---")
        background_tasks.add_task(process_text_message, text_body, sender, bot_number)

    return PlainTextResponse("OK", status_code=200)