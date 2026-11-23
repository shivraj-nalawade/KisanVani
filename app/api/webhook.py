import os
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

def send_whatsapp_reply(to_number: str, text: str, audio_filename: str, bot_number: str):
    print(f"Sending final reply to WhatsApp from {bot_number}...")
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    ngrok_url = os.getenv("NGROK_URL", "").strip().rstrip("/")

    client = Client(account_sid, auth_token)
    media_url = f"{ngrok_url}/audio/{audio_filename}"

    message = client.messages.create(
        from_=bot_number,  # Dynamically using the exact number Twilio expects
        to=to_number,
        body=text,
        media_url=[media_url]
    )
    print(f"Message sent! SID: {message.sid}")

def process_voice_note(media_url: str, sender: str, bot_number: str):
    """Background task to handle the entire voice pipeline"""
    # 1. Download and convert
    wav_path = download_and_convert_audio(media_url, sender)
    
    # 2. Transcribe (Speech-to-Text)
    transcript = transcribe_audio(wav_path)
    
    if not transcript:
        return
        
    # 3. Retrieve relevant info from the PDF knowledge base
    context = search_knowledge_base(transcript)
    
    # 4. Generate the final answer
    final_answer = generate_response(transcript, context)
    
    # 5. Convert text answer to spoken audio (TTS)
    audio_filename = text_to_speech(final_answer, lang="en")
    
    # 6. Send text and audio reply back to WhatsApp
    send_whatsapp_reply(sender, final_answer, audio_filename, bot_number)

@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    form_data = await request.form()
    
    sender = form_data.get("From")
    media_url = form_data.get("MediaUrl0")
    bot_number = form_data.get("To")  # Auto-detects your Twilio number from the incoming message
    
    if media_url:
        print(f"\n--- New Voice Note from {sender} to {bot_number} ---")
        background_tasks.add_task(process_voice_note, media_url, sender, bot_number)
    else:
        print(f"\n--- New Text from {sender} ---")
        
    return PlainTextResponse("OK", status_code=200)