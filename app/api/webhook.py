from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from app.services.audio import download_and_convert_audio
from app.services.asr import transcribe_audio

router = APIRouter()

def process_voice_note(media_url: str, sender: str):
    """Background task to handle the entire voice pipeline"""
    # 1. Download and convert
    wav_path = download_and_convert_audio(media_url, sender)
    
    # 2. Transcribe
    transcript = transcribe_audio(wav_path)
    
    # Next step will go here (Querying the Vector Database)

@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    form_data = await request.form()
    
    sender = form_data.get("From")
    media_url = form_data.get("MediaUrl0")
    
    if media_url:
        print(f"\n--- New Voice Note from {sender} ---")
        # Run the full pipeline in the background
        background_tasks.add_task(process_voice_note, media_url, sender)
    else:
        print(f"\n--- New Text from {sender} ---")
        print(f"Body: {form_data.get('Body')}")
        
    return PlainTextResponse("OK", status_code=200)