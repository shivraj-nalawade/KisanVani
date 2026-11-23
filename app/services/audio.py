import os
import requests
from pydub import AudioSegment
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

def download_and_convert_audio(media_url: str, sender: str) -> str:
    # 1. Download the file from Twilio securely
    print(f"Downloading audio from {media_url}...")
    response = requests.get(media_url, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    
    # Extract the phone number to use as the filename
    clean_sender = sender.replace("whatsapp:", "").replace("+", "")
    ogg_path = f"data/temp_audio/{clean_sender}.ogg"
    wav_path = f"data/temp_audio/{clean_sender}.wav"
    
    # Save the original OGG file
    with open(ogg_path, "wb") as f:
        f.write(response.content)
        
    # 2. Convert to WAV format for Whisper
    print("Converting to WAV...")
    audio = AudioSegment.from_ogg(ogg_path)
    audio.export(wav_path, format="wav")
    
    print(f"Audio ready: {wav_path}")
    return wav_path