import os
from gtts import gTTS
import uuid

def text_to_speech(text: str, lang: str = 'en') -> str:
    print(f"Generating spoken audio in language: {lang}...")
    
    # Ensure the output directory exists
    output_dir = "data/output_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique filename
    filename = f"response_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(output_dir, filename)
    
    # Generate the speech and save it
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filepath)
    
    print(f"Audio generated successfully: {filepath}")
    return filename