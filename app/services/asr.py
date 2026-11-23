from transformers import pipeline
import warnings

# Suppress minor warnings from the transformers library
warnings.filterwarnings("ignore")

print("Loading Whisper model... (This will take a minute to download on the first run)")
# Using the 'tiny' model for rapid CPU inference. It auto-detects the language.
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def transcribe_audio(file_path: str) -> str:
    print(f"Transcribing audio from {file_path}...")
    
    # Generate the transcription
    result = transcriber(file_path)
    transcript = result.get("text", "").strip()
    
    print(f"\n[ASR RESULT] Transcript: {transcript}\n")
    return transcript