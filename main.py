import sys
from transformers import pipeline

def run_speech_to_text(audio_file):
    print("Memuat model Whisper Tiny...")
    model = pipeline("automatic-speech-recognition",
                     model="openai/whisper-tiny",
                     device="cpu")

    print(f"Memproses file: {audio_file}")
    result = model(audio_file)

    print("\n=== HASIL TRANSKRIP ===\n")
    print(result["text"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    run_speech_to_text(audio_path)
