import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_input.audio_loader import load_audio, list_labeled_audio_files

input_dir = "audio_input/normalized"
files = list_labeled_audio_files(input_dir)

print(f"🔍 Found {len(files)} audio files.")

for path, label in files[:3]:
    audio, sr = load_audio(path)
    duration = len(audio) / sr
    print(f"{label}: {os.path.basename(path)} - {duration:.2f}s at {sr}Hz")
