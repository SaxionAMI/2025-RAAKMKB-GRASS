import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_input.audio_loader import load_audio, list_labeled_audio_files
from cpp_processing.segmenter import segment_audio

filepath, label = list_labeled_audio_files("audio_input/normalized")[0]
audio, sr = load_audio(filepath)

segments = segment_audio(audio, sr)
print(f"🔍 Original duration: {len(audio)/sr:.2f}s")
print(f"🧩 Segments: {segments.shape[0]} x {segments.shape[1]} samples ({segments.shape[1]/sr:.2f}s each)")
