import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from audio_input.audio_loader import load_audio, list_labeled_audio_files
from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram

# Load one file
filepath, label = list_labeled_audio_files("audio_input/normalized")[0]
audio, sr = load_audio(filepath)
segments = segment_audio(audio, sr)

# Compute spectrogram for the first segment
spec = compute_spectrogram(segments[0], sr)
print(f"Spectrogram shape: {spec.shape}")

# Plot
plt.imshow(spec, aspect='auto', origin='lower')
plt.title("Log Spectrogram")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")
plt.colorbar(label="Log Magnitude")
plt.show()
