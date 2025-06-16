import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_inference.dummy_model import DummySentimentModel, predict
from audio_input.audio_loader import load_audio, list_labeled_audio_files
from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram

filepath, label = list_labeled_audio_files("audio_input/normalized")[0]
audio, sr = load_audio(filepath)
segment = segment_audio(audio, sr)[0]
spec = compute_spectrogram(segment, sr)

model = DummySentimentModel()
label_idx, confidence = predict(model, spec)

print(f"Predicted class index: {label_idx}")
print(f"Confidence vector: {confidence}")
