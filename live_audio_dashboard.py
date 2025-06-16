import os
import torch
import numpy as np
import sounddevice as sd
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
from torchvision import transforms, models
from cpp_processing.spectrogram import compute_spectrogram
import tkinter as tk

# Settings
CLASS_LABELS = ["Positive", "Negative", "Neutral"]
SAMPLE_RATE = 16000
DURATION = 1.0
MODEL_PATH = os.path.abspath("grass_pipeline_dagster/dagster_output/model/cnn_model.pth")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
root = tk.Tk()
root.title("🎤 Live Crowd Sentiment Monitor")
root.geometry("520x280")
root.configure(bg="#1e1e1e")

label_var = tk.StringVar(value="Waiting for sound...")
confidence_var = tk.StringVar(value="Confidence: -")

sentiment_label = tk.Label(root, textvariable=label_var, font=("Helvetica", 24, "bold"),
                           bg="#1e1e1e", fg="white", pady=30)
sentiment_label.pack()

confidence_label = tk.Label(root, textvariable=confidence_var, font=("Helvetica", 16),
                            bg="#1e1e1e", fg="#90caf9")
confidence_label.pack()

def predict(audio_chunk):
    spec = compute_spectrogram(audio_chunk, SAMPLE_RATE)
    log_spec = 10 * np.log10(spec + 1e-10)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.imsave(tmp.name, log_spec, origin="lower", cmap="jet")
        image = Image.open(tmp.name).convert("RGB")
    os.unlink(tmp.name)

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_index = np.argmax(probs)
        confidence = probs[pred_index]
        label = CLASS_LABELS[pred_index]
    return label, confidence

def update_prediction():
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    label, conf = predict(np.squeeze(audio))

    label_var.set(f"Sentiment: {label}")
    confidence_var.set(f"Confidence: {conf*100:.1f}%")

    color_map = {
        "Positive": "#4CAF50",
        "Negative": "#F44336",
        "Neutral": "#9E9E9E"
    }
    sentiment_label.config(fg=color_map.get(label, "white"))

    root.after(int(DURATION * 1000), update_prediction)

root.after(1000, update_prediction)
root.mainloop()
