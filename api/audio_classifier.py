import matplotlib
matplotlib.use('Agg')

import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torchaudio
from collections import defaultdict

# === Manually add project root to sys.path ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grass_pipeline_dagster"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Import internal modules ===
from grass_pipeline_dagster.utils import load_settings, get_minio_client
from cpp_processing.segmenter import segment_audio
from grass_pipeline_dagster.audio_utils.segmentation import compute_spectrogram

# === Load settings ===
settings = load_settings()
model_path = os.path.abspath(settings["model_path"])
class_labels = [label.lower() for label in settings["class_labels"]]
segment_duration = float(settings["segment_duration"])
segment_overlap = float(settings["segment_overlap"])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))

def try_load_from_minio():
    """
    Try to download and load the latest LS model from MinIO.
    Returns True if successful, False otherwise.
    """
    minio_bucket = "grass-pipeline"
    minio_object = "models/ls/latest/ls_cnn_model.pth"
    client = get_minio_client()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        try:
            client.download_file(minio_bucket, minio_object, tmp_file.name)
            print(f"✅ Downloaded model from MinIO: {minio_bucket}/{minio_object}")
            state_dict = torch.load(tmp_file.name, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"✅ Loaded LS model from MinIO.")
            return True
        except Exception as e:
            print(f"⚠️ Could not load model from MinIO: {e}")
            return False
        finally:
            tmp_file.close()
            try:
                os.remove(tmp_file.name)
            except Exception:
                pass

# --- Try MinIO first, fallback to local path ---
if not try_load_from_minio():
    try:
        print(f"ℹ️ Falling back to local model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Loaded model from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model from MinIO and local: {e}")

def classify_audio(audio_path: str, window_size: int = 3, stride: int = 1) -> str:
    print("🧪 classify_audio (sliding window) called with:", audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        if audio_path.lower().endswith(".mp3"):
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_np = waveform[0].numpy()
        else:
            sample_rate, audio_np = wavfile.read(audio_path)
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
    except Exception as e:
        raise RuntimeError(f"Failed to read audio: {e}")

    # Normalize audio
    audio_np = (audio_np - np.mean(audio_np)).astype(np.float32)
    audio_np /= (np.std(audio_np) + 1e-8)

    segments = segment_audio(audio_np, sample_rate, segment_duration, segment_overlap)
    print(f"✂️ {len(segments)} segments created")

    if len(segments) == 0:
        raise RuntimeError("❌ No segments available after segmentation")

    segment_probs = []

    for i, segment in enumerate(segments):
        image_path = None
        try:
            spec = compute_spectrogram(segment, sample_rate)
            log_spec = 10 * np.log10(spec + 1e-10)
            log_spec -= np.max(log_spec)
            log_spec = np.clip(log_spec, -80, 0)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                image_path = tmp_img.name
                plt.figure(figsize=(4, 4), frameon=False)
                ax = plt.gca()
                ax.axis("off")
                plt.imshow(log_spec, origin='lower', aspect='auto', cmap='jet',
                           extent=[0, segment_duration, 0, sample_rate / 2])
                plt.tight_layout()
                plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

            class_prob = {label: probs[idx] for idx, label in enumerate(class_labels)}
            segment_probs.append(class_prob)

        except Exception as e:
            print(f"⚠️ Segment {i} failed:", e)
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    if not segment_probs:
        raise RuntimeError("❌ No segment predictions")

    # === Sliding window averaging ===
    window_predictions = []
    for i in range(0, len(segment_probs) - window_size + 1, stride):
        window = segment_probs[i:i + window_size]
        window_avg = defaultdict(float)
        for seg in window:
            for label in class_labels:
                window_avg[label] += seg[label]
        for label in class_labels:
            window_avg[label] /= window_size
        top_label = max(window_avg, key=window_avg.get)
        window_predictions.append((top_label, dict(window_avg)))

    # === Final audio label based on average over all windows ===
    final_scores = defaultdict(float)
    for _, scores in window_predictions:
        for label in class_labels:
            final_scores[label] += scores[label]
    for label in final_scores:
        final_scores[label] /= len(window_predictions)

    final_label = max(final_scores, key=final_scores.get)
    print(f"✅ Final label (sliding window avg): {final_label.capitalize()}")
    return final_label.capitalize()
