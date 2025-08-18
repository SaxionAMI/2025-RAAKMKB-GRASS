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

# --- tiny, optional guards (tunable via env; safe defaults) ---
MIN_CONF = float(os.getenv("LS_MIN_CONF", "0.0"))               # e.g., 0.5 to require stronger predictions
BRIDGE_NEUTRAL_GAP_SEC = float(os.getenv("LS_BRIDGE_NEUTRAL_GAP_SEC", "0.0"))  # e.g., 0.5 to merge tiny neutral gaps
NEUTRAL_LABEL = "neutral"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))

def try_load_from_minio():
    """
    Try to download and load the latest model from MinIO.
    Returns True if successful, False otherwise.
    """
    minio_bucket = "grass-pipeline"
    minio_object = "models/cnn/latest/cnn_model.pth"
    client = get_minio_client()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        try:
            client.download_file(minio_bucket, minio_object, tmp_file.name)
            print(f"Downloaded model from MinIO: {minio_bucket}/{minio_object}")
            state_dict = torch.load(tmp_file.name, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("Loaded model from MinIO.")
            return True
        except Exception as e:
            print(f"Could not load model from MinIO: {e}")
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
        print(f"Falling back to local model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MinIO and local: {e}")

def _capitalize_label(lbl: str) -> str:
    # 'disapproval' -> 'Disapproval', etc.
    return lbl.capitalize()

def classify_audio(audio_path: str):
    """
    Classify an audio file into consecutive sentiment regions.
    Returns: list of {"start": float, "end": float, "label": str}
    """
    print("classify_audio called with:", audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # --- Load audio ---
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

    # --- Normalize ---
    audio_np = (audio_np - np.mean(audio_np)).astype(np.float32)
    audio_np /= (np.std(audio_np) + 1e-8)

    # --- Segment ---
    hop_sec = max(1e-6, segment_duration - segment_overlap)
    segments = segment_audio(audio_np, sample_rate, segment_duration, hop_sec)
    print(f"{0 if segments is None else len(segments)} segments created")
    if len(segments) == 0:
        raise RuntimeError("No segments available after segmentation")

    # --- Per-segment inference ---
    segment_preds = []
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
                plt.imshow(
                    log_spec,
                    origin="lower",
                    aspect="auto",
                    cmap="jet",
                    extent=[0, segment_duration, 0, sample_rate / 2],
                )
                plt.tight_layout()
                plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
                plt.close()

            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = class_labels[pred_idx]
            confidence = float(probs[pred_idx])

            # Optional confidence guard: low-conf -> neutral
            if MIN_CONF > 0.0 and confidence < MIN_CONF:
                pred_label = NEUTRAL_LABEL

            t0 = i * hop_sec
            t1 = t0 + segment_duration
            segment_preds.append({
                "start": t0,
                "end": t1,
                "label": pred_label,
                "conf": confidence
            })

            print("Seg {}: ".format(i) + ", ".join(
                f"{lbl}={probs[j]:.2f}" for j, lbl in enumerate(class_labels)
            ) + f" | chosen={pred_label} ({confidence:.2f})")

        except Exception as e:
            print(f"⚠️ Segment {i} failed:", e)
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    if not segment_preds:
        raise RuntimeError("No segment predictions")

    # --- Collapse consecutive same labels into non-overlapping regions ---
    regions = []
    current = segment_preds[0].copy()
    for seg in segment_preds[1:]:
        if seg["label"] != current["label"]:
            # boundary at start of new segment -> no overlap
            regions.append({
                "start": current["start"],
                "end": seg["start"],
                "label": _capitalize_label(current["label"]),
                "score": current.get("conf", 0.0)
            })
            current = seg.copy()
        else:
            # extend current region to this segment's end
            current["end"] = seg["end"]
            current["conf"] = max(current.get("conf", 0.0), seg.get("conf", 0.0))
    # close last region
    regions.append({
        "start": current["start"],
        "end": current["end"],
        "label": _capitalize_label(current["label"]),
        "score": current.get("conf", 0.0)
    })

    # --- Optional: bridge tiny NEUTRAL gaps between same labels on both sides ---
    if BRIDGE_NEUTRAL_GAP_SEC > 0 and len(regions) >= 3:
        changed = True
        while changed:
            changed = False
            i = 1
            while i < len(regions) - 1:
                mid = regions[i]
                left = regions[i - 1]
                right = regions[i + 1]
                if (mid["label"].lower() == NEUTRAL_LABEL and
                    (mid["end"] - mid["start"]) <= BRIDGE_NEUTRAL_GAP_SEC and
                    left["label"] == right["label"]):
                    # merge left + mid + right into one region
                    left["end"] = right["end"]
                    # drop mid and right
                    del regions[i:i+2]
                    changed = True
                else:
                    i += 1

    print(f"Generated {len(regions)} regions: {regions}")
    # LS only needs start/end/label; score kept for debugging if you want it upstream
    return [{"start": r["start"], "end": r["end"], "label": r["label"]} for r in regions]
