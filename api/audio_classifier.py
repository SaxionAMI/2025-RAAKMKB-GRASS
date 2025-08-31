import matplotlib
matplotlib.use('Agg')

import os
import sys
import math
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
segment_overlap  = float(settings["segment_overlap"])

# --- Derive hop-aware dynamics (no magic numbers) ---
hop_sec = max(1e-6, segment_duration - segment_overlap)
H_MIN = max(1, math.ceil(segment_duration / hop_sec))                 # min region length in hops
DWELL_HOPS = H_MIN                                                    # require this many consecutive hops to switch
BRIDGE_K_HOPS = max(0, math.ceil(segment_overlap / hop_sec))          # bridge pinholes up to overlap width

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
    return lbl.capitalize()

# ====== Hop-grid, labels-only smoothing helpers ======

def _stable_labels_with_dwell(hop_labels, dwell_hops: int):
    """
    Require a new candidate label to persist for 'dwell_hops' consecutive hops
    before switching. Purely labels-based (class- and confidence-agnostic).
    """
    if not hop_labels:
        return []
    stable = [hop_labels[0]]
    cur = hop_labels[0]
    streak = 0
    for lab in hop_labels[1:]:
        if lab == cur:
            streak = 0
        else:
            streak += 1
            if streak >= dwell_hops:
                cur = lab
                streak = 0
        stable.append(cur)
    return stable

def _hops_to_regions(hop_labels, hop_sec: float, segment_duration: float):
    """
    Convert per-hop labels to contiguous regions for LS.
    Boundaries snap to hop starts; last region ends at last hop start + segment_duration.
    """
    regions = []
    if not hop_labels:
        return regions
    cur = hop_labels[0]
    start = 0.0
    for i in range(1, len(hop_labels)):
        if hop_labels[i] != cur:
            end = i * hop_sec
            regions.append({"start": start, "end": end, "label": _capitalize_label(cur), "score": 0.0})
            cur = hop_labels[i]
            start = end
    end = (len(hop_labels) - 1) * hop_sec + segment_duration
    regions.append({"start": start, "end": end, "label": _capitalize_label(cur), "score": 0.0})
    return regions

def _bridge_pinhole(regions, hop_sec: float, k_hops: int):
    """
    Bridge tiny middle regions of <= k_hops where left and right labels match.
    """
    if len(regions) < 3 or k_hops <= 0:
        return regions
    max_gap = k_hops * hop_sec
    i = 1
    while i < len(regions) - 1:
        L, M, R = regions[i - 1], regions[i], regions[i + 1]
        if L["label"].lower() == R["label"].lower() and (M["end"] - M["start"]) <= max_gap:
            L["end"] = R["end"]
            del regions[i:i + 2]
        else:
            i += 1
    return regions

def _enforce_min_region_hops(regions, hop_sec: float, segment_duration: float):
    """
    Ensure each region spans at least H_MIN hops (derived from settings).
    Prefer L+M+R merge when L==R; otherwise absorb into the longer neighbor.
    """
    if len(regions) <= 1:
        return regions
    H_MIN_local = max(1, math.ceil(segment_duration / hop_sec))

    def hops(r):
        return max(1, int(round((r["end"] - r["start"]) / hop_sec + 1e-6)))
    def dur(r):
        return r["end"] - r["start"]

    i = 0
    while i < len(regions):
        r = regions[i]
        if hops(r) >= H_MIN_local:
            i += 1
            continue

        left  = regions[i - 1] if i - 1 >= 0 else None
        right = regions[i + 1] if i + 1 < len(regions) else None

        if left and right and left["label"].lower() == right["label"].lower():
            left["end"] = right["end"]
            del regions[i:i + 2]
            continue

        if left and (not right or dur(left) >= dur(right)):
            left["end"] = max(left["end"], r["end"])
            del regions[i]
        elif right:
            right["start"] = min(right["start"], r["start"])
            del regions[i]
        else:
            i += 1
    return regions

# ====== Main classification ======

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
    segments = segment_audio(audio_np, sample_rate, segment_duration, hop_sec)
    print(f"{0 if segments is None else len(segments)} segments created")
    if len(segments) == 0:
        raise RuntimeError("No segments available after segmentation")

    # --- Per-segment inference (labels only; confidence ignored) ---
    hop_labels = []
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
            hop_labels.append(pred_label)

            print("Seg {}: ".format(i) + ", ".join(
                f"{lbl}={probs[j]:.2f}" for j, lbl in enumerate(class_labels)
            ) + f" | chosen={pred_label}")

        except Exception as e:
            print(f"⚠️ Segment {i} failed:", e)
            # fallback: keep timeline length; repeat previous or default to first class
            hop_labels.append(hop_labels[-1] if hop_labels else class_labels[0])
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    # --- Labels-only smoothing on hop grid ---
    stable_labels = _stable_labels_with_dwell(hop_labels, dwell_hops=DWELL_HOPS)

    # --- Convert to LS regions, then fix tiny artifacts ---
    regions = _hops_to_regions(stable_labels, hop_sec, segment_duration)
    regions = _bridge_pinhole(regions, hop_sec, BRIDGE_K_HOPS)
    regions = _enforce_min_region_hops(regions, hop_sec, segment_duration)

    print(f"Generated {len(regions)} regions: {regions}")

    # LS start/end/label
    return [{"start": r["start"], "end": r["end"], "label": r["label"]} for r in regions]
