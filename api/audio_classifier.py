import matplotlib
matplotlib.use('Agg')

import os
import sys
import math
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torchaudio  # <- use torchaudio for all formats

# adding project root to sys.path ===
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

# --- Derive hop-aware dynamics ---
hop_sec = max(1e-6, segment_duration - segment_overlap)
H_MIN = max(1, math.ceil(segment_duration / hop_sec))                 # min region length in hops
DWELL_HOPS = H_MIN                                                    # require this many consecutive hops to switch
BRIDGE_K_HOPS = max(0, math.ceil(segment_overlap / hop_sec))          # bridge pinholes up to overlap width

# --- Match train-time normalization (ImageNet) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))

# --- Model resolution: LS -> CNN -> local ---
MINIO_BUCKET = "grass-pipeline"
LS_KEY  = "models/ls/latest/ls_cnn_model.pth"
CNN_KEY = "models/cnn/latest/cnn_model.pth"

def _try_stat(key: str) -> bool:
    try:
        get_minio_client().head_object(Bucket=MINIO_BUCKET, Key=key)
        return True
    except Exception:
        return False

def _download_minio(key: str) -> str:
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    get_minio_client().download_file(MINIO_BUCKET, key, tmp)
    return tmp

def resolve_checkpoint() -> tuple[str, str]:
    """
    Returns (source_label, local_path_to_checkpoint)
    Priority: LS -> CNN -> local model_path
    """
    if _try_stat(LS_KEY):
        return ("ls", _download_minio(LS_KEY))
    if _try_stat(CNN_KEY):
        return ("cnn", _download_minio(CNN_KEY))
    if os.path.isfile(model_path):
        return ("local", model_path)
    raise RuntimeError(
        "No suitable checkpoint found (LS, CNN, local). "
        "Seed a baseline at models/cnn/latest/ or ensure settings['model_path'] exists."
    )

# --- Resolve checkpoint in priority: LS -> CNN -> local ---
model_src, ckpt = resolve_checkpoint()
state_dict = torch.load(ckpt, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[MODEL] Loaded source={model_src} path={ckpt}")

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

    # --- Load audio (torchaudio for all formats) ---
    try:
        waveform, sample_rate = torchaudio.load(audio_path)  # [C, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
    except Exception as e:
        raise RuntimeError(f"Failed to read audio: {e}")

    # --- Enforce target sample rate to match training/segmentation ---
    TARGET_SR = 44100
    if sample_rate != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)
        sample_rate = TARGET_SR

    # --- To numpy + normalize (same recipe as segmentation/training) ---
    audio_np = waveform.squeeze(0).to(torch.float32).numpy()
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
            hop_labels.append(hop_labels[-1] if hop_labels else class_labels[0])
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    # --- Labels smoothing ---
    stable_labels = _stable_labels_with_dwell(hop_labels, dwell_hops=DWELL_HOPS)

    # --- Convert to LS regions, then fix tiny artifacts ---
    regions = _hops_to_regions(stable_labels, hop_sec, segment_duration)
    regions = _bridge_pinhole(regions, hop_sec, BRIDGE_K_HOPS)
    regions = _enforce_min_region_hops(regions, hop_sec, segment_duration)

    print(f"Generated {len(regions)} regions: {regions}")

    # LS start/end/label
    return [{"start": r["start"], "end": r["end"], "label": r["label"]} for r in regions]
