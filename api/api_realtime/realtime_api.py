import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
from grass_pipeline_dagster.utils import load_settings, get_influx_client
from datetime import datetime, timezone
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from time import perf_counter
import os
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque, Counter
import torchaudio  # <-- use same audio path as training

app = FastAPI()

# Global objects
model = None
settings = None
transform = None
label_map = None
inverse_label_map = None
influx = None
write_api = None
TARGET_SR = 44100  # must match training

# Sliding window buffers (for smoothing on server-side if needed)
label_buffer = deque(maxlen=5)
confidence_buffer = deque(maxlen=5)

@app.on_event("startup")
def load_model():
    global model, settings, transform, label_map, inverse_label_map, influx, write_api, TARGET_SR

    settings = load_settings()
    model_path = os.path.abspath(settings["model_path"])
    # Optionally allow settings to override SR
    TARGET_SR = int(settings.get("target_sr", TARGET_SR))

    num_classes = len(settings["class_labels"])
    # Use same API family as training; checkpoint will set final weights
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Match training transforms (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    label_map = {label.lower(): i for i, label in enumerate(settings["class_labels"])}
    inverse_label_map = {i: label for label, i in label_map.items()}

    influx = get_influx_client()
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    # Grafana label mapping
    try:
        with open("config/grafana_label_order.json", "r") as f:
            app.state.grafana_label_map = json.load(f)
        print("Loaded Grafana label mapping.")
        print("Grafana label map (for verification):", app.state.grafana_label_map)
    except Exception as e:
        print("⚠️ Failed to load grafana_label_order.json:", e)
        app.state.grafana_label_map = {}

    print("Real-time model and InfluxDB logger ready")

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    source: str = Form(...),
    session_id: str = Form("default_session"),
    segment_idx: int = Form(0)
):
    try:
        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ---- Audio load & resample (match training path via torchaudio) ----
        waveform, sample_rate = torchaudio.load(tmp_path)  # [C, T], float32 in [-1, 1]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if sample_rate != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)
            sample_rate = TARGET_SR

        audio_np = waveform.squeeze(0).to(torch.float32).numpy()
        # Normalize like training (zero mean / unit variance)
        audio_np = (audio_np - np.mean(audio_np)) / (np.std(audio_np) + 1e-8)

        # --- segmentation params (interpret overlap as seconds) ---
        segment_duration = float(settings["segment_duration"])     # window (s), e.g., 1.0
        segment_overlap = float(settings["segment_overlap"])       # desired overlap (s), e.g., 0.25
        hop_sec = max(1e-6, segment_duration - segment_overlap)    # step = window - overlap

        # Call segmenter with hop (step) seconds
        segments = segment_audio(audio_np, sample_rate, segment_duration, hop_sec)
        # normalize to list to avoid NumPy truth-value ambiguity
        if isinstance(segments, np.ndarray):
            segments = list(segments)
        if segments is None or len(segments) == 0:
            return JSONResponse(content={"error": "No segments found"}, status_code=400)


        predictions = []
        influx_points = []

        inference_start = perf_counter()

        for i, segment in enumerate(segments):
            try:
                start = perf_counter()

                # Per-segment normalization safety (as in training code)
                segment = (segment - np.mean(segment)).astype(np.float32)
                segment /= (np.std(segment) + 1e-8)

                # Compute spectrogram and render exactly like training (Matplotlib + 'jet')
                spec = compute_spectrogram(segment, sample_rate)
                log_spec = 10 * np.log10(spec + 1e-10)
                log_spec -= np.max(log_spec)
                log_spec = np.clip(log_spec, -80, 0)

                # Use matplotlib route to preserve exact training parity
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                    image_path = tmp_img.name
                    plt.figure(figsize=(4, 4))
                    plt.axis("off")
                    plt.imshow(
                        log_spec,
                        origin='lower',
                        aspect='auto',
                        cmap='jet',
                        extent=[0, segment_duration, 0, sample_rate / 2]
                    )
                    plt.tight_layout()
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                image = Image.open(image_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0)
                os.remove(image_path)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).squeeze().numpy()

                label_idx = int(np.argmax(probs))
                label = inverse_label_map[label_idx]
                confidence = float(probs[label_idx])
                latency_ms = round((perf_counter() - start) * 1000, 2)
                timestamp = datetime.now(timezone.utc)

                # Smoothing buffers
                label_buffer.append(label.lower())
                confidence_buffer.append(confidence)
                smoothed_label = Counter(label_buffer).most_common(1)[0][0] if label_buffer else label.lower()
                smoothed_conf = round(np.mean(confidence_buffer), 4) if confidence_buffer else confidence

                grafana_map = app.state.grafana_label_map
                label_numeric = grafana_map.get(label.lower(), -1)
                smoothed_label_numeric = grafana_map.get(smoothed_label.lower(), -1)

                point = (
                    Point("sentiment_realtime")
                    .tag("source", source)
                    .tag("label", label.lower())
                    .tag("smoothed_label", smoothed_label)
                    .tag("session", session_id)
                    .field("confidence", round(confidence, 4))
                    .field("smoothed_confidence", smoothed_conf)
                    .field("label_numeric", label_numeric)
                    .field("smoothed_label_numeric", smoothed_label_numeric)
                    .field("latency_ms", latency_ms)
                    .field("segment", segment_idx + i)
                    .time(timestamp, WritePrecision.MS)
                )
                influx_points.append(point)

                predictions.append({
                    "segment": i,
                    "label": label.capitalize(),
                    "confidence": round(confidence, 4),
                    "latency_ms": latency_ms,
                    "smoothed_label": smoothed_label.capitalize(),
                    "smoothed_confidence": smoothed_conf
                })

            except Exception as e:
                predictions.append({
                    "segment": i,
                    "error": str(e)
                })

        inference_end = perf_counter()
        inference_time_ms = round((inference_end - inference_start) * 1000, 2)

        influx_log_time = -1
        try:
            if influx_points:
                t0 = perf_counter()
                write_api.write(bucket="predictions", org="grass", record=influx_points)
                influx_log_time = round((perf_counter() - t0) * 1000, 2)
        except Exception as influx_error:
            print("InfluxDB write failed:", influx_error)

        return {
            "segments_processed": len(predictions),
            "avg_latency_ms": round(np.mean([p["latency_ms"] for p in predictions if "latency_ms" in p]), 2),
            "inference_time_ms": inference_time_ms,
            "influx_log_time_ms": influx_log_time,
            "predictions": predictions
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
