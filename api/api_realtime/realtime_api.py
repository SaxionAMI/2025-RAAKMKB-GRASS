import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
from grass_pipeline_dagster.utils import load_settings, get_influx_client
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from time import perf_counter
import os
import uuid
import numpy as np
from scipy.io import wavfile
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque, Counter

app = FastAPI()

# Global objects
model = None
settings = None
transform = None
label_map = None
inverse_label_map = None
influx = None
write_api = None

# Sliding window buffers (for smoothing on server-side if needed)
label_buffer = deque(maxlen=5)
confidence_buffer = deque(maxlen=5)

@app.on_event("startup")
def load_model():
    global model, settings, transform, label_map, inverse_label_map, influx, write_api

    settings = load_settings()
    model_path = os.path.abspath(settings["model_path"])

    num_classes = len(settings["class_labels"])
    model = torch.hub.load('pytorch/vision:v0.14.0', 'resnet18', weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    label_map = {label.lower(): i for i, label in enumerate(settings["class_labels"])}
    inverse_label_map = {i: label for label, i in label_map.items()}

    influx = get_influx_client()
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    #  Grafana label mapping
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        sample_rate, audio_np = wavfile.read(tmp_path)
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        audio_np = (audio_np - np.mean(audio_np)).astype(np.float32)
        audio_np /= (np.std(audio_np) + 1e-8)

        segment_duration = float(settings["segment_duration"])
        segment_overlap = float(settings["segment_overlap"])
        segments = segment_audio(audio_np, sample_rate, segment_duration, segment_overlap)
        if segments is None or len(segments) == 0:
            return JSONResponse(content={"error": "No segments found"}, status_code=400)

        predictions = []
        influx_points = []

        inference_start = perf_counter()

        for i, segment in enumerate(segments):
            try:
                start = perf_counter()

                spec = compute_spectrogram(segment, sample_rate)
                log_spec = 10 * np.log10(spec + 1e-10)
                log_spec -= np.max(log_spec)
                log_spec = np.clip(log_spec, -80, 0)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                    image_path = tmp_img.name
                    plt.figure(figsize=(4, 4))
                    plt.axis("off")
                    plt.imshow(log_spec, origin='lower', aspect='auto', cmap='jet',
                               extent=[0, segment_duration, 0, sample_rate / 2])
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
                    .field("segment", segment_idx)
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
