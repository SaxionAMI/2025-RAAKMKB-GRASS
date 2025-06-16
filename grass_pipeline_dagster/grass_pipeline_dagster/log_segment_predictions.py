import os
import torch
import pandas as pd
import tempfile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from dagster import asset, get_dagster_logger
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from grass_pipeline_dagster.utils import (
    load_settings,
    get_influx_client,
    get_minio_client
)


class SegmentPredictionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        settings = load_settings()
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.class_labels = [label.lower() for label in settings["class_labels"]]
        self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        label_str = self.data.iloc[idx]['label'].lower()
        label = self.label_map[label_str]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # Return path for logging


@asset
def log_segment_predictions():
    logger = get_dagster_logger()
    settings = load_settings()
    class_labels = [label.lower() for label in settings["class_labels"]]
    num_classes = len(class_labels)

    minio_bucket = "grass-pipeline"
    minio_model_key = "models/cnn/latest/cnn_model.pth"
    minio_test_key = "datasets/test.csv"
    client = get_minio_client()

    # === Try loading test.csv from MinIO first ===
    try:
        test_csv = os.path.join(tempfile.gettempdir(), "test_log.csv")
        client.download_file(minio_bucket, minio_test_key, test_csv)
        logger.info(f"✅ Loaded test.csv from MinIO: {minio_test_key}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load test.csv from MinIO: {e}")
        fallback_csv = os.path.join(os.getcwd(), "dagster_output", "dataset", "test.csv")
        if os.path.exists(fallback_csv):
            test_csv = fallback_csv
            logger.info(f"✅ Using fallback local test.csv: {test_csv}")
        else:
            logger.error("❌ No test.csv found in MinIO or locally. Aborting.")
            raise FileNotFoundError("test.csv not found.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SegmentPredictionDataset(test_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # === Try loading model from MinIO first ===
    try:
        model_path = os.path.join(tempfile.gettempdir(), "cnn_model_predict.pth")
        client.download_file(minio_bucket, minio_model_key, model_path)
        logger.info(f"✅ Loaded model from MinIO: {minio_model_key}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load model from MinIO: {e}")
        fallback_model = os.path.join(os.getcwd(), "dagster_output", "model", "cnn_model.pth")
        if os.path.exists(fallback_model):
            model_path = fallback_model
            logger.info(f"✅ Using fallback local model: {model_path}")
        else:
            logger.error("❌ No trained model found in MinIO or locally. Aborting.")
            raise FileNotFoundError("Model not found.")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    influx = get_influx_client()
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    logger.info("Running inference and logging predictions to InfluxDB")
    with torch.no_grad():
        for image, true_label, img_path in dataloader:
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

            pred_label = dataset.inverse_label_map[pred.item()]
            true_label_name = dataset.inverse_label_map[true_label.item()]
            confidence_score = float(confidence.item())

            point = (
                Point("segment_predictions")
                .tag("predicted_label", pred_label)
                .tag("true_label", true_label_name)
                .field("confidence", confidence_score)
                .field("segment_path", img_path[0])
                .time(datetime.utcnow())
            )

            write_api.write(bucket="predictions", org="grass", record=point)

    logger.info("✅ Logged all segment-level predictions to InfluxDB.")
