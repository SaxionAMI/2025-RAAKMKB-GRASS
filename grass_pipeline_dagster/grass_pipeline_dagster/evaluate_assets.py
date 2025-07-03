import os
import json
import torch
import tempfile
import mlflow
import pandas as pd
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from dagster import asset, get_dagster_logger
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from grass_pipeline_dagster.utils import load_settings, get_minio_client, get_influx_client


class SpectrogramTestDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        settings = load_settings()
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.class_labels = [label.lower() for label in settings["class_labels"]]
        self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['path']).convert("RGB")
        label = self.label_map[row['label'].lower()]
        return self.transform(image) if self.transform else image, label


@asset
def evaluate_model() -> dict:
    logger = get_dagster_logger()
    settings = load_settings()
    client = get_minio_client()
    influx = get_influx_client()
    bucket = "grass-pipeline"
    class_labels = settings["class_labels"]
    num_classes = len(class_labels)

    logger.info("Starting evaluate_model asset...")

    # === Locate latest versioned model path ===
    logger.info("Scanning MinIO for latest versioned model...")
    response = client.list_objects_v2(Bucket=bucket, Prefix="models/cnn/versions/")
    all_keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith("cnn_model.pth")]

    if not all_keys:
        raise FileNotFoundError("No versioned cnn_model.pth files found in MinIO.")

    version_folders = sorted({key.split("/")[3] for key in all_keys})
    timestamp = version_folders[-1]
    logger.info(f"Using versioned model: {timestamp}")

    model_key = f"models/cnn/versions/{timestamp}/cnn_model.pth"
    metrics_key = f"models/cnn/versions/{timestamp}/metrics.json"

    # === Load test.csv ===
    test_csv = os.path.join(tempfile.gettempdir(), "test_eval.csv")
    try:
        client.download_file(bucket, "datasets/test.csv", test_csv)
        logger.info("Loaded test.csv from MinIO.")
    except Exception:
        fallback = os.path.join("dagster_output", "dataset", "test.csv")
        if os.path.exists(fallback):
            test_csv = fallback
            logger.warning(f"Using local fallback for test.csv: {fallback}")
        else:
            raise FileNotFoundError("test.csv not found")

    # === Prepare test dataset ===
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_loader = DataLoader(SpectrogramTestDataset(test_csv, transform), batch_size=32, shuffle=False)

    # === Load model ===
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_temp = os.path.join(tempfile.gettempdir(), "cnn_model.pth")
    try:
        client.download_file(bucket, model_key, model_temp)
        logger.info(f"Downloaded model from MinIO: {model_key}")
    except:
        local_fallback = os.path.join("dagster_output", "model", "versions", timestamp, "cnn_model.pth")
        if not os.path.exists(local_fallback):
            raise FileNotFoundError("Model not found")
        model_temp = local_fallback
        logger.warning(f"Using local fallback model: {local_fallback}")

    model.load_state_dict(torch.load(model_temp, map_location=device))
    logger.info("Model loaded.")

    # === Inference ===
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # === Update current metrics.json with eval results ===
    local_metrics_path = os.path.join(tempfile.gettempdir(), "metrics_eval.json")
    client.download_file(bucket, metrics_key, local_metrics_path)
    with open(local_metrics_path, "r") as f:
        metrics_data = json.load(f)

    current_avg_loss = metrics_data.get("avg_loss", float("inf"))
    metrics_data.update({
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    with open(local_metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    client.upload_file(local_metrics_path, bucket, metrics_key)
    logger.info(f"Current model avg_loss: {current_avg_loss:.4f}")

    # === Load latest F1 for comparison ===
    try:
        latest_metrics_temp = os.path.join(tempfile.gettempdir(), "latest_metrics.json")
        client.download_file(bucket, "models/cnn/latest/metrics.json", latest_metrics_temp)
        with open(latest_metrics_temp) as f:
            latest_f1 = json.load(f).get("f1", -1)
        logger.info(f"Latest promoted model F1: {latest_f1:.4f}")
    except:
        latest_f1 = -1
        logger.info("No promoted model — assuming this is first run")

    # === Promotion decision ===
    promoted = f1 > latest_f1
    if promoted:
        logger.info("Promoting model to latest/")
        client.upload_file(model_temp, bucket, "models/cnn/latest/cnn_model.pth")
        client.upload_file(local_metrics_path, bucket, "models/cnn/latest/metrics.json")
    else:
        logger.info("Model not promoted (F1 not improved)")

    # === MLflow ===
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"CrowdSentimentEval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with mlflow.start_run() as run:
        logger.info(f"MLflow run: {run.info.run_id}")
        mlflow.set_tags({
            "stage": "evaluation",
            "timestamp": timestamp,
            "promoted": promoted
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_loss": current_avg_loss
        })
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("dataset_size", len(test_loader.dataset))
        mlflow.set_tag("model_type", "ResNet18")
        cm_path = f"dagster_output/eval/confusion_matrix_{timestamp}.csv"
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

    # === InfluxDB ===
    influx.write_api(write_options=SYNCHRONOUS).write(
        bucket="predictions",
        org="grass",
        record=Point("evaluation_metrics")
            .tag("model", "ResNet18")
            .field("accuracy", acc)
            .field("precision", precision)
            .field("recall", recall)
            .field("f1_score", f1)
            .field("avg_loss", current_avg_loss)
            .field("promoted", promoted)
            .field("samples", len(test_loader.dataset))
            .time(datetime.utcnow())
    )

    logger.info("evaluate_model asset completed.")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_loss": current_avg_loss,
        "promoted": promoted,
        "timestamp": timestamp,
        "confusion_matrix": cm.tolist()
    }
