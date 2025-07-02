import os
import torch
import pandas as pd
import mlflow
import tempfile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from PIL import Image
from datetime import datetime
from dagster import asset, get_dagster_logger

from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from grass_pipeline_dagster.utils import load_settings, get_influx_client, get_minio_client


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
        img_path = self.data.iloc[idx]['path']
        label_str = self.data.iloc[idx]['label'].lower()
        label = self.label_map[label_str]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


@asset
def evaluate_model() -> dict:
    logger = get_dagster_logger()
    settings = load_settings()
    class_labels = [label.lower() for label in settings["class_labels"]]
    num_classes = len(class_labels)

    client = get_minio_client()
    minio_bucket = "grass-pipeline"
    minio_model_object = "models/cnn/latest/cnn_model.pth"
    minio_test_object = "datasets/test.csv"

    # === Try loading test.csv from MinIO first ===
    try:
        test_csv = os.path.join(tempfile.gettempdir(), "test_eval.csv")
        client.download_file(minio_bucket, minio_test_object, test_csv)
        logger.info(f"Loaded test.csv from MinIO: {minio_test_object}")
    except Exception as e:
        logger.warning(f"Could not load test.csv from MinIO: {e}")
        local_fallback = os.path.join(os.getcwd(), "dagster_output", "dataset", "test.csv")
        if os.path.exists(local_fallback):
            test_csv = local_fallback
            logger.info(f"Using fallback local test.csv: {test_csv}")
        else:
            logger.error("No test.csv found in MinIO or locally. Evaluation aborted.")
            raise FileNotFoundError("Test CSV not found in MinIO or local path.")

    # === Prepare Dataset ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SpectrogramTestDataset(test_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # === Model Loading ===
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_loaded = False
    try:
        temp_model_path = os.path.join(tempfile.gettempdir(), "cnn_model_eval.pth")
        client.download_file(minio_bucket, minio_model_object, temp_model_path)
        model.load_state_dict(torch.load(temp_model_path, map_location=device))
        logger.info(f"Loaded model from MinIO: {minio_model_object}")
        model_loaded = True
    except Exception as e:
        logger.warning(f"Could not load model from MinIO: {e}")
        local_model_path = os.path.join(os.getcwd(), "dagster_output", "model", "cnn_model.pth")
        if os.path.exists(local_model_path):
            model.load_state_dict(torch.load(local_model_path, map_location=device))
            logger.info(f"Loaded model from local path: {local_model_path}")
            model_loaded = True
        else:
            logger.error("No trained model found locally or in MinIO. Evaluation aborted.")
            raise FileNotFoundError("Trained model not found in MinIO or local path.")

    model.eval()

    y_true, y_pred = [], []
    logger.info("Running inference on test set")

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # ===== MLflow Logging =====
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = f"CrowdSentimentEval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow evaluation run: {run_id}")

        mlflow.set_tags({
            "stage": "evaluation",
            "model_type": "ResNet18",
            "developer": "Ahmed",
        })

        mlflow.log_params({
            "num_classes": num_classes,
            "dataset_size": len(dataset)
        })

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        cm_path = os.path.join(os.getcwd(), "dagster_output", "eval", "confusion_matrix.csv")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="evaluation")

    # ===== InfluxDB Logging =====
    influx = get_influx_client()
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    point = (
        Point("evaluation_metrics")
        .tag("model", "ResNet18")
        .tag("stage", "evaluation")
        .tag("developer", "Ahmed")
        .field("accuracy", float(acc))
        .field("precision", float(precision))
        .field("recall", float(recall))
        .field("f1_score", float(f1))
        .field("samples", len(dataset))
        .time(datetime.utcnow())
    )

    write_api.write(bucket="predictions", org="grass", record=point)
    logger.info("Logged evaluation metrics to InfluxDB.")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
