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
import matplotlib.pyplot as plt

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
        image = self.transform(image) if self.transform else image
        return image, label


@asset
def evaluate_model() -> dict:
    logger = get_dagster_logger()
    settings = load_settings()
    client = get_minio_client()
    influx = get_influx_client()
    bucket = "grass-pipeline"
    class_labels = [label.lower() for label in settings["class_labels"]]
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
            raise FileNotFoundError("test.csv not found in MinIO or locally")

    # === Prepare test dataset ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
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
    except Exception:
        local_fallback = os.path.join("dagster_output", "model", "versions", timestamp, "cnn_model.pth")
        if not os.path.exists(local_fallback):
            raise FileNotFoundError("Model not found in MinIO or locally")
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
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"[Weighted] P: {precision_w:.4f} | R: {recall_w:.4f} | F1: {f1_w:.4f}")
    logger.info(f"Macro-F1 (promotion metric): {f1_macro:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # === Update metrics.json ===
    local_metrics_path = os.path.join(tempfile.gettempdir(), "metrics_eval.json")
    client.download_file(bucket, metrics_key, local_metrics_path)
    with open(local_metrics_path, "r") as f:
        metrics_data = json.load(f)

    current_avg_loss = metrics_data.get("avg_loss", float("inf"))
    metrics_data.update({
        "accuracy": acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro
    })
    with open(local_metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    client.upload_file(local_metrics_path, bucket, metrics_key)

    # === Load latest Macro-F1 for promotion check ===
    try:
        latest_metrics_temp = os.path.join(tempfile.gettempdir(), "latest_metrics.json")
        client.download_file(bucket, "models/cnn/latest/metrics.json", latest_metrics_temp)
        with open(latest_metrics_temp) as f:
            latest_f1 = json.load(f).get("f1_macro", -1)
        logger.info(f"Latest promoted model Macro-F1: {latest_f1:.4f}")
    except:
        latest_f1 = -1
        logger.info("No promoted model — assuming this is first run")

    # === Promotion decision (Macro-F1) ===
    promoted = f1_macro > latest_f1
    if promoted:
        logger.info("Promoting model to latest/ (based on Macro-F1)")
        client.upload_file(model_temp, bucket, "models/cnn/latest/cnn_model.pth")
        client.upload_file(local_metrics_path, bucket, "models/cnn/latest/metrics.json")

        try:
            os.makedirs("dagster_output/model/latest", exist_ok=True)
            torch.save(model.state_dict(), "dagster_output/model/latest/cnn_model.pth")
            with open("dagster_output/model/latest/metrics.json", "w") as f:
                json.dump(metrics_data, f, indent=4)
            logger.info("Model and metrics also saved locally under latest/")
        except Exception as e:
            logger.warning(f"Failed to save latest model locally: {e}")
    else:
        logger.info("Model not promoted (Macro-F1 not improved)")

    # === Build evaluation manifest (dataset traceability) ===
    try:
        df_eval = pd.read_csv(test_csv)
        class_dist = (
            df_eval["label"].str.lower().value_counts().reindex(class_labels, fill_value=0).to_dict()
        )
        eval_manifest = {
            "type": "evaluation_manifest",
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "version_timestamp": timestamp,
            "dataset": {
                "csv_path": os.path.abspath(test_csv),
                "rows": int(len(df_eval)),
                "class_labels": class_labels,
                "class_distribution": class_dist,
            },
        }
        eval_manifest_path = os.path.join(tempfile.gettempdir(), f"evaluation_manifest_{timestamp}.json")
        with open(eval_manifest_path, "w") as f:
            json.dump(eval_manifest, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to build evaluation manifest: {e}")
        eval_manifest_path = None

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
            "precision_weighted": precision_w,
            "recall_weighted": recall_w,
            "f1_weighted": f1_w,
            "f1_macro": f1_macro,
            "avg_loss": current_avg_loss
        })
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("dataset_size", len(test_loader.dataset))
        mlflow.set_tag("model_type", "ResNet18")

        cm_path = f"dagster_output/eval/confusion_matrix_{timestamp}.csv"
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # === NEW (visual artifacts) ===
        try:
            cm_fig_path = f"dagster_output/eval/confusion_matrix_{timestamp}.png"
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest", cmap="coolwarm")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha="right")
            plt.yticks(range(len(class_labels)), class_labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(cm_fig_path, dpi=150)
            plt.close()
            mlflow.log_artifact(cm_fig_path, artifact_path="confusion_matrix")

            metrics_fig_path = f"dagster_output/eval/metrics_{timestamp}.png"
            plt.figure(figsize=(6, 4))
            metric_names = ["accuracy", "precision_w", "recall_w", "f1_w", "f1_macro"]
            metric_values = [acc, precision_w, recall_w, f1_w, f1_macro]
            colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]
            plt.bar(metric_names, metric_values, color=colors)
            plt.ylim(0, 1)
            plt.title("Evaluation Metrics")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(metrics_fig_path, dpi=150)
            plt.close()
            mlflow.log_artifact(metrics_fig_path, artifact_path="metrics")
        except Exception as e:
            logger.warning(f"Failed to generate/log visual artifacts: {e}")

        mlflow.log_artifact(test_csv, artifact_path="dataset")
        if eval_manifest_path and os.path.exists(eval_manifest_path):
            mlflow.log_artifact(eval_manifest_path, artifact_path="dataset")

    # === InfluxDB ===
    influx.write_api(write_options=SYNCHRONOUS).write(
        bucket="predictions",
        org="grass",
        record=Point("evaluation_metrics")
            .tag("model", "ResNet18")
            .field("accuracy", acc)
            .field("precision_weighted", precision_w)
            .field("recall_weighted", recall_w)
            .field("f1_weighted", f1_w)
            .field("f1_macro", f1_macro)
            .field("avg_loss", current_avg_loss)
            .field("promoted", promoted)
            .field("samples", len(test_loader.dataset))
            .time(datetime.utcnow())
    )

    logger.info("evaluate_model asset completed.")
    return {
        "accuracy": acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
        "avg_loss": current_avg_loss,
        "promoted": promoted,
        "timestamp": timestamp,
        "confusion_matrix": cm.tolist()
    }
