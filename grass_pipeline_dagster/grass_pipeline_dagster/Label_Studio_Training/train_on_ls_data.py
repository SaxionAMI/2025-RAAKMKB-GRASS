import os
import json
import tempfile
from datetime import datetime
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from dagster import get_dagster_logger

from grass_pipeline_dagster.utils import load_settings, get_minio_client


class IncrementalSpectrogramDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        settings = load_settings()
        self.class_labels = [label.lower() for label in settings["class_labels"]]
        self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label_str = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_map[label_str.lower()]


def upload_to_minio(local_path, bucket, remote_path):
    client = get_minio_client()
    try:
        client.upload_file(local_path, bucket, remote_path)
    except Exception as e:
        raise RuntimeError(f"Failed to upload {local_path} to MinIO ({bucket}/{remote_path}): {e}")


def train_on_ls_data(segments: list[tuple[str, str]]) -> str:
    logger = get_dagster_logger()
    settings = load_settings()
    class_labels = settings["class_labels"]
    num_classes = len(class_labels)
    num_epochs = settings.get("ls_training_epochs", 3)

    if not segments:
        raise ValueError("No segments provided for training.")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"/app/dagster_output/label_studio/model/{now}"
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "ls_cnn_model.pth")
    metrics_output_path = os.path.join(model_dir, "metrics.json")

    logger.info(f"Received {len(segments)} corrected LS segments for training.")

    # === DataLoader ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = IncrementalSpectrogramDataset(segments, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # === Class balancing ===
    label_series = [label.lower() for _, label in segments]
    label_counts = {lbl: label_series.count(lbl) for lbl in class_labels}
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else 1
    use_weighted_loss = imbalance_ratio > 1.5

    if use_weighted_loss:
        total = sum(label_counts.values())
        weights = [total / (len(class_labels) * label_counts.get(lbl, 1)) for lbl in class_labels]
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
    else:
        class_weights = None
        logger.info("Using unweighted loss.")

    # === Model ===
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Resume from LS latest ===
    resumed = False
    minio_bucket = "grass-pipeline"
    minio_latest = "models/ls/latest/ls_cnn_model.pth"
    try:
        tmp_model_path = os.path.join(tempfile.gettempdir(), "resume_ls_model.pth")
        get_minio_client().download_file(minio_bucket, minio_latest, tmp_model_path)
        model.load_state_dict(torch.load(tmp_model_path, map_location=device))
        logger.info(f"Resumed model from MinIO: {minio_latest}")
        resumed = True
    except Exception as e:
        logger.warning(f"Could not resume model from MinIO: {e}")

    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if use_weighted_loss else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # === MLflow ===
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("CrowdSentiment_LS_Incremental")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        mlflow.set_tags({
            "project": "GRASS Sentiment",
            "model_type": "ResNet18",
            "ls_pipeline": True,
            "mode": "fine-tune",
        })

        mlflow.log_params({
            "epochs": num_epochs,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "device": device.type,
            "resumed": resumed,
            "num_segments": len(segments),
            "weighted_loss": use_weighted_loss
        })

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                mlflow.log_metric("batch_loss", loss.item(), step=epoch * len(dataloader) + batch_idx)
            avg_loss = running_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save model locally
        torch.save(model.state_dict(), model_output_path)
        logger.info(f"Saved LS model to: {model_output_path}")

        # Save metrics.json
        metrics = {
            "timestamp": now,
            "num_segments": len(segments),
            "avg_loss": avg_loss,
            "resumed": resumed,
            "weights": class_weights.tolist() if class_weights is not None else None
        }
        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Upload versioned model & metrics
        versioned_prefix = f"models/ls/versions/{now}/"
        try:
            upload_to_minio(model_output_path, minio_bucket, versioned_prefix + "ls_cnn_model.pth")
            upload_to_minio(metrics_output_path, minio_bucket, versioned_prefix + "metrics.json")
            logger.info(f"Uploaded versioned LS model and metrics to: {versioned_prefix}")
        except Exception as e:
            logger.warning(f"Failed to upload versioned LS model: {e}")

        # Upload to latest/ only if this is the first or best so far
        try:
            upload_to_minio(model_output_path, minio_bucket, "models/ls/latest/ls_cnn_model.pth")
            logger.info("Updated latest LS model in MinIO.")
        except Exception as e:
            logger.warning(f"Could not update latest LS model: {e}")

        # MLflow artifact
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="CrowdSentimentLS"
        )

    return model_output_path
