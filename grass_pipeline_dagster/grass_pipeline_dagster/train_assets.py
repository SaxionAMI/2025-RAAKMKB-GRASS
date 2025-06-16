import os
import time
import tempfile
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from dagster import asset, get_dagster_logger

from grass_pipeline_dagster.utils import load_settings, get_minio_client


class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        settings = load_settings()
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.class_labels = [label.lower() for label in settings["class_labels"]]
        self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}  # ✅ Fixed

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


def upload_to_minio(local_path, bucket, remote_path):
    client = get_minio_client()
    try:
        client.upload_file(local_path, bucket, remote_path)
    except Exception as e:
        raise RuntimeError(f"Failed to upload {local_path} to MinIO ({bucket}/{remote_path}): {e}")


@asset
def train_model() -> str:
    logger = get_dagster_logger()
    settings = load_settings()
    class_labels = settings["class_labels"]
    num_classes = len(class_labels)

    train_csv = os.path.join(os.getcwd(), "dagster_output", "dataset", "train.csv")
    model_output_path = os.path.join(os.getcwd(), "dagster_output", "model", "cnn_model.pth")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    logger.info(f"Loading training data from {train_csv}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SpectrogramDataset(train_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Handle class imbalance
    label_counts = pd.read_csv(train_csv)['label'].str.lower().value_counts()
    min_count = label_counts.min()
    max_count = label_counts.max()
    imbalance_ratio = max_count / min_count
    use_weighted_loss = imbalance_ratio > 1.5

    if use_weighted_loss:
        total = label_counts.sum()
        weights = [total / (len(class_labels) * label_counts.get(label.lower(), 1)) for label in class_labels]
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
    else:
        class_weights = None
        logger.info("Using unweighted loss (balanced classes).")

    logger.info("Initializing ResNet18 model")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Resume Logic ===
    client = get_minio_client()
    minio_bucket = "grass-pipeline"
    minio_object = "models/cnn/latest/cnn_model.pth"
    resume_model_found = False

    try:
        temp_model_path = os.path.join(tempfile.gettempdir(), "cnn_model_resume.pth")
        client.download_file(minio_bucket, minio_object, temp_model_path)
        model.load_state_dict(torch.load(temp_model_path, map_location=device))
        logger.info(f"✅ Resumed training from model in MinIO: {minio_object}")
        resume_model_found = True
    except Exception as e:
        logger.warning(f"⚠️ Could not resume from MinIO: {e}")
        if os.path.exists(model_output_path):
            model.load_state_dict(torch.load(model_output_path, map_location=device))
            logger.info(f"✅ Resumed training from local model: {model_output_path}")
            resume_model_found = True
        else:
            logger.info("🆕 No existing model found. Training from scratch.")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if use_weighted_loss else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ===== MLflow Setup =====
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = f"CrowdSentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"🎯 Started MLflow run: {run_id}")

        mlflow.set_tags({
            "project": "GRASS Sentiment",
            "model_type": "ResNet18",
            "developer": "Ahmed",
        })

        mlflow.log_params({
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "device": device.type,
            "num_classes": num_classes,
            "weighted_loss": use_weighted_loss,
            "resumed": resume_model_found
        })

        mlflow.set_tag("num_samples", len(dataset))
        mlflow.log_artifact(train_csv, artifact_path="data")

        logger.info("🚀 Training model...")
        model.train()
        for epoch in range(5):
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
            logger.info(f"📊 Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save locally
        torch.save(model.state_dict(), model_output_path)
        logger.info(f"💾 Model saved locally: {model_output_path}")

        # Upload to MinIO
        try:
            upload_to_minio(model_output_path, minio_bucket, minio_object)
            logger.info(f"☁️ Model uploaded to MinIO: {minio_object}")
        except Exception as e:
            logger.warning(f"❌ Failed to upload model to MinIO: {e}")

        # Register to MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="CrowdSentimentResNet"
        )

    return model_output_path
