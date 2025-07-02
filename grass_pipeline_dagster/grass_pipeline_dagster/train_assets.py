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

# Utility: hash file content for segment identity
def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

class SpectrogramDataset(Dataset):
    def __init__(self, df, transform=None):
        settings = load_settings()
        self.data = df.reset_index(drop=True)
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

    # ===== Load Settings and CSV =====
    force_retrain = bool(settings.get("force_retrain", False))
    logger.info(f"Force retrain from settings.json: {force_retrain}")

    minio_bucket = "grass-pipeline"
    minio_train_object = "datasets/train.csv"
    client = get_minio_client()

    # === Try loading train.csv from MinIO first ===
    try:
        train_csv = os.path.join(tempfile.gettempdir(), "train_model.csv")
        client.download_file(minio_bucket, minio_train_object, train_csv)
        logger.info(f"Loaded train.csv from MinIO: {minio_train_object}")
    except Exception as e:
        logger.warning(f"Could not load train.csv from MinIO: {e}")
        local_fallback = os.path.join(os.getcwd(), "dagster_output", "dataset", "train.csv")
        if os.path.exists(local_fallback):
            train_csv = local_fallback
            logger.info(f"Using fallback local train.csv: {train_csv}")
        else:
            logger.error("No train.csv found in MinIO or locally. Training aborted.")
            raise FileNotFoundError("Train CSV not found in MinIO or local path.")

    model_output_path = os.path.join(os.getcwd(), "dagster_output", "model", "cnn_model.pth")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    logger.info(f"Loading training data from {train_csv}")

    # ===== Hash All Segments for Incremental Training =====
    df = pd.read_csv(train_csv)
    logger.info("Hashing all train segments for incremental training...")
    df["segment_hash"] = df["path"].apply(file_hash)

    minio_segment_cache_path = "models/cnn/latest/global_segment_hashes.txt"
    local_cache_path = os.path.join(tempfile.gettempdir(), "global_segment_hashes.txt")

    # ===== Load Hash Cache =====
    try:
        client.download_file(minio_bucket, minio_segment_cache_path, local_cache_path)
        with open(local_cache_path, "r") as f:
            trained_hashes = set(line.strip() for line in f)
        logger.info(f"Loaded {len(trained_hashes)} trained segment hashes from cache.")
    except Exception:
        trained_hashes = set()
        logger.info("No segment hash cache found in MinIO; all segments considered new.")

    # ===== Conditional Selection of Segments to Train On =====
    if force_retrain:
        new_df = df
        trained_hashes = set()  # All segments will be considered new
        logger.info(f" Force retrain active: using ALL {len(new_df)} segments from train.csv.")
    else:
        new_df = df[~df["segment_hash"].isin(trained_hashes)].reset_index(drop=True)
        logger.info(f"Segments in train.csv: {len(df)}, New (never-trained) segments in train.csv: {len(new_df)}")
        if new_df.empty:
            logger.info("No new segments to train on. Skipping training.")
            return model_output_path

    # ===== DataLoader Construction =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SpectrogramDataset(new_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ===== Handle Class Imbalance =====
    label_counts = new_df['label'].str.lower().value_counts()
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
    minio_object = "models/cnn/latest/cnn_model.pth"
    resume_model_found = False
    try:
        temp_model_path = os.path.join(tempfile.gettempdir(), "cnn_model_resume.pth")
        client.download_file(minio_bucket, minio_object, temp_model_path)
        model.load_state_dict(torch.load(temp_model_path, map_location=device))
        logger.info(f"Resumed training from model in MinIO: {minio_object}")
        resume_model_found = True
    except Exception as e:
        logger.warning(f"Could not resume from MinIO: {e}")
        if os.path.exists(model_output_path):
            model.load_state_dict(torch.load(model_output_path, map_location=device))
            logger.info(f"Resumed training from local model: {model_output_path}")
            resume_model_found = True
        else:
            logger.info("No existing model found. Training from scratch.")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if use_weighted_loss else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ===== MLflow Setup =====
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = f"CrowdSentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

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
            "resumed": resume_model_found,
            "new_segments": len(new_df),
            "total_segments": len(df)
        })

        mlflow.set_tag("num_samples", len(new_df))
        mlflow.log_artifact(train_csv, artifact_path="data")

        logger.info("Training model...")
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
            logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save locally
        torch.save(model.state_dict(), model_output_path)
        logger.info(f"Model saved locally: {model_output_path}")

        # Upload to MinIO
        try:
            upload_to_minio(model_output_path, minio_bucket, minio_object)
            logger.info(f"Model uploaded to MinIO: {minio_object}")
        except Exception as e:
            logger.warning(f"Failed to upload model to MinIO: {e}")

        # Register to MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="CrowdSentimentResNet"
        )

    # ===== Update Segment Hash Cache =====
    if force_retrain:
        updated_hashes = set(new_df["segment_hash"].tolist())
    else:
        updated_hashes = trained_hashes.union(set(new_df["segment_hash"].tolist()))
    with open(local_cache_path, "w") as f:
        for seg_hash in sorted(updated_hashes):
            f.write(f"{seg_hash}\n")
    client.upload_file(local_cache_path, minio_bucket, minio_segment_cache_path)
    logger.info(f"Updated segment hash cache in MinIO: {minio_segment_cache_path}")

    return model_output_path
