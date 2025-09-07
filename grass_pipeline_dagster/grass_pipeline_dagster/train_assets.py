import os
import json
import tempfile
from datetime import datetime
import time
import torch
import pandas as pd
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from dagster import asset, get_dagster_logger

from grass_pipeline_dagster.utils import load_settings, get_minio_client


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


@asset
def train_model() -> dict:
    logger = get_dagster_logger()
    start_time = time.time()
    logger.info("Starting train_model asset...")

    settings = load_settings()
    class_labels = settings["class_labels"]
    num_classes = len(class_labels)
    resume_from_ls_model = bool(settings.get("resume_from_ls_model", False))
    force_retrain = bool(settings.get("force_retrain", False))

    logger.info(f"Settings loaded: {num_classes} classes")
    logger.info(f"Flags — Force retrain: {force_retrain}, Resume from LS model: {resume_from_ls_model}")

    minio_bucket = "grass-pipeline"
    minio_train_object = "datasets/train.csv"
    client = get_minio_client()

    # === Load train.csv ===
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
            raise FileNotFoundError("train.csv not found in MinIO or locally.")

    df = pd.read_csv(train_csv)
    logger.info(f"Loaded train.csv with {len(df)} rows")
    logger.info("Hashing segments for incremental training...")
    df["segment_hash"] = df["path"].apply(file_hash)

    # === Load existing hash cache ===
    minio_segment_cache_path = "models/cnn/latest/global_segment_hashes.txt"
    local_cache_path = os.path.join(tempfile.gettempdir(), "global_segment_hashes.txt")
    try:
        client.download_file(minio_bucket, minio_segment_cache_path, local_cache_path)
        with open(local_cache_path, "r") as f:
            trained_hashes = set(line.strip() for line in f)
        logger.info(f"Loaded {len(trained_hashes)} existing segment hashes")
    except Exception:
        trained_hashes = set()
        logger.info("No existing hash cache found. All segments treated as new.")

    # === Determine segments to train on ===
    if force_retrain:
        new_df = df
        trained_hashes = set()
        logger.info("Force retrain enabled: using ALL segments.")
    else:
        new_df = df[~df["segment_hash"].isin(trained_hashes)].reset_index(drop=True)
        logger.info(f"➕ Found {len(new_df)} new segments to train on")

    if new_df.empty:
        logger.info("No new segments found. Skipping training.")
        return {
            "status": "skipped",
            "reason": "No new segments found.",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    # === DataLoader Setup ===
    class SpectrogramDataset(Dataset):
        def __init__(self, df, transform=None):
            self.data = df.reset_index(drop=True)
            self.transform = transform
            self.label_map = {label.lower(): idx for idx, label in enumerate(class_labels)}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path = self.data.iloc[idx]['path']
            label_str = self.data.iloc[idx]['label'].lower()
            label = self.label_map[label_str]
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                logger.warning(f"Skipping corrupted image: {img_path}, error: {e}")
                return self.__getitem__((idx + 1) % len(self.data))

    logger.info("Preparing DataLoader...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = SpectrogramDataset(new_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,)

    # === Handle class imbalance ===
    label_counts = new_df['label'].str.lower().value_counts()
    imbalance_ratio = label_counts.max() / label_counts.min()
    use_weighted_loss = imbalance_ratio > 1.5
    if use_weighted_loss:
        total = label_counts.sum()
        weights = [total / (len(class_labels) * label_counts.get(label.lower(), 1)) for label in class_labels]
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
    else:
        class_weights = None
        logger.info("Balanced dataset. Using unweighted loss.")

    # === Initialize model ===
    logger.info("Initializing ResNet18 model...")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Resume model if available ===
    resume_source = "scratch"
    if not force_retrain:
        if resume_from_ls_model:
            try:
                path = "models/ls/latest/ls_cnn_model.pth"
                temp = os.path.join(tempfile.gettempdir(), "ls_resume.pth")
                client.download_file(minio_bucket, path, temp)
                model.load_state_dict(torch.load(temp, map_location=device))
                resume_source = "ls_latest"
                logger.info("Resumed model from Label Studio checkpoint.")
            except Exception as e:
                logger.warning(f"Failed to resume from Label Studio: {e}")
        if resume_source == "scratch":
            try:
                path = "models/cnn/latest/cnn_model.pth"
                temp = os.path.join(tempfile.gettempdir(), "cnn_resume.pth")
                client.download_file(minio_bucket, path, temp)
                model.load_state_dict(torch.load(temp, map_location=device))
                resume_source = "cnn_latest"
                logger.info("Resumed model from latest CNN checkpoint.")
            except Exception as e:
                logger.warning(f"Failed to resume from CNN checkpoint: {e}")
                logger.info("Training will start from scratch.")
    else:
        logger.info("Force retrain enabled — model will start from scratch (no resume).")

    # === MLflow tracking ===
    mlflow.set_tracking_uri("http://mlflow:5000")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(f"CrowdSentiment_{timestamp}")
    local_model_path = os.path.join("dagster_output", "model", "versions", timestamp, "cnn_model.pth")
    minio_model_path = f"models/cnn/versions/{timestamp}/cnn_model.pth"
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    logger.info("Starting MLflow run...")
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "project": "GRASS Sentiment",
            "model_type": "ResNet18",
            "resume_source": resume_source
        })
        mlflow.log_param("resumed", resume_source != "scratch")
        mlflow.log_params({
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "device": device.type,
            "num_classes": num_classes,
            "weighted_loss": use_weighted_loss,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "new_segments": len(new_df),
            "total_segments": len(df)
        })
        mlflow.log_dict(label_counts.to_dict(), "label_distribution.json")
        logger.info(f"MLflow experiment_id: {run.info.experiment_id}, run_id: {run.info.run_id}")

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if use_weighted_loss else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        total_loss = 0.0
        for epoch in range(5):
            epoch_loss = 0.0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                mlflow.log_metric("batch_loss", loss.item(), step=epoch * len(dataloader) + i)
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            logger.info(f"Epoch {epoch+1}/5 completed. Avg Loss: {avg_loss:.4f}")
            total_loss += avg_loss
        avg_loss = total_loss / 5

        torch.save(model.state_dict(), local_model_path)
        logger.info(f"Saved trained model locally: {local_model_path}")
        client.upload_file(local_model_path, minio_bucket, minio_model_path)
        logger.info(f"Uploaded model to MinIO: {minio_model_path}")

        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name="CrowdSentimentResNet")
        logger.info("Model registered in MLflow.")

        metrics = {
            "avg_loss": avg_loss,
            "timestamp": timestamp,
            "resume_source": resume_source,
            "new_segments": len(new_df),
            "total_segments": len(df)
        }
        metrics_path = os.path.join(os.path.dirname(local_model_path), "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        client.upload_file(metrics_path, minio_bucket, f"models/cnn/versions/{timestamp}/metrics.json")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        mlflow.log_artifact(train_csv, artifact_path="data")

    updated_hashes = trained_hashes.union(set(new_df["segment_hash"].tolist()))
    with open(local_cache_path, "w") as f:
        for seg_hash in sorted(updated_hashes):
            f.write(f"{seg_hash}\n")
    client.upload_file(local_cache_path, minio_bucket, minio_segment_cache_path)
    logger.info(f"Updated segment hash cache in MinIO: {minio_segment_cache_path}")

    total_time = round(time.time() - start_time, 2)
    logger.info(f"train_model completed successfully in {total_time} seconds.")

    return {
        "status": "success",
        "model_path": minio_model_path,
        "metrics_path": f"models/cnn/versions/{timestamp}/metrics.json",
        "avg_loss": avg_loss,
        "new_segments": len(new_df),
        "total_segments": len(df),
        "resume_source": resume_source,
        "timestamp": timestamp,
        "duration_sec": total_time
    }
