import os
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

    if not segments:
        raise ValueError("No segments provided for training.")

    model_dir = "/app/dagster_output/label_studio/model"
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "ls_cnn_model.pth")
    logger.info(f"Received {len(segments)} corrected LS segments for training.")

    # === DataLoader ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = IncrementalSpectrogramDataset(segments, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # === Class balancing check ===
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

    # === Model Setup ===
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Try Resuming from MinIO ===
    resumed = False
    minio_bucket = "grass-pipeline"
    minio_object = "models/ls/latest/ls_cnn_model.pth"
    try:
        tmp_model_path = os.path.join(tempfile.gettempdir(), "resume_ls_model.pth")
        client = get_minio_client()
        client.download_file(minio_bucket, minio_object, tmp_model_path)
        model.load_state_dict(torch.load(tmp_model_path, map_location=device))
        logger.info(f"Resumed model weights from MinIO: {minio_object}")
        resumed = True
    except Exception as e:
        logger.warning(f"Could not resume model from MinIO: {e}")
        resumed = False

    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if use_weighted_loss else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # === MLflow Setup ===
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = f"CrowdSentiment_LS_Incremental"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        mlflow.set_tags({
            "project": "GRASS Sentiment",
            "model_type": "ResNet18",
            "developer": "Ahmed",
            "ls_pipeline": True,
            "mode": "incremental"
        })

        mlflow.log_params({
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "device": device.type,
            "num_classes": num_classes,
            "weighted_loss": use_weighted_loss,
            "source": "LabelStudio",
            "num_segments": len(segments),
            "resumed": resumed
        })

        # === Train ===
        model.train()
        for epoch in range(3):
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

        # === Save + Upload ===
        torch.save(model.state_dict(), model_output_path)
        logger.info(f"Model saved locally: {model_output_path}")

        try:
            upload_to_minio(model_output_path, minio_bucket, minio_object)
            logger.info("Model uploaded to MinIO.")
        except Exception as e:
            logger.warning(f"MinIO upload failed: {e}")

        # Register to MLflow model registry
        mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="CrowdSentimentLS"
        )

    return model_output_path
