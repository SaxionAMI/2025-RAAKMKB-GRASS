import os
from datetime import datetime
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

# === Custom Dataset for LS data ===
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
def train_on_ls_data() -> str:
    logger = get_dagster_logger()
    settings = load_settings()
    class_labels = settings["class_labels"]
    num_classes = len(class_labels)

    # ===== Load LS Train CSV =====
    ls_train_csv = "/app/dagster_output/label_studio/ls_dataset/ls_train.csv"
    if not os.path.exists(ls_train_csv):
        logger.error(f"❌ LS train.csv not found at {ls_train_csv}. Aborting.")
        raise FileNotFoundError("LS train.csv not found!")

    model_dir = "/app/dagster_output/label_studio/model"
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "ls_cnn_model.pth")
    logger.info(f"Loading LS training data from {ls_train_csv}")

    # ===== Load Data =====
    df = pd.read_csv(ls_train_csv)
    logger.info(f"Loaded {len(df)} LS-segmented spectrograms for training.")

    # ===== DataLoader =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SpectrogramDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ===== Class Imbalance Handling =====
    label_counts = df['label'].str.lower().value_counts()
    min_count = label_counts.min()
    max_count = label_counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else 1
    use_weighted_loss = imbalance_ratio > 1.5
    if use_weighted_loss:
        total = label_counts.sum()
        weights = [total / (len(class_labels) * label_counts.get(label.lower(), 1)) for label in class_labels]
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
    else:
        class_weights = None
        logger.info("Using unweighted loss (balanced classes).")

    # ===== Model Setup =====
    logger.info("Initializing ResNet18 model for LS pipeline.")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if use_weighted_loss else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ===== MLflow Setup =====
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = f"CrowdSentiment_LS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"🎯 Started MLflow run: {run_id}")

        mlflow.set_tags({
            "project": "GRASS Sentiment",
            "model_type": "ResNet18",
            "developer": "Ahmed",
            "ls_pipeline": True
        })

        mlflow.log_params({
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "device": device.type,
            "num_classes": num_classes,
            "weighted_loss": use_weighted_loss,
            "source": "LabelStudio",
            "num_segments": len(df)
        })

        mlflow.log_artifact(ls_train_csv, artifact_path="data")

        logger.info("🚀 Training LS model...")
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
            logger.info(f"📊 LS Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save locally
        torch.save(model.state_dict(), model_output_path)
        logger.info(f"💾 LS Model saved locally: {model_output_path}")

        # Upload to MinIO (separate path for LS models)
        minio_bucket = "grass-pipeline"
        minio_object = "models/ls/latest/ls_cnn_model.pth"
        try:
            upload_to_minio(model_output_path, minio_bucket, minio_object)
            logger.info(f"☁️ LS Model uploaded to MinIO: {minio_object}")
        except Exception as e:
            logger.warning(f"❌ Failed to upload LS model to MinIO: {e}")

        # Register to MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="CrowdSentimentResNet_LS"
        )

    return model_output_path
