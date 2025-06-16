# prepare_assets.py

from dagster import asset, get_dagster_logger
import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from grass_pipeline_dagster.utils import load_settings, get_minio_client


def upload_to_minio(local_path, bucket, remote_path):
    client = get_minio_client()
    try:
        client.upload_file(local_path, bucket, remote_path)
    except Exception as e:
        raise RuntimeError(f"Failed to upload {local_path} to MinIO ({bucket}/{remote_path}): {e}")


def extract_clip_id(image_path):
    # Assumes filenames like "approval/cheer1_segment_03.png" → clip_id = "cheer1"
    filename = os.path.basename(image_path)
    parts = filename.split("_segment")
    return parts[0] if len(parts) > 1 else filename.split(".")[0]


@asset
def prepare_dataset() -> dict:
    """
    Prepares dataset with no audio leakage and ensures class balance:
    segments are grouped by audio clip and split using stratified sampling.
    """
    logger = get_dagster_logger()
    settings = load_settings()
    allowed_labels = [label.lower() for label in settings["class_labels"]]

    base_dir = os.path.abspath(os.path.join(os.getcwd(), "dagster_output", "spectrogram_images"))
    dataset_output_dir = os.path.abspath(os.path.join(os.getcwd(), "dagster_output", "dataset"))
    os.makedirs(dataset_output_dir, exist_ok=True)

    logger.info(f"Scanning base directory: {base_dir}")
    records = []

    for label_name in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        normalized_label = label_name.lower()
        if normalized_label not in allowed_labels:
            logger.warning(f"Skipping unexpected label: {label_name}")
            continue
        for img_file in glob.glob(os.path.join(label_path, "*.png")):
            clip_id = extract_clip_id(img_file)
            records.append({
                "path": img_file,
                "label": normalized_label,
                "clip_id": clip_id
            })

    if not records:
        raise RuntimeError("No spectrogram images found. Check input folders and labels.")

    df = pd.DataFrame(records)
    logger.info(f"Total segments: {len(df)}, Unique clips: {df['clip_id'].nunique()}")

    # Step 1: Create a clip-level DataFrame for stratification
    clip_df = df.groupby("clip_id").first().reset_index()[["clip_id", "label"]]

    # Step 2: Stratified split based on clip-level labels
    train_clip_ids, test_clip_ids = train_test_split(
        clip_df["clip_id"],
        test_size=0.15,
        stratify=clip_df["label"],
        random_state=42
    )

    # Step 3: Filter segment-level records
    train_df = df[df["clip_id"].isin(train_clip_ids)].drop(columns=["clip_id"])
    test_df = df[df["clip_id"].isin(test_clip_ids)].drop(columns=["clip_id"])

    # Shuffle
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save CSVs
    train_path = os.path.join(dataset_output_dir, "train.csv")
    test_path = os.path.join(dataset_output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set: {len(train_df)} segments from {len(train_clip_ids)} clips → {train_path}")
    logger.info(f"Test set: {len(test_df)} segments from {len(test_clip_ids)} clips → {test_path}")

    # Upload to MinIO
    bucket = "grass-pipeline"
    upload_to_minio(train_path, bucket, "datasets/train.csv")
    upload_to_minio(test_path, bucket, "datasets/test.csv")
    logger.info(f"Uploaded train/test CSVs to MinIO bucket '{bucket}'")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }
