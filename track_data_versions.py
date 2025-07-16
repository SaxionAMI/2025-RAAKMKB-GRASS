import os
import json
import subprocess
from pathlib import Path

def load_settings():
    config_path = Path(__file__).resolve().parent / "config" / "settings.json"
    with open(config_path, "r") as f:
        return json.load(f)

def run_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd_list)}")
        print(f"Error: {e}")
        exit(1)

def dvc_track_and_push(path, commit_msg):
    path = os.path.abspath(path)
    print(f"📦 Tracking: {path}")
    run_cmd(["dvc", "add", path])
    run_cmd(["git", "add", f"{path}.dvc", ".gitignore"])
    run_cmd(["git", "commit", "-m", commit_msg])
    run_cmd(["dvc", "push"])

def main():
    print("🔧 Loading config/settings.json...")
    settings = load_settings()

    # === Track spectrograms ===
    spec_dir = os.path.join(os.getcwd(), "dagster_output", "spectrogram_images")
    if os.path.exists(spec_dir):
        dvc_track_and_push(spec_dir, "Track spectrogram images")
    else:
        print(f"⚠️ Spectrogram directory not found: {spec_dir}")

    # === Track train/test CSVs ===
    dataset_dir = os.path.join(os.getcwd(), "dagster_output", "dataset")
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    if os.path.exists(train_csv):
        dvc_track_and_push(train_csv, "Track train.csv split")
    else:
        print(f"⚠️ train.csv not found: {train_csv}")

    if os.path.exists(test_csv):
        dvc_track_and_push(test_csv, "Track test.csv split")
    else:
        print(f"⚠️ test.csv not found: {test_csv}")

    print("✅ All DVC tracking complete.")

if __name__ == "__main__":
    main()
