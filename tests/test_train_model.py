import os
import sys
from types import ModuleType
from pathlib import Path
import importlib.util
from unittest.mock import MagicMock
import pandas as pd
import pytest
from PIL import Image
import json


def _stub_utils_in_sysmodules(class_labels=None, resume=False, force=False):
    if "grass_pipeline_dagster" not in sys.modules:
        sys.modules["grass_pipeline_dagster"] = ModuleType("grass_pipeline_dagster")
    utils_stub = ModuleType("grass_pipeline_dagster.utils")
    utils_stub.load_settings = lambda: {
        "class_labels": class_labels or ["approval", "neutral"],
        "resume_from_ls_model": resume,
        "force_retrain": force,
    }
    utils_stub.get_minio_client = lambda: MagicMock()
    sys.modules["grass_pipeline_dagster.utils"] = utils_stub


def _load_train_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "grass_pipeline_dagster" / "grass_pipeline_dagster" / "train_assets.py"
    assert mod_path.exists(), f"train_model.py not found at {mod_path}"
    spec = importlib.util.spec_from_file_location("train_model_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _make_png(path: Path, size=(224, 224), color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_train_model_skips_when_no_new_segments(monkeypatch, tmp_path):
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_train_module()
    monkeypatch.chdir(tmp_path)

    # Prepare local fallback train.csv
    dset_dir = tmp_path / "dagster_output" / "dataset"
    dset_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "data"
    _make_png(img_dir / "approval_0.png")
    _make_png(img_dir / "neutral_0.png")
    pd.DataFrame(
        [{"path": str(img_dir / "approval_0.png"), "label": "approval"},
         {"path": str(img_dir / "neutral_0.png"), "label": "neutral"}]
    ).to_csv(dset_dir / "train.csv", index=False)

    # Stub MinIO client: force local fallback for train.csv,
    # but provide a hash cache that marks all as already trained
    client = MagicMock()
    def _download_file(bucket, key, dest):
        if key.endswith("datasets/train.csv"):
            raise Exception("no remote")
        if key.endswith("global_segment_hashes.txt"):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w") as f:
                f.write("H\n")
    client.download_file.side_effect = _download_file
    monkeypatch.setattr(m, "get_minio_client", lambda: client)

    # Make all file hashes "H" to match the cache
    monkeypatch.setattr(m, "file_hash", lambda p: "H")

    out = m.train_model()
    assert out["status"] == "skipped"
    assert "No new segments" in out["reason"]


def test_train_model_happy_path_tiny(monkeypatch, tmp_path):
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_train_module()
    monkeypatch.chdir(tmp_path)

    # Tiny dataset
    dset_dir = tmp_path / "dagster_output" / "dataset"
    dset_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "data"
    _make_png(img_dir / "approval_0.png")
    _make_png(img_dir / "approval_1.png")
    _make_png(img_dir / "neutral_0.png")
    _make_png(img_dir / "neutral_1.png")
    pd.DataFrame(
        [{"path": str(img_dir / "approval_0.png"), "label": "approval"},
         {"path": str(img_dir / "approval_1.png"), "label": "approval"},
         {"path": str(img_dir / "neutral_0.png"), "label": "neutral"},
         {"path": str(img_dir / "neutral_1.png"), "label": "neutral"}]
    ).to_csv(dset_dir / "train.csv", index=False)

    # MinIO client: force no remote files (so it uses local CSV; no resumes)
    client = MagicMock()
    client.download_file.side_effect = Exception("no remote")
    monkeypatch.setattr(m, "get_minio_client", lambda: client)

    # Force CPU
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Replace torchvision resnet18 with a tiny linear model
    import torch.nn as nn
    class TinyNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(224 * 224 * 3, num_classes)
        def forward(self, x): return self.fc(self.flatten(x))

    import torchvision.models as tvm
    monkeypatch.setattr(tvm, "resnet18", lambda pretrained=True: TinyNet(num_classes=2))

    # Stub all MLflow calls
    import mlflow
    class DummyRun:
        class Info:
            experiment_id = "exp"; run_id = "run"
        info = Info()
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "start_run", lambda: DummyRun())
    monkeypatch.setattr(mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_params", lambda d: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda d, p: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v, step=None: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda p, artifact_path=None: None)
    import mlflow.pytorch as mlfpt
    monkeypatch.setattr(mlfpt, "log_model", lambda model, artifact_path, registered_model_name=None: None)

    out = m.train_model()
    assert out["status"] == "success"
    assert out["new_segments"] == 4
    assert out["total_segments"] == 4
    assert isinstance(out["avg_loss"], float)
    assert out["model_path"].startswith("models/cnn/versions/")

    # Two or more uploads (model + metrics + cache)
    assert client.upload_file.call_count >= 2

    # local metrics.json exists and is valid JSON
    metrics_path = Path("dagster_output") / "model" / "versions" / out["timestamp"] / "metrics.json"
    assert metrics_path.exists()
    json.loads(metrics_path.read_text())
