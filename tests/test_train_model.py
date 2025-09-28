# tests/test_train_model.py
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

import os, logging
os.environ["MLFLOW_SQLALCHEMY_STORE_POOL_CLASS"] = "sqlalchemy.pool.NullPool"
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)


# ---------- helpers ----------
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
    spec.loader.exec_module(mod)  
    return mod


def _make_png(path: Path, size=(224, 224), color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


# ---------- fixtures for two scenarios ----------
@pytest.fixture
def setup_skip_no_new(monkeypatch, tmp_path):
    """Environment where training should be skipped due to no new segments."""
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_train_module()
    monkeypatch.chdir(tmp_path)

    # Local fallback train.csv
    dset_dir = tmp_path / "dagster_output" / "dataset"
    dset_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "data"
    _make_png(img_dir / "approval_0.png")
    _make_png(img_dir / "neutral_0.png")
    pd.DataFrame(
        [
            {"path": str(img_dir / "approval_0.png"), "label": "approval"},
            {"path": str(img_dir / "neutral_0.png"), "label": "neutral"},
        ]
    ).to_csv(dset_dir / "train.csv", index=False)

    # MinIO client: force local CSV & provide global hash cache
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

    # All files hashed as "H" so all are already trained
    monkeypatch.setattr(m, "file_hash", lambda p: "H")

    return m, client


@pytest.fixture
def setup_happy(monkeypatch, tmp_path):
    """Happy tiny training setup."""
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
        [
            {"path": str(img_dir / "approval_0.png"), "label": "approval"},
            {"path": str(img_dir / "approval_1.png"), "label": "approval"},
            {"path": str(img_dir / "neutral_0.png"), "label": "neutral"},
            {"path": str(img_dir / "neutral_1.png"), "label": "neutral"},
        ]
    ).to_csv(dset_dir / "train.csv", index=False)

    # MinIO: no remote files (use local CSV; no resume)
    client = MagicMock()
    client.download_file.side_effect = Exception("no remote")
    monkeypatch.setattr(m, "get_minio_client", lambda: client)

    # Force CPU
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Replace torchvision resnet18 with tiny linear net
    import torch.nn as nn

    class TinyNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(224 * 224 * 3, num_classes)

        def forward(self, x):
            return self.fc(self.flatten(x))

    import torchvision.models as tvm
    monkeypatch.setattr(tvm, "resnet18", lambda pretrained=True: TinyNet(num_classes=2))

    # MLflow stubs
    import mlflow

    class DummyRun:
        class Info:
            experiment_id = "exp"
            run_id = "run"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

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

    monkeypatch.setattr(
        mlfpt, "log_model", lambda model, artifact_path, registered_model_name=None: None
    )

    return m, client




# 1) Skip path: status + reason
def test_train_model_skips_status_reason(setup_skip_no_new):
    m, client = setup_skip_no_new
    out = m.train_model()
    assert out["status"] == "skipped"
    assert "No new segments" in out["reason"]


# 2) Skip path: no uploads occurred when skipping
def test_train_model_skips_no_uploads(setup_skip_no_new):
    m, client = setup_skip_no_new
    _ = m.train_model()
    client.upload_file.assert_not_called()


# 3) Happy path: success & counts
def test_train_model_happy_status_and_counts(setup_happy):
    m, client = setup_happy
    out = m.train_model()
    assert out["status"] == "success"
    assert out["new_segments"] == 4
    assert out["total_segments"] == 4


# 4) Happy path: avg_loss type and model_path prefix
def test_train_model_happy_metrics_and_model_path(setup_happy):
    m, client = setup_happy
    out = m.train_model()
    assert isinstance(out["avg_loss"], float)
    assert out["model_path"].startswith("models/cnn/versions/")


# 5) Happy path: ensures uploads (model + metrics + cache… at least 2)
def test_train_model_happy_minio_uploads(setup_happy):
    m, client = setup_happy
    _ = m.train_model()
    assert client.upload_file.call_count >= 2


# 6) Happy path: metrics.json uploaded to the expected remote key 
def test_train_model_happy_metrics_uploaded_remote_key(setup_happy):
    m, client = setup_happy
    out = m.train_model()

    # Collect remote keys uploaded
    uploaded_remote_keys = [call.args[2] for call in client.upload_file.call_args_list if len(call.args) >= 3]

    # Expect a metrics upload into the versioned path
    expected_metrics_key = f"models/cnn/versions/{out['timestamp']}/metrics.json"
    assert expected_metrics_key in uploaded_remote_keys
