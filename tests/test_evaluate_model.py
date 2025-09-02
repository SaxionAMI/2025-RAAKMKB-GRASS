# tests/test_evaluate_model_unit.py
import sys
from types import ModuleType
from pathlib import Path
import importlib.util
from unittest.mock import MagicMock
import json
import pandas as pd
import pytest
from PIL import Image


# ---------- helpers ----------
def _stub_utils_in_sysmodules(class_labels=None):
    """Provide a stub for grass_pipeline_dagster.utils so imports don't pull real deps."""
    if "grass_pipeline_dagster" not in sys.modules:
        sys.modules["grass_pipeline_dagster"] = ModuleType("grass_pipeline_dagster")

    utils_stub = ModuleType("grass_pipeline_dagster.utils")
    utils_stub.load_settings = lambda: {
        "class_labels": class_labels or ["approval", "neutral"],
    }
    utils_stub.get_minio_client = lambda: MagicMock()
    utils_stub.get_influx_client = lambda: MagicMock()
    sys.modules["grass_pipeline_dagster.utils"] = utils_stub


def _load_eval_module():
    """Import the module containing evaluate_model & SpectrogramTestDataset directly from file."""
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "grass_pipeline_dagster" / "grass_pipeline_dagster" / "evaluate_assets.py"
    assert mod_path.exists(), f"evaluate_assets.py not found at {mod_path}"
    spec = importlib.util.spec_from_file_location("evaluate_assets_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _make_png(path: Path, size=(224, 224), color=(120, 120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


# ---------- Test 1: dataset behavior ----------
def test_spectrogram_test_dataset_reads_and_maps_labels(monkeypatch, tmp_path):
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_eval_module()

    # Make two tiny images, one per class
    img_dir = tmp_path / "imgs"
    a_path = img_dir / "a.png"
    n_path = img_dir / "n.png"
    _make_png(a_path)
    _make_png(n_path)

    df = pd.DataFrame(
        [
            {"path": str(a_path), "label": "approval"},
            {"path": str(n_path), "label": "neutral"},
        ]
    )
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    # No transform so we don't pull torch/torchvision in this test
    ds = m.SpectrogramTestDataset(str(csv_path), transform=None)

    assert len(ds) == 2
    # label_map should be {"approval": 0, "neutral": 1}
    img0, label0 = ds[0]
    img1, label1 = ds[1]
    assert label0 in (0, 1) and label1 in (0, 1)
    # order comes from load_settings enumeration
    assert ds.label_map == {"approval": 0, "neutral": 1}


# ---------- common fixture for the next two tests ----------
@pytest.fixture
def base_eval_setup(monkeypatch, tmp_path):
    """Minimal unit test setup: stub utils, module, torch model, MLflow, Influx, and MinIO."""
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_eval_module()
    monkeypatch.chdir(tmp_path)

    # Tiny test dataset with 4 samples
    data_dir = tmp_path / "data"
    a0 = data_dir / "approval_0.png"
    a1 = data_dir / "approval_1.png"
    n0 = data_dir / "neutral_0.png"
    n1 = data_dir / "neutral_1.png"
    for p in (a0, a1, n0, n1):
        _make_png(p)

    test_df = pd.DataFrame(
        [
            {"path": str(a0), "label": "approval"},
            {"path": str(a1), "label": "approval"},
            {"path": str(n0), "label": "neutral"},
            {"path": str(n1), "label": "neutral"},
        ]
    )

    # Torch / torchvision: force CPU and swap resnet18 for a tiny linear net
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    class TinyNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(224 * 224 * 3, num_classes)
        def forward(self, x):
            return self.fc(self.flatten(x))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tvm, "resnet18", lambda pretrained=False: TinyNet(num_classes=2))

    # MinIO stubs
    client = MagicMock()
    version_ts = "20990101_000000"
    model_key = f"models/cnn/versions/{version_ts}/cnn_model.pth"
    metrics_key = f"models/cnn/versions/{version_ts}/metrics.json"

    def list_objects_v2(**kwargs):
        return {"Contents": [{"Key": model_key}]}

    client.list_objects_v2.side_effect = list_objects_v2

    uploads = []
    client.upload_file.side_effect = lambda local, bucket, remote: uploads.append((local, remote))

    # Influx stub
    influx = MagicMock()
    write_api = MagicMock()
    influx.write_api.return_value = write_api

    # MLflow stubs (no side effects)
    import mlflow
    class DummyRun:
        class Info:
            experiment_id = "exp"
            run_id = "run"
        info = Info()
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "start_run", lambda: DummyRun())
    monkeypatch.setattr(mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v=None: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_params", lambda d: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v, step=None: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda d: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda p, artifact_path=None: None)

    # Matplotlib: avoid any GUI backend issues
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)

    # Patch utils accessors
    monkeypatch.setattr(m, "get_minio_client", lambda: client)
    monkeypatch.setattr(m, "get_influx_client", lambda: influx)

    return {
        "m": m,
        "client": client,
        "influx_write_api": write_api,
        "version_ts": version_ts,
        "model_key": model_key,
        "metrics_key": metrics_key,
        "uploads": uploads,
        "test_df": test_df,
        "TinyNet_class": TinyNet,
        "tmp_path": tmp_path,
    }


# ---------- Test 2: promotion when F1 improves ----------
def test_promotes_when_f1_improves(monkeypatch, base_eval_setup):
    m = base_eval_setup["m"]
    client = base_eval_setup["client"]
    uploads = base_eval_setup["uploads"]
    version_ts = base_eval_setup["version_ts"]
    model_key = base_eval_setup["model_key"]
    metrics_key = base_eval_setup["metrics_key"]
    test_df = base_eval_setup["test_df"]
    TinyNet = base_eval_setup["TinyNet_class"]

    # MinIO download behavior:
    # - return the test.csv
    # - write a valid TinyNet state dict for the model
    # - metrics.json with avg_loss
    # - latest/metrics absent -> will promote on any f1 >= 0
    import torch
    def download_file(bucket, key, dest):
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False))
            return
        if key == model_key:
            net = TinyNet(num_classes=2)
            torch.save(net.state_dict(), dest_path)
            return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.42}, indent=2))
            return
        if key == "models/cnn/latest/metrics.json":
            raise Exception("no latest metrics")
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = download_file

    out = m.evaluate_model()
    assert out["timestamp"] == version_ts
    # With no latest metrics, any F1 should promote
    remote_keys = {rk for _, rk in uploads}
    assert metrics_key in remote_keys
    assert "models/cnn/latest/cnn_model.pth" in remote_keys
    assert "models/cnn/latest/metrics.json" in remote_keys


# ---------- Test 3: no promotion when F1 <= latest ----------
def test_no_promotion_when_f1_not_improved(monkeypatch, base_eval_setup):
    m = base_eval_setup["m"]
    client = base_eval_setup["client"]
    uploads = base_eval_setup["uploads"]
    version_ts = base_eval_setup["version_ts"]
    model_key = base_eval_setup["model_key"]
    metrics_key = base_eval_setup["metrics_key"]
    test_df = base_eval_setup["test_df"]
    TinyNet = base_eval_setup["TinyNet_class"]

    # Make the model predict a constant class to yield a weak F1,
    # and make latest f1 = 1.0 so promotion should NOT happen.
    import torch
    import torch.nn as nn

    class ConstantTinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(224 * 224 * 3, 2)
            with torch.no_grad():
                self.fc.weight.zero_()
                self.fc.bias[:] = torch.tensor([1.0, 0.0])  # class-0 bias -> predict class 0 mostly
        def forward(self, x):
            return self.fc(self.flatten(x))

    # Override torchvision.resnet18 again for this test
    import torchvision.models as tvm
    monkeypatch.setattr(tvm, "resnet18", lambda pretrained=False: ConstantTinyNet())

    def download_file(bucket, key, dest):
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False))
            return
        if key == model_key:
            net = ConstantTinyNet()
            torch.save(net.state_dict(), dest_path)
            return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.9}, indent=2))
            return
        if key == "models/cnn/latest/metrics.json":
            dest_path.write_text(json.dumps({"f1": 1.0}, indent=2))  # strong latest -> no promotion
            return
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = download_file

    out = m.evaluate_model()
    assert out["timestamp"] == version_ts
    # Uploaded versioned metrics should still be updated
    remote_keys = {rk for _, rk in uploads}
    assert metrics_key in remote_keys
    # But no promotion to latest
    assert "models/cnn/latest/cnn_model.pth" not in remote_keys
    assert "models/cnn/latest/metrics.json" not in remote_keys
