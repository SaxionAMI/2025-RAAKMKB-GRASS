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


# ---------- Test 1A: dataset length & label map ----------
def test_spectrogram_test_dataset_len_and_labelmap(monkeypatch, tmp_path):
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

    ds = m.SpectrogramTestDataset(str(csv_path), transform=None)

    assert len(ds) == 2
    assert ds.label_map == {"approval": 0, "neutral": 1}


# ---------- Test 1B: dataset getitem returns correct label indices ----------
def test_spectrogram_test_dataset_getitem(monkeypatch, tmp_path):
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_eval_module()

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

    ds = m.SpectrogramTestDataset(str(csv_path), transform=None)

    img0, label0 = ds[0]
    img1, label1 = ds[1]
    assert label0 in (0, 1) and label1 in (0, 1)
    assert {label0, label1} == {0, 1}


# ---------- common fixture for the model evaluation tests ----------
@pytest.fixture
def base_eval_setup(monkeypatch, tmp_path):
    """Minimal unit test setup: stub utils, module, torch model, MLflow, Influx, and MinIO."""
    _stub_utils_in_sysmodules(["approval", "neutral"])
    m = _load_eval_module()
    monkeypatch.chdir(tmp_path)

    # Keep all temp artifacts inside pytest's sandbox
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

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

    # swap resnet18 for a tiny linear net
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

    # MLflow stubs
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


# ---------- Test 2: promotion when F1 improves (no latest metrics) ----------
def test_promotes_when_f1_improves(monkeypatch, base_eval_setup):
    m = base_eval_setup["m"]
    client = base_eval_setup["client"]
    uploads = base_eval_setup["uploads"]
    version_ts = base_eval_setup["version_ts"]
    model_key = base_eval_setup["model_key"]
    metrics_key = base_eval_setup["metrics_key"]
    test_df = base_eval_setup["test_df"]
    TinyNet = base_eval_setup["TinyNet_class"]

    import torch
    def download_file(bucket, key, dest):
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False)); return
        if key == model_key:
            net = TinyNet(num_classes=2); torch.save(net.state_dict(), dest_path); return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.42}, indent=2)); return
        if key == "models/cnn/latest/metrics.json":
            raise Exception("no latest metrics")
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = download_file

    out = m.evaluate_model()
    assert out["timestamp"] == version_ts
    remote_keys = {rk for _, rk in uploads}
    assert metrics_key in remote_keys
    assert "models/cnn/latest/cnn_model.pth" in remote_keys
    assert "models/cnn/latest/metrics.json" in remote_keys


# ---------- Test 3: no promotion when Macro-F1 not improved (latest has f1_macro) ----------
def test_no_promotion_when_f1_not_improved(monkeypatch, base_eval_setup):
    m = base_eval_setup["m"]
    client = base_eval_setup["client"]
    uploads = base_eval_setup["uploads"]
    version_ts = base_eval_setup["version_ts"]
    model_key = base_eval_setup["model_key"]
    metrics_key = base_eval_setup["metrics_key"]
    test_df = base_eval_setup["test_df"]

    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    class ConstantTinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(224 * 224 * 3, 2)
            with torch.no_grad():
                self.fc.weight.zero_()
                self.fc.bias[:] = torch.tensor([1.0, 0.0])  # predict class-0 mostly
        def forward(self, x):
            return self.fc(self.flatten(x))

    monkeypatch.setattr(tvm, "resnet18", lambda pretrained=False: ConstantTinyNet())

    def download_file(bucket, key, dest):
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False)); return
        if key == model_key:
            net = ConstantTinyNet(); torch.save(net.state_dict(), dest_path); return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.9}, indent=2)); return
        if key == "models/cnn/latest/metrics.json":
            # IMPORTANT: align with new rule (Macro-F1)
            dest_path.write_text(json.dumps({"f1_macro": 1.0}, indent=2)); return
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = download_file

    out = m.evaluate_model()
    assert out["timestamp"] == version_ts
    remote_keys = {rk for _, rk in uploads}
    assert metrics_key in remote_keys
    assert "models/cnn/latest/cnn_model.pth" not in remote_keys
    assert "models/cnn/latest/metrics.json" not in remote_keys


# ---------- Test 4: no promotion when new == latest (strict '>') ----------
def test_no_promotion_when_equal_macro_f1(monkeypatch, base_eval_setup):
    """
    Ensure promotion requires strictly better Macro-F1 (new > latest).
    We seed latest with the exact Macro-F1 we expect from the run.
    """
    m = base_eval_setup["m"]
    client = base_eval_setup["client"]
    uploads = base_eval_setup["uploads"]
    version_ts = base_eval_setup["version_ts"]
    model_key = base_eval_setup["model_key"]
    metrics_key = base_eval_setup["metrics_key"]
    test_df = base_eval_setup["test_df"]
    TinyNet = base_eval_setup["TinyNet_class"]

    import torch

    # First pass: behave like "no latest" to compute the macro-F1 of this setup
    def dl_first(bucket, key, dest):
        dest_path = Path(dest); dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False)); return
        if key == model_key:
            net = TinyNet(num_classes=2); torch.save(net.state_dict(), dest_path); return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.42}, indent=2)); return
        if key == "models/cnn/latest/metrics.json":
            raise Exception("no latest metrics")
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = dl_first
    out1 = m.evaluate_model()
    new_macro = float(out1["f1_macro"])

    # Second pass: set latest f1_macro = new_macro -> must NOT promote
    uploads.clear()
    def dl_second(bucket, key, dest):
        dest_path = Path(dest); dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False)); return
        if key == model_key:
            net = TinyNet(num_classes=2); torch.save(net.state_dict(), dest_path); return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.42}, indent=2)); return
        if key == "models/cnn/latest/metrics.json":
            dest_path.write_text(json.dumps({"f1_macro": new_macro}, indent=2)); return
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = dl_second
    out2 = m.evaluate_model()
    assert out2["timestamp"] == version_ts

    remote_keys = {rk for _, rk in uploads}
    assert metrics_key in remote_keys
    assert "models/cnn/latest/cnn_model.pth" not in remote_keys
    assert "models/cnn/latest/metrics.json" not in remote_keys


# ---------- Test 5: versioned metrics are always uploaded ----------
def test_versioned_metrics_always_uploaded(monkeypatch, base_eval_setup):
    """
    Regardless of promotion, the versioned metrics.json must be uploaded.
    """
    m = base_eval_setup["m"]
    client = base_eval_setup["client"]
    uploads = base_eval_setup["uploads"]
    version_ts = base_eval_setup["version_ts"]
    model_key = base_eval_setup["model_key"]
    metrics_key = base_eval_setup["metrics_key"]
    test_df = base_eval_setup["test_df"]
    TinyNet = base_eval_setup["TinyNet_class"]

    import torch
    def download_file(bucket, key, dest):
        dest_path = Path(dest); dest_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "datasets/test.csv":
            dest_path.write_text(test_df.to_csv(index=False)); return
        if key == model_key:
            net = TinyNet(num_classes=2); torch.save(net.state_dict(), dest_path); return
        if key == metrics_key:
            dest_path.write_text(json.dumps({"avg_loss": 0.42}, indent=2)); return
        if key == "models/cnn/latest/metrics.json":
            # seed a high macro so there is no promotion
            dest_path.write_text(json.dumps({"f1_macro": 0.99}, indent=2)); return
        raise Exception(f"Unexpected key: {key}")

    client.download_file.side_effect = download_file

    out = m.evaluate_model()
    assert out["timestamp"] == version_ts
    remote_keys = {rk for _, rk in uploads}
    assert metrics_key in remote_keys  
