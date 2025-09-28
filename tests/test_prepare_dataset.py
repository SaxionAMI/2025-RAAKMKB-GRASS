# tests/test_prepare_dataset.py
import os
import sys
from types import ModuleType
from pathlib import Path
import importlib.util
import pandas as pd
import pytest
from unittest.mock import MagicMock


# load prepare_assets.py 
def _load_prepare_assets_module():
    tests_dir = Path(__file__).resolve().parent
    repo_root = tests_dir.parent
    mod_path = repo_root / "grass_pipeline_dagster" / "grass_pipeline_dagster" / "prepare_assets.py"
    assert mod_path.exists(), f"prepare_assets.py not found at {mod_path}"

    
    if "grass_pipeline_dagster" not in sys.modules:
        sys.modules["grass_pipeline_dagster"] = ModuleType("grass_pipeline_dagster")
    utils_stub = ModuleType("grass_pipeline_dagster.utils")

    
    def _default_load_settings():
        return {"class_labels": ["approval", "neutral", "disapproval"]}
    def _default_get_minio_client():
        return MagicMock()

    utils_stub.load_settings = _default_load_settings
    utils_stub.get_minio_client = _default_get_minio_client
    sys.modules["grass_pipeline_dagster.utils"] = utils_stub

    spec = importlib.util.spec_from_file_location("prepare_assets_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  
    return mod


def _touch_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")  


def _make_balanced_dataset(base_img_dir: Path):
    """
    Create enough clips to satisfy stratified split with test_size=0.15:
    - 16 clips total, 8 per class.
    - Each clip gets 1–2 segment images.
    """
    
    for i in range(8):
        _touch_png(base_img_dir / "approval" / f"A{i}_segment_00.png")
        if i % 2 == 0:
            _touch_png(base_img_dir / "approval" / f"A{i}_segment_01.png")
    
    for i in range(8):
        _touch_png(base_img_dir / "neutral" / f"N{i}_segment_00.png")
        if i % 2 == 0:
            _touch_png(base_img_dir / "neutral" / f"N{i}_segment_01.png")

# (Split from original test_extract_clip_id)
def test_extract_clip_id_basic():
    m = _load_prepare_assets_module()
    assert m.extract_clip_id("approval/cheer1_segment_03.png") == "cheer1"


def test_extract_clip_id_underscored():
    m = _load_prepare_assets_module()
    assert m.extract_clip_id("neutral/foo_bar_segment_12.png") == "foo_bar"


def test_extract_clip_id_no_segment_suffix():
    m = _load_prepare_assets_module()
    assert m.extract_clip_id("approval/clip.png") == "clip"


# Split from original test_prepare_dataset_happy_path
def test_prepare_dataset_happy_path_schema_and_counts(monkeypatch, tmp_path):
    m = _load_prepare_assets_module()
    monkeypatch.chdir(tmp_path)

    base_img_dir = tmp_path / "dagster_output" / "spectrogram_images"
    _make_balanced_dataset(base_img_dir)

    
    monkeypatch.setattr(m, "load_settings", lambda: {"class_labels": ["approval", "neutral"]})
    mock_client = MagicMock()
    monkeypatch.setattr(m, "get_minio_client", lambda: mock_client)

    result = m.prepare_dataset()

    # return structure
    assert set(result.keys()) == {"train_path", "test_path", "train_size", "test_size"}

    # CSVs exist & schema
    train_df = pd.read_csv(result["train_path"])
    test_df = pd.read_csv(result["test_path"])
    for df in (train_df, test_df):
        assert set(df.columns) == {"path", "label"}
    assert result["train_size"] + result["test_size"] == 24
    assert result["test_size"] > 0

# (Split from original test_prepare_dataset_happy_path) — part 2: uploads
def test_prepare_dataset_happy_path_minio_uploads(monkeypatch, tmp_path):
    m = _load_prepare_assets_module()
    monkeypatch.chdir(tmp_path)

    base_img_dir = tmp_path / "dagster_output" / "spectrogram_images"
    _make_balanced_dataset(base_img_dir)

    monkeypatch.setattr(m, "load_settings", lambda: {"class_labels": ["approval", "neutral"]})
    mock_client = MagicMock()
    monkeypatch.setattr(m, "get_minio_client", lambda: mock_client)

    result = m.prepare_dataset()
    assert result  # not strictly needed; ensures call executed

    # two MinIO uploads to datasets/train.csv & datasets/test.csv
    calls = mock_client.upload_file.call_args_list
    assert len(calls) == 2
    remote_keys = {calls[0].args[2], calls[1].args[2]}
    assert remote_keys == {"datasets/train.csv", "datasets/test.csv"}
    assert calls[0].args[1] == calls[1].args[1] == "grass-pipeline"


# (Original test_prepare_dataset_minio_upload_failure)
def test_prepare_dataset_minio_upload_failure(monkeypatch, tmp_path):
    m = _load_prepare_assets_module()
    monkeypatch.chdir(tmp_path)

    base_img_dir = tmp_path / "dagster_output" / "spectrogram_images"
    _make_balanced_dataset(base_img_dir)

    monkeypatch.setattr(m, "load_settings", lambda: {"class_labels": ["approval", "neutral"]})

    failing_client = MagicMock()
    failing_client.upload_file.side_effect = Exception("Simulated MinIO failure")
    monkeypatch.setattr(m, "get_minio_client", lambda: failing_client)

    with pytest.raises(RuntimeError, match="Failed to upload"):
        m.prepare_dataset()
