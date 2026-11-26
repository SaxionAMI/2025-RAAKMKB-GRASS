"""Checks that the sentiment task works with the installed Audio Ops Framework.

Run with ``make test`` (or ``poetry run pytest``). The last test trains for one
epoch on a few seconds of synthetic audio, so the whole file takes ~30 s on CPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "tasks" / "sentiment"
LABELS = ["approval", "disapproval", "neutral"]


@pytest.fixture(scope="module")
def task():
    os.environ["AUDIO_OPS_TASKS_DIR"] = str(REPO / "tasks")
    from audio_ops_dagster.registry import discover_tasks, get_task

    discover_tasks()
    config = yaml.safe_load((TASK_DIR / "config.yaml").read_text())
    return get_task("sentiment", config)


def test_config_is_valid():
    from audio_ops import validate_task_config

    errors, _ = validate_task_config(yaml.safe_load((TASK_DIR / "config.yaml").read_text()))
    assert errors == []


def test_model_accepts_the_configured_input_shape(task):
    shape = tuple(task.config.input_shape)
    model = task.build_model(shape).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, *shape))
    assert out.shape == (2, len(LABELS))


def test_output_has_label_scores_and_probabilities(task):
    result = task.postprocess(torch.tensor([0.1, 2.0, -1.0]))
    assert result["label"] == "disapproval"
    assert set(result["scores"]) == set(LABELS)
    assert result["probabilities"] == result["scores"]
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-5


def test_metrics_include_weighted_f1(task):
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.eye(3)[[0, 1, 2, 1]]
    metrics = task.compute_metrics(y_true, y_pred)
    assert {"accuracy", "f1_macro", "f1_weighted"} <= set(metrics)


def test_influx_record_has_per_class_fields(task):
    record = task.format_influx_record("seg", task.postprocess(torch.zeros(3)),
                                       audio_offset_sec=1.5)
    assert {f"prob_{c}" for c in LABELS} <= set(record["fields"])
    assert record["tags"]["predicted_label"] in LABELS


def _write_audio(root: Path) -> None:
    import soundfile as sf

    sr, rng = 16000, np.random.default_rng(0)
    for label, freq in zip(LABELS, (500, 180, 900)):
        (root / label).mkdir(parents=True)
        for rec in range(4):
            t = np.arange(int(sr * 2.0)) / sr
            y = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(t.size)
            sf.write(root / label / f"match{rec}_000.wav", y.astype(np.float32), sr)


def test_pipeline_trains_and_predicts(tmp_path):
    """features, splits, one epoch of training, evaluation, batch scoring, predict."""
    tasks = tmp_path / "tasks"
    shutil.copytree(TASK_DIR, tasks / "sentiment",
                    ignore=shutil.ignore_patterns("__pycache__"))
    cfg_path = tasks / "sentiment" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(epochs=1, input_dir=str(tmp_path / "audio"), output_dir=str(tmp_path / "out"))
    cfg_path.write_text(yaml.safe_dump(cfg))
    _write_audio(tmp_path / "audio")

    env = {**os.environ, "AUDIO_OPS_TASKS_DIR": str(tasks), "ENABLE_INFLUXDB": "false",
           "AUDIO_OPS_DISABLE_BUILTIN_TASKS": "true", "STORAGE_BACKEND": "local",
           "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path}/mlflow.db"}
    cli = [sys.executable, "-m", "audio_ops_dagster.cli"]
    run = subprocess.run([*cli, "run", "sentiment"], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / "out" / "checkpoints" / "best_model.pt").exists()
    assert (tmp_path / "out" / "outputs" / "inference_results_sentiment.csv").exists()

    clip = tmp_path / "audio" / "neutral" / "match0_000.wav"
    pred = subprocess.run([*cli, "predict", "sentiment", str(clip)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)
    assert pred.returncode == 0, pred.stdout + pred.stderr
    assert any(label in pred.stdout for label in LABELS)
