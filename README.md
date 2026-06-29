# Football Stadium Sentiment

Crowd-sentiment classification for football stadium audio — **approval /
disapproval / neutral** — built on the [Audio Ops Framework](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework).

This repo is intentionally thin: it contains only this project's task and data.
All the pipeline machinery (feature extraction, splitting, training, evaluation,
batch inference, the Dagster wiring) comes from the framework package.

```
football-stadium-sentiment/
  pyproject.toml          # depends on the framework (git package)
  workspace.yaml          # Dagster loads audio_ops_dagster.definitions
  tasks/sentiment/        # THIS project's task (the only code here)
    config.yaml
    task_sentiment.py
  audio_input/sentiment/CrowdAudioDataport/<label>/*.wav
```

## Prerequisites

- Python 3.10–3.14 and [Poetry](https://python-poetry.org/).
- Access to the framework repo. It's private, so installing it needs a
  credential — an **SSH key** on your GitHub account is the simplest (the
  dependency uses the `git@github.com:...` SSH URL).

## Setup

```bash
cp .env.example .env
poetry install          # pulls the framework from GitHub and installs it
mkdir -p dagster_home storage mlruns
```

If `poetry install` can't reach the framework repo, it's an auth issue: confirm
`ssh -T git@github.com` succeeds, or switch the dependency in `pyproject.toml` to
the HTTPS+token or local `path` form (both are shown there as comments).

## Add the audio

Drop labelled clips under the input dir — **the folder name is the label**:

```
audio_input/sentiment/CrowdAudioDataport/approval/*.wav
audio_input/sentiment/CrowdAudioDataport/disapproval/*.wav
audio_input/sentiment/CrowdAudioDataport/neutral/*.wav
```

Tip for the group split: if you cut one recording into many clips, give them a
shared filename prefix (e.g. `match3_001.wav`, `match3_002.wav`) so segments of
the same recording stay together across train/val/test.

## Run

```bash
make dev
# or explicitly:
export AUDIO_OPS_TASKS_DIR=$PWD/tasks
export DAGSTER_HOME=$PWD/dagster_home
poetry run dagster dev -w workspace.yaml      # http://localhost:3000
```

The framework discovers `tasks/sentiment` via `AUDIO_OPS_TASKS_DIR`, builds its
assets, and you materialize **`sentiment_full_training_job`**:
`raw_audio → features → splits → train → eval`. The checkpoint lands in
`storage/sentiment/checkpoints/best_model.pt`; evaluation metrics
(accuracy, weighted F1) show on the eval asset.

### Score a new batch (no retraining)

Launch `sentiment_inference_job` and set the run config on
`sentiment_raw_audio_files` to point at any folder:

```yaml
ops:
  sentiment_raw_audio_files:
    config:
      input_dir: /data/new_match
      output_dir: /data/new_match_out
```

Results (JSON + CSV) are written under `<output_dir>/outputs/`.

## What this project owns vs. the framework

- **This repo:** the `sentiment` task — a ResNet18 classifier head over
  spectrograms, its 3 classes, and its InfluxDB/Grafana/LLM hooks. Plus the
  audio data.
- **The framework:** everything else. Upgrade by bumping the framework tag in
  `pyproject.toml` and `poetry update audio-ops-framework`.

## Object storage / metrics (optional)

Local by default. To sync artifacts to MinIO/S3, set `STORAGE_BACKEND=minio`
+ `MINIO_*` in `.env` and flip `use_remote_storage: true` in
`tasks/sentiment/config.yaml`. InfluxDB metrics: `ENABLE_INFLUXDB=true` + the
`INFLUXDB_*` vars.
