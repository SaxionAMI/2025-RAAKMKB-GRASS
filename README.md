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

---

## Full stack (Docker) — MinIO + InfluxDB + Grafana + MediaMTX + live inference

The project ships a `docker-compose.yml` that stands up the whole demo stack
around the packaged framework:

| Service   | Purpose                                   | URL |
|-----------|-------------------------------------------|-----|
| dagster   | training + batch inference UI             | http://localhost:3000 |
| inference | streaming sentiment (SSE), reads RTSP     | http://localhost:8000 |
| grafana   | dashboards incl. the stadium canvas       | http://localhost:3001 |
| minio     | object storage (models/artifacts)         | http://localhost:9001 (console) |
| influxdb  | sentiment time-series                     | http://localhost:8086 |
| mediamtx  | RTSP/WebRTC media server                  | rtsp://localhost:8554 |

### Build & run

The image installs the framework from the **private** repo, so the build needs
your SSH key available to BuildKit:

```bash
cp .env.example .env          # then edit INFLUXDB_TOKEN etc. for anything non-local
eval "$(ssh-agent -s)" && ssh-add
make build                    # DOCKER_BUILDKIT=1 docker compose build --ssh default
make up                       # or: make up-gpu
```

Grafana → http://localhost:3001 (admin/admin) → the **Football Stadium — Live
Crowd Sentiment** dashboard is auto-provisioned.

### Feeding it live streams

Publish up to N concurrent RTSP streams to MediaMTX as `stream1..streamN`:

```bash
ffmpeg -re -stream_loop -1 -i match1.mp4 -an -f rtsp rtsp://localhost:8554/stream1
# ...stream2, stream3, stream4 in parallel  (or: put clips in demo_clips/ and `make publish-streams`)
```

The framework's stream sensor sees them via MediaMTX's API and registers each
with the inference service, which runs `sentiment` per stream and writes
`inference_sentiment` points to InfluxDB tagged with `stream_id`. Every Grafana
panel keys off `stream_id`, so 4 streams show as 4 series automatically.

### The dashboard

Auto-provisioned panels (querying measurement `inference_sentiment`):
confidence-over-time per stream, per-class score, sentiment distribution, active
stream count, average confidence, and a **latest-sentiment-per-stream table**.

### Honest caveats

- **The stadium Canvas is a starter, not a finished visual.** Grafana has no
  built-in "stadium" panel. The provided `canvas` panel has four placeholder
  zones and the right query wired in; you add a stadium background image and bind
  each zone's color to its stream's sentiment in the panel editor. The table next
  to it shows the same data and works immediately, so the dashboard is useful
  before the art is done.
- **4 concurrent streams is the target, not a guarantee.** The inference service
  is built to handle multiple streams, but real concurrent throughput depends on
  your hardware (CPU vs GPU) and model size. Start with 2 streams, confirm both
  appear in Grafana, then scale to 4.
- **Change the InfluxDB token** (`INFLUXDB_TOKEN`) before anything beyond a local
  demo. The Flux queries assume bucket `audio_metrics`; if you rename it, update
  the dashboard queries.
