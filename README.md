# GRASS: crowd sentiment in football stadiums

Classifies stadium crowd audio as **approval**, **disapproval** or **neutral**,
per second and per live stream, with a Grafana dashboard for operators.

This repository belongs to the research project
[GRASS](https://www.saxion.nl/onderzoek/overige-projecten/ambient-intelligence/grass---gastvrij-en-respectvol-gedrag-van-supporters-in-stadions)
(*Gastvrij en Respectvol gedrAg van Supporters in Stadions*) of the Saxion
research group Ambient Intelligence, co-funded by Regieorgaan SIA (RAAK-mkb).

It is built on the
[Audio Ops Framework](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework)
([docs](https://saxionami.github.io/2026-AMI-AudioOpsFramework/)). This repo
contains only the project's own parts: the `sentiment` task, the Docker stack
and the dashboard. Feature extraction, splits, training, evaluation, batch
scoring and the live inference service come from the framework package.

```text
tasks/sentiment/
  config.yaml            labels, features, training settings
  task_sentiment.py      ResNet18 model, extra metric, InfluxDB record
scripts/setup.sh         one-time local setup (make setup)
scripts/stack.sh         Docker stack with checks (make up, make down, ...)
tests/                   checks the task against the installed framework
audio_input/sentiment/CrowdAudioDataport/<label>/*.wav    training audio (not in git)
docker-compose.yml       MinIO, InfluxDB, Grafana, MediaMTX, Dagster, inference
grafana/                 datasource and dashboard, provisioned automatically
```

## Requirements

- Linux or macOS. On Windows, use [WSL2](https://learn.microsoft.com/windows/wsl/install)
  with Ubuntu and run everything inside it.
- Python 3.10 to 3.14 (3.12 is a good choice), git and make.
- ffmpeg, for live RTSP streams.
- Docker with Compose v2, only for the full stack.

While the framework repository is private you also need read access to
[SaxionAMI/2026-AMI-AudioOpsFramework](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework)
and an [SSH key on your GitHub account](https://github.com/settings/keys).
Alternatively, clone the framework next to this repo
(`../2026-AMI-AudioOpsFramework`); the setup uses that checkout when GitHub
can't be reached.

## Quick start

```bash
make setup                                   # one time, safe to run again
# put audio in audio_input/sentiment/CrowdAudioDataport/<label>/
make train                                   # train, evaluate and score
make predict FILE=some_clip.wav              # score a file
make up                                      # the full stack in Docker
```

## Setup

```bash
make setup          # runs ./scripts/setup.sh
```

The script:

1. finds Python 3.10 to 3.14 and creates a virtual environment in `.venv`;
2. installs PyTorch and torchvision: the CUDA build if `nvidia-smi` finds an
   NVIDIA GPU, otherwise the much smaller CPU-only build;
3. installs the framework at the version in `pyproject.toml` (`v1.2.0`). It
   tries GitHub over HTTPS, then over SSH, then a framework checkout next to
   this repo;
4. creates `.env` and the runtime folders;
5. checks the task and reports how many audio files it finds per label.

If something is missing, the script says what to do. `./scripts/setup.sh --help`
lists the options, for example `--framework ../2026-AMI-AudioOpsFramework` to
work on the framework at the same time, or `--torch cpu` to force the CPU build.

The `make` targets use `.venv` automatically. To run `audio-ops` yourself,
activate it first: `source .venv/bin/activate`.

Poetry also works (`poetry install`) if you prefer it. The script is the
tested way.

## Training audio

Put labelled clips in `audio_input/sentiment/CrowdAudioDataport/`, one folder
per label:

```text
audio_input/sentiment/CrowdAudioDataport/approval/*.wav
audio_input/sentiment/CrowdAudioDataport/disapproval/*.wav
audio_input/sentiment/CrowdAudioDataport/neutral/*.wav
```

WAV, FLAC, MP3, OGG and M4A all work. Run `make setup` again to see the number of
files per label.

The split keeps each recording in one of train, validation or test. If you cut
one match recording into many files, give them a shared prefix
(`match3_001.wav`, `match3_002.wav`) and enable `group_pattern` in
`tasks/sentiment/config.yaml`. The whole match then stays in one split. The
split needs at least three recordings (or matches) in total; for a reliable
evaluation, use many more per label.

## Train and score

```bash
make train                                  # features, splits, training, evaluation, batch scoring
make predict FILE=some_clip.wav             # score one or more files
make test                                   # check that the task works with the framework
```

`make train` runs `audio-ops run sentiment`. It writes:

- the model to `storage/sentiment/checkpoints/best_model.pt`;
- the evaluation (accuracy, macro and weighted F1) to
  `storage/sentiment/outputs/evaluation_sentiment.json`;
- the scores for every segment to `storage/sentiment/outputs/` as JSON and CSV.

To follow runs in the Dagster UI instead, use `make dev` and open
<http://localhost:3000>. The jobs are `sentiment_full_training_job` and
`sentiment_inference_job`. To score a new folder without retraining, launch
`sentiment_inference_job` with this run config:

```yaml
ops:
  sentiment_raw_audio_files:
    config:
      input_dir: /data/new_match
      output_dir: /data/new_match_out
```

`make serve` starts the inference API on <http://localhost:8000/docs> without
Docker, for quick tests with file uploads.

## Full stack with Docker

```bash
make up             # builds the image the first time, starts everything, waits until ready
make up-gpu         # the same with an NVIDIA GPU (needs the NVIDIA Container Toolkit)
```

`make up` checks that Docker is installed and running, builds the image if
needed and waits for the inference service. It warns if no model has been
trained yet or if `.env` still has the demo InfluxDB token.

| Service | Purpose | URL |
|---|---|---|
| dagster | training and batch scoring | <http://localhost:3000> |
| inference | live sentiment per stream | <http://localhost:8000/docs> |
| grafana | dashboard (login from `.env`, default admin / admin) | <http://localhost:3001> |
| minio | object storage console (default minioadmin / minioadmin) | <http://localhost:9001> |
| influxdb | time series | <http://localhost:8086> |
| mediamtx | RTSP server | rtsp://localhost:8554 |

The tasks folder and `storage/` are shared with the containers, so a model
trained with `make train` on your machine is also used by the stack. After
training (on your machine or in Dagster), run `make restart` so the inference
service loads the new model.

Other commands: `make status`, `make logs`, `make down` (stops everything and
keeps the data) and `make build` (rebuild the image, for example after changing
the framework version).

The image installs the framework from GitHub. While the repository is private,
`make build` uses your SSH key for that. It starts an SSH agent and asks for
the key's passphrase if needed.

### Live streams

Publish audio to MediaMTX as `stream1`, `stream2` and so on:

```bash
ffmpeg -re -stream_loop -1 -i match1.mp4 -vn -c:a aac -f rtsp rtsp://localhost:8554/stream1
# or put match1.mp4 to match4.mp4 in demo_clips/ and run: make publish-streams
```

The inference service checks MediaMTX every few seconds and starts listening
to every new stream. For each second of audio it writes a point to the
InfluxDB measurement `inference_sentiment` with the tag `stream_id` and the
fields `label`, `confidence` and `score_approval`, `score_disapproval`,
`score_neutral`. The results are also available as Server-Sent Events on
`/streams/{stream_id}/events`.

### Dashboard

The *Football Stadium: Live Crowd Sentiment* dashboard is provisioned
automatically. It shows confidence over time per stream, the score per class,
the label distribution, the number of active streams, the average confidence
and a table with the latest label per stream. All panels group by `stream_id`,
so every stream appears as its own series.

## Troubleshooting

| Message or problem | What to do |
|---|---|
| `No Python 3.10 to 3.14 found` | Install one, e.g. `sudo apt install python3.12 python3.12-venv`, or point to it: `./scripts/setup.sh --python /path/to/python3.12` |
| `Could not create the virtual environment` | `sudo apt install python3-venv` (Debian/Ubuntu) |
| `Could not get the framework` | Ask for access to the framework repo and add an SSH key to GitHub, or clone the framework next to this repo. If your key has a passphrase: `eval "$(ssh-agent -s)" && ssh-add`, then `make setup` again |
| `Permission denied` when running a script | `bash scripts/setup.sh` (the Makefile already does this) |
| `make: command not found` | `sudo apt install make`, or run the scripts directly: `./scripts/setup.sh` |
| `audio-ops: command not found` | Use the `make` targets, or `source .venv/bin/activate` first |
| `Need at least 3 recordings` | Add more recordings per label, or check `group_pattern` |
| `Docker is installed but not running` | Start Docker Desktop, or `sudo systemctl start docker` |
| Grafana shows no data | Check `make status`: is a model loaded? Are streams published? After training: `make restart` |
| InfluxDB refuses the token after you changed it | InfluxDB keeps the token it was first created with. `docker compose down -v` recreates it (this deletes stored metrics) |

### Known limitations

- The stadium canvas panel is a starting point. It has four placeholder zones
  with the query already set up. Add a stadium image as background and bind
  each zone's colour to its stream in the panel editor. The table next to it
  shows the same data.
- How many streams run in real time depends on the hardware. Start with two
  streams, check that both appear in Grafana, then add more. A GPU
  (`make up-gpu`) helps most.
- The InfluxDB queries assume the bucket `audio_metrics`. If you rename it,
  update the dashboard queries.

## Upgrading the framework

Change the tag in `pyproject.toml`, `Dockerfile` (`ARG FRAMEWORK_REF`) and
`docker-compose.yml` (`make test` checks that they match), then run
`make setup`, `make test` and `make build`. The framework's
[changelog](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/blob/main/CHANGELOG.md)
lists what changed.

Since framework 1.1, checkpoints are loaded in weights-only mode. If an older
`best_model.pt` doesn't load, retrain it with `make train`, or set
`ALLOW_UNSAFE_CHECKPOINTS=true` once for a checkpoint you trust.

## Contact

E.M. van Wolfswinkel, Saxion research group Ambient Intelligence:
[e.m.vanwolfswinkel@saxion.nl](mailto:e.m.vanwolfswinkel@saxion.nl)

## License

Copyright 2026 Saxion University of Applied Sciences. Licensed under the Apache
License 2.0, like the framework. Audio recordings are not part of this
repository.
