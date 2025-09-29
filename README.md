# GRASS Crowd Sentiment Detection System

This branch contains the full MLOps system for the GRASS Project , a simulation-based near-real-time crowd sentiment detection system that analyzes crowd sentiment using audio data from stadiums other relevant audio data. The pipeline segments raw audio using C++ and FFTW, generates spectrograms, prepare datasets, trains and evaluates CNN models, and logs predictions using differnt ML tools.

---

## Install Local Requirements

Install the requirements via:

```bash
pip install -r requirements.tx
```

---

## Required Folder Structure

Before running the pipeline, **create a folder named `audio_input` in the project root**. This folder must contain **labeled class subfolders** with WAV files for each sentiment class.

Example structure:

```
GRASS_Pipeline/
├── audio_input/
│   ├── approval/
│   ├── disapproval/
│   └── neutral/
```

These folders are used for segmentation and dataset preparation.


## Required for the Realtime Project_root\api\api_realtime\stream_listener.py

The GUI sends live audio clips to the real-time FastAPI container (`localhost:8010/infer`), which uses a trained model to perform inference. This model must already exist at the path defined in your configuration file Project_root\config\settings.json:

```json
{
  "model_path": "dagster_output/model/latest/cnn_model.pth",
  ...
}
```
Ensure the model exists at this path before launching the GUI. You can generate it by running the Dagster pipeline.

After everything is set up:

```bash
python stream_listener.py
```

This launches a desktop interface to:
- Start/stop mic listening
- Stream audio segments to FastAPI
- Display real-time predictions, confidence, latency, and logs

---


## How to Run the System (via Docker Compose)

The system uses multiple services: Dagster, MLflow, MinIO, InfluxDB, FastAPI, Label Studio, and Grafana.

### Start all services:

```bash
docker-compose up
```

### Access the services:

- **Dagster** – Orchestration UI: http://localhost:3000  
- **MLflow** – Experiment tracking: http://localhost:5000  
- **MinIO** – Model/artifact storage: http://localhost:9001  *(minioadmin:minioadmin)*  
- **InfluxDB** – Sentiment logs: http://localhost:8086 *(admin:admin123)*  
- **FastAPI** – Inference endpoint: http://localhost:8000 *(docs: /docs)*  
- **Label Studio** – Annotation tool: http://localhost:8080  
- **Grafana** – Dashboards: http://localhost:33000 *(admin:admin)*
- **grass_pipeline-realtime** - Realtime API Inference: http://localhost:8010 *(docs: /docs)*

---

## Project Root Structure

| Path | Description |
|------|-------------|
| `.gitlab-ci.yml` | GitLab CI/CD pipeline |
| `docker-compose.yml` | Multi-container service definitions |
| `docker-compose.override.yml` | Local overrides for GPU training |
| `tests/` | Unit tests |
| `api/` | FastAPI sentiment classification for Label Studio |
| `audio_input/` |  labeled `.wav` inputs (not versioned for now) |
| `build/` | C++ integration related, u can ignore it |
| `config/` | System configs (`settings.json`) |
| `cpp_processing/` | C++ pybind11 FFTW segmentation logic, ignore it too |

---

## Dagster Core: `grass_pipeline_dagster/`

| Path | Description |
|------|-------------|
| `pyproject.toml` | Module definition |
| `requirements.dagster.txt` | Dependencies |
| `Dockerfile.dagster` | Dagster Docker container |

---

## Dagster Python Module: `grass_pipeline_dagster/grass_pipeline_dagster/`

| File/Folder | Description |
|-------------|-------------|
| `__init__.py` | Python package init |
| `definitions.py` | Dagster `Definitions(...)` declaration |
| `assets.py` | segmentation + spectrogram generation |
| `prepare_assets.py` | Dataset preparation, train.csv, test.csv |
| `train_assets.py` | CNN training with versioning |
| `evaluate_assets.py` | Metrics & confusion matrix evaluation |
| `log_segment_predictions.py` | Logs segment predictions to InfluxDB |
| `jobs.py` | Dagster job definitions |
| `utils.py` | MinIO/config helpers |

---

## Label Studio Integration: `Label_Studio_Training/`

| File | Description |
|------|-------------|
| `jobs_ls.py` | Job to retrain model on LS-labeled segments |
| `ls_segment_audios.py` | Segments and maps audio via LS JSON |
| `train_on_ls_data.py` | Trains model from LS-generated segments |


# Label Studio Setup

Label Studio is used in GRASS for annotation and human-in-the-loop retraining. It runs as a separate container in the Docker Compose stack.

## 1. Accessing Label Studio

Once services are up with `docker-compose up`, Label Studio is available at:

- **Label Studio UI**: http://localhost:8080

On first access, you will be asked to create an account (username/email + password). This account is used to log into the Label Studio dashboard.

---

## 2. Creating a Project

1. After login, click **Create Project** and choose a name (e.g., *Crowd Sentiment Labeling*).
2. Under **Labeling Interface**, configure audio classification e.g., (3 classes: Approval, Disapproval, Neutral).

```bash
<View>
  <Audio name="audio" value="$audio"/>
  <Labels name="region_labels" toName="audio">
    <Label value="Approval" background="#22c55e"/>
    <Label value="Disapproval" background="#ef4444"/>
    <Label value="Neutral" background="#9ca3af"/>
  </Labels>
</View>
```
3. Upload your dataset (audio files or pre-segmented tasks).

---

## 3. Connecting the Model Backend

The FastAPI service in GRASS exposes an API that Label Studio uses for pre-labeling and prediction retrieval.

- **Model URL (inside Docker network)**: http://api:8000  
- **Local access**: http://localhost:8000  

In the project settings, go to **Settings → Model → Connected Model** and set:

```
http://api:8000
```

This allows Label Studio to send segments for inference and receive predicted labels.

---

## 4. Webhook Configuration

To enable active learning retraining:

1. Navigate to **Settings → Webhooks** in your project.
2. Add a webhook with the following URL:

```
http://host.docker.internal:8000/webhook
```
Make sure to tick 3 options when adding the webhook link

1.Send payload,
2.Annotation created,
3.Annotation updated,

This sends annotation events from Label Studio to the GRASS FastAPI service, which can trigger Dagster jobs for retraining.

---

## 5. Workflow Summary

- Annotators upload or label audio segments in Label Studio.
- Label Studio can pre-fill labels using the connected GRASS model (`/infer`).
- On submission, a webhook notifies GRASS (`/webhook`).
- Dagster `ls_training_job` can then retrain the model with updated data.

---

## System Components Overview

| Component | Description |
|----------|-------------|
| **Audio ingestion** | From `.wav` files |
| **Segmentation** | FFTW-based 1s windows (0.25s overlap) via C++ |
| **Spectrogram generation** | Each segment → RGB image |
| **Dataset preparation** | Train/test CSV + metadata |
| **Model training** | CNN (ResNet18) using PyTorch |
| **Evaluation** | Accuracy, F1, CM, loss |
| **Logging** | Latency, confidence, true label to InfluxDB |
| **Storage** | MinIO (datasets, models, metrics) |
| **Monitoring** | Grafana visualizes InfluxDB logs |
| **Experiment tracking** | MLflow logs losses, metrics, tags |
| **Annotation** | Label Studio enables interactive labeling |
| **CI/CD** | GitLab auto-tests & deployments |
| **Containerization** | All orchestrated using Docker Compose |

---

## System Requirements

- Docker  
- Docker Compose  
- Labeled audio under `audio_input/`

---

## License

This project is licensed under the MIT License.
