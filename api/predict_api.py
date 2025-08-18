# predict_api.py
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import tempfile
import yaml
import subprocess
from pathlib import Path
import json

# Updated classifier (returns regions: [{"start":..,"end":..,"label":..}, ...])
from api.audio_classifier import classify_audio

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/setup")
def setup():
    return {"status": "setup complete"}

class ManualInput(BaseModel):
    audio_path: str

def _regions_to_ls_results(regions, from_name="region_labels", to_name="audio"):
    """Convert regions into Label Studio 'labels' results."""
    region_results = []
    for reg in regions:
        region_results.append({
            "from_name": from_name,
            "to_name": to_name,
            "type": "labels",
            "value": {
                "start": float(reg["start"]),
                "end": float(reg["end"]),
                "labels": [reg["label"]],
            },
        })
    return region_results

@app.post("/predict_manual")
def predict_manual(input: ManualInput):
    path = input.audio_path
    print(" Swagger UI manual path:", path)
    sys.stdout.flush()

    if not os.path.exists(path):
        print(" Path not found:", path)
        return JSONResponse(content={"error": f"Manual path not found: {path}"}, status_code=404)

    try:
        regions = classify_audio(path)
        print(" Manual predicted regions:", regions)
        # Return LS-style results for consistency with /predict
        ls_results = _regions_to_ls_results(regions)
        return JSONResponse(content={"result": ls_results, "model_version": "v1.0-regions"})
    except Exception as e:
        print(" Error in manual classification:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        print(" Failed to parse JSON:", e)
        return JSONResponse(content={"error": "Invalid JSON payload."}, status_code=400)

    print("Received payload:", payload)
    sys.stdout.flush()

    # === Direct audio_path use (manual/debug) ===
    if "audio_path" in payload:
        path = payload["audio_path"]
        print(" Manual mode path:", path)
        sys.stdout.flush()

        if not os.path.exists(path):
            print("Path not found:", path)
            return JSONResponse(content={"error": f"Manual path not found: {path}"}, status_code=404)

        try:
            regions = classify_audio(path)
            print("Manual predicted regions:", regions)
            ls_results = _regions_to_ls_results(regions)
            return JSONResponse(content={"result": ls_results, "model_version": "v1.0-regions"})
        except Exception as e:
            print("Error in manual classification:", e)
            return JSONResponse(content={"error": str(e)}, status_code=500)

    results = []
    MOUNT_ROOT = "/data"
    tasks = payload.get("tasks", [])

    if not tasks:
        print("No tasks found in payload")
        return JSONResponse(content={"error": "No tasks provided."}, status_code=400)

    for idx, task in enumerate(tasks):
        print(f"Processing task {idx} with ID {task.get('id')}")
        original_audio_path = task["data"].get("audio", "").replace("\\", "/")
        if not original_audio_path:
            print(f"No 'audio' field in task {idx}")
            return JSONResponse(content={"error": f"No audio field for task {task.get('id')}"}, status_code=400)

        # Resolve file path
        rel_path = original_audio_path.lstrip("/")
        if rel_path.startswith("data/"):
            rel_path = rel_path[5:]
        local_path = os.path.join(MOUNT_ROOT, rel_path)

        found_path = None
        candidate_paths = [
            local_path,
            original_audio_path,
            os.path.join(MOUNT_ROOT, "media", rel_path),
            os.path.join(MOUNT_ROOT, "upload", rel_path),
            os.path.join(MOUNT_ROOT, rel_path.lstrip("media/")),
            os.path.join(MOUNT_ROOT, rel_path.lstrip("upload/")),
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                found_path = candidate
                break

        if not found_path:
            print(f"File not found (tried): {candidate_paths}")
            return JSONResponse(content={"error": f"Audio file not found: {candidate_paths}"}, status_code=404)

        print(f"Final resolved path: {found_path}")
        sys.stdout.flush()

        try:
            regions = classify_audio(found_path)
            print("classify_audio regions:", regions)
        except Exception as e:
            print(f"Error classifying task {task.get('id')}:", e)
            return JSONResponse(content={"error": f"Classification error: {str(e)}"}, status_code=500)

        # === Build Label Studio region results ===
        region_results = _regions_to_ls_results(regions)

        result = {
            "result": region_results,
            "score": 1.0,
            "model_version": "v1.0-regions",
            "task": task["id"]
        }

        print(f"Prediction result: {result}")
        results.append(result)
        sys.stdout.flush()

    if not results:
        print("No predictions generated")
        return JSONResponse(content={"error": "No predictions produced."}, status_code=500)

    result_payload = {"results": results}
    print("Final response:", result_payload)
    sys.stdout.flush()
    return JSONResponse(content=result_payload)


# -----------------------------
# TRAIN / WEBHOOK
# -----------------------------
@app.post("/train")
async def train(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        print("Received training payload:", payload)
        sys.stdout.flush()
    except Exception as e:
        print("Failed to parse JSON:", e)
        return JSONResponse(content={"error": "Invalid JSON payload."}, status_code=400)

    # Quick path: LS project-level START_TRAINING -> process latest export and trigger jobs
    if isinstance(payload, dict) and payload.get("action") == "START_TRAINING":
        print("Received START_TRAINING action; will process /data/export latest file and trigger training jobs in background.")
        sys.stdout.flush()

        def run_export_training():
            export_dir = Path("/data/export")
            if not export_dir.exists():
                print("Export dir /data/export does not exist; nothing to do.")
                return

            # pick latest JSON export (ignore -info.json)
            export_files = sorted([p for p in export_dir.glob("*.json") if not str(p).endswith("-info.json")],
                                  key=lambda p: p.stat().st_mtime, reverse=True)
            if not export_files:
                print("No export JSON files found in /data/export")
                return

            latest = export_files[0]
            print(f"Using export file: {latest}")
            try:
                with open(latest, "r") as fh:
                    ls_data = json.load(fh)
            except Exception as e:
                print(f"Failed to read export file {latest}: {e}")
                return

            # normalize to list of tasks
            tasks = []
            if isinstance(ls_data, list):
                tasks = ls_data
            elif isinstance(ls_data, dict):
                tasks = ls_data.get("tasks") or next((v for v in ls_data.values() if isinstance(v, list)), [])
            else:
                print("Unrecognized LS export format.")
                return

            for task in tasks:
                anns = task.get("annotations") or task.get("results") or []
                if not isinstance(anns, list) or len(anns) == 0:
                    continue
                annotation = anns[0]
                audio_path = task.get("data", {}).get("audio") or task.get("file_upload")
                if not audio_path:
                    print("Task missing audio path; skipping task.")
                    continue

                # extract region segments if present
                segments = []
                ann_results = annotation.get("result", []) if isinstance(annotation, dict) else []
                for res in ann_results:
                    if not isinstance(res, dict):
                        continue
                    val = res.get("value", {}) or {}
                    if "start" in val and "end" in val:
                        lbl = None
                        if "labels" in val and isinstance(val["labels"], list) and val["labels"]:
                            lbl = val["labels"][0]
                        elif "choices" in val and isinstance(val["choices"], list) and val["choices"]:
                            lbl = val["choices"][0]
                        if lbl is not None:
                            try:
                                segments.append({
                                    "start": float(val["start"]),
                                    "end": float(val["end"]),
                                    "label": str(lbl)
                                })
                            except Exception:
                                continue

                # fallback to single label (legacy)
                label_fallback = None
                if not segments:
                    if isinstance(annotation, dict):
                        ann_res = annotation.get("result", [])
                        if isinstance(ann_res, list) and len(ann_res) > 0:
                            first = ann_res[0]
                            val = first.get("value", {}) if isinstance(first, dict) else {}
                            if "choices" in val and val["choices"]:
                                label_fallback = val["choices"][0]
                            elif "labels" in val and val["labels"]:
                                label_fallback = val["labels"][0]

                if not segments and not label_fallback:
                    print("No segments or fallback label for task; skipping.")
                    continue

                # Build config and run dagster job for this task
                tmp_config_path = None
                try:
                    if segments:
                        config_dict = {
                            "ops": {
                                "run_ls_segment_and_train": {
                                    "config": {
                                        "audio_path": audio_path,
                                        "segments": segments
                                    }
                                }
                            }
                        }
                    else:
                        config_dict = {
                            "ops": {
                                "run_ls_segment_and_train": {
                                    "config": {"audio_path": audio_path, "label": label_fallback}
                                }
                            }
                        }

                    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_config:
                        yaml.dump(config_dict, tmp_config)
                        tmp_config_path = tmp_config.name

                    print(f"Launching Dagster LS training job with config: {tmp_config_path}")
                    subprocess.run([
                        "dagster", "job", "execute",
                        "-f", "grass_pipeline_dagster/grass_pipeline_dagster/Label_Studio_Training/jobs_ls.py",
                        "-j", "ls_training_job",
                        "-c", tmp_config_path
                    ], check=True)
                    print("Training job triggered for task.")
                except Exception as e:
                    print(f"Failed to launch Dagster job for task: {e}")
                finally:
                    if tmp_config_path and os.path.exists(tmp_config_path):
                        try:
                            os.remove(tmp_config_path)
                        except Exception:
                            pass

        background_tasks.add_task(run_export_training)
        return JSONResponse({"status": "START_TRAINING accepted; training jobs queued"}, status_code=200)

    # For other shapes: try to extract (annotation, task) as before
    annotation = payload.get("annotation")
    task = payload.get("task")

    # If payload is list (export), take first task
    if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
        task = payload[0]
        anns = task.get("annotations") or task.get("results") or []
        if isinstance(anns, list) and len(anns) > 0:
            annotation = anns[0]

    # If payload is a task dict (task with annotations)
    if isinstance(payload, dict) and ("annotations" in payload or "predictions" in payload or "data" in payload):
        if task is None:
            task = payload
        anns = task.get("annotations") or task.get("results") or []
        if isinstance(anns, list) and len(anns) > 0:
            annotation = anns[0]

    if not annotation or not task:
        print("Missing annotation or task in payload")
        return JSONResponse(content={"error": "Missing annotation or task in payload"}, status_code=400)

    # extract audio_path from task
    audio_path = task.get("data", {}).get("audio")
    # Try to build region segments from the annotation result entries
    segments = []
    ann_results = annotation.get("result", []) if isinstance(annotation, dict) else []

    for res in ann_results:
        if not isinstance(res, dict):
            continue
        val = res.get("value", {}) or {}
        # look for timed region result {start, end, labels} or {start, end, choices}
        if "start" in val and "end" in val:
            lbl = None
            if "labels" in val and isinstance(val["labels"], list) and val["labels"]:
                lbl = val["labels"][0]
            elif "choices" in val and isinstance(val["choices"], list) and val["choices"]:
                lbl = val["choices"][0]
            if lbl is not None:
                try:
                    segments.append({
                        "start": float(val["start"]),
                        "end": float(val["end"]),
                        "label": str(lbl)
                    })
                except Exception:
                    # skip malformed numeric values
                    continue

    # If no segments found, fall back to extracting a single label (legacy)
    label = None
    if not segments:
        if isinstance(ann_results, list) and len(ann_results) > 0:
            first = ann_results[0]
            val = first.get("value", {}) if isinstance(first, dict) else {}
            if "choices" in val and val["choices"]:
                label = val["choices"][0]
            elif "labels" in val and val["labels"]:
                label = val["labels"][0]

    if not audio_path or (not segments and not label):
        print("Missing audio path or label/segments in annotation")
        return JSONResponse(content={"error": "Missing audio path or label/segments in annotation"}, status_code=400)

    print(f"Incremental training triggered for audio: {audio_path} with segments_count={len(segments)} label_fallback={label}")
    sys.stdout.flush()

    def run_training_job():
        tmp_config_path = None
        try:
            if segments:
                config_dict = {
                    "ops": {
                        "run_ls_segment_and_train": {
                            "config": {
                                "audio_path": audio_path,
                                "segments": segments
                            }
                        }
                    }
                }
            else:
                config_dict = {
                    "ops": {
                        "run_ls_segment_and_train": {
                            "config": {"audio_path": audio_path, "label": label}
                        }
                    }
                }

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_config:
                yaml.dump(config_dict, tmp_config)
                tmp_config_path = tmp_config.name

            print(f"Launching Dagster LS training job with config: {tmp_config_path}")
            subprocess.run([
                "dagster", "job", "execute",
                "-f", "grass_pipeline_dagster/grass_pipeline_dagster/Label_Studio_Training/jobs_ls.py",
                "-j", "ls_training_job",
                "-c", tmp_config_path
            ], check=True)
            print("Training job triggered.")
            sys.stdout.flush()

        except Exception as e:
            print(f"Failed to launch Dagster job: {e}")
            sys.stdout.flush()

        finally:
            if tmp_config_path and os.path.exists(tmp_config_path):
                try:
                    os.remove(tmp_config_path)
                except Exception:
                    pass

    background_tasks.add_task(run_training_job)
    return JSONResponse({"status": "Incremental training started"}, status_code=200)


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    return await train(request, background_tasks)
