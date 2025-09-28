# predict_api.py
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, sys, tempfile, yaml, subprocess, json
from typing import List, Dict, Any


from api.audio_classifier import classify_audio

app = FastAPI()


# Utilities

class ManualInput(BaseModel):
    audio_path: str

def _regions_to_ls_results(regions, from_name="region_labels", to_name="audio"):
    out = []
    for reg in regions:
        out.append({
            "from_name": from_name,
            "to_name": to_name,
            "type": "labels",
            "value": {
                "start": float(reg["start"]),
                "end": float(reg["end"]),
                "labels": [reg["label"]],
            },
        })
    return out

def _launch_dagster(config_dict: dict):
    tmp_config_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_config:
            yaml.dump(config_dict, tmp_config)
            tmp_config_path = tmp_config.name
        print(f"[TRAIN] Launching Dagster job with config: {tmp_config_path}")
        subprocess.run([
            "dagster", "job", "execute",
            "-f", "grass_pipeline_dagster/grass_pipeline_dagster/Label_Studio_Training/jobs_ls.py",
            "-j", "ls_training_job",
            "-c", tmp_config_path
        ], check=True)
        print("[TRAIN] Dagster job completed.")
        sys.stdout.flush()
    except Exception as e:
        print(f"[TRAIN] Failed to launch Dagster job: {e}")
    finally:
        if tmp_config_path and os.path.exists(tmp_config_path):
            try:
                os.remove(tmp_config_path)
            except Exception:
                pass

def _train_from_segments(audio_path: str, segments: List[Dict[str, Any]], background_tasks: BackgroundTasks):
    def run():
        config = {
            "ops": {
                "run_ls_segment_and_train": {
                    "config": {"audio_path": audio_path, "segments": segments}
                }
            }
        }
        _launch_dagster(config)
    background_tasks.add_task(run)
    return JSONResponse(
        {"status": "Incremental training started", "audio": audio_path, "segments": len(segments)},
        status_code=200
    )

def _train_from_fallback_label(audio_path: str, label: str, background_tasks: BackgroundTasks):
    def run():
        config = {
            "ops": {
                "run_ls_segment_and_train": {
                    "config": {"audio_path": audio_path, "label": label}
                }
            }
        }
        _launch_dagster(config)
    background_tasks.add_task(run)
    return JSONResponse(
        {"status": "Incremental training started", "audio": audio_path, "label": label},
        status_code=200
    )


# Endpoints

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/setup")
def setup():
    return {"status": "setup complete"}

@app.post("/predict_manual")
def predict_manual(input: ManualInput):
    path = input.audio_path
    print("[PREDICT_MANUAL] path:", path); sys.stdout.flush()
    if not os.path.exists(path):
        return JSONResponse(content={"error": f"Manual path not found: {path}"}, status_code=404)

    try:
        regions = classify_audio(path)
        print("[PREDICT_MANUAL] regions:", regions)
        return JSONResponse(content={"result": _regions_to_ls_results(regions), "model_version": "v1.0-regions"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON payload."}, status_code=400)

    print("[PREDICT] payload:", payload); sys.stdout.flush()

    results = []
    MOUNT_ROOT = "/data"
    tasks = payload.get("tasks", [])

    if not tasks:
        return JSONResponse(content={"error": "No tasks provided."}, status_code=400)

    for task in tasks:
        tid = str(task.get("id"))
        original_audio_path = (task.get("data", {}) or {}).get("audio", "")
        if not original_audio_path:
            return JSONResponse(content={"error": f"No audio field for task {tid}"}, status_code=400)

        # Try common path resolutions
        rel_path = original_audio_path.lstrip("/")
        if rel_path.startswith("data/"): rel_path = rel_path[5:]
        candidates = [
            os.path.join(MOUNT_ROOT, rel_path),
            original_audio_path,
            os.path.join(MOUNT_ROOT, "media", rel_path),
            os.path.join(MOUNT_ROOT, "upload", rel_path),
        ]
        found_path = next((c for c in candidates if os.path.exists(c)), None)
        if not found_path:
            print(f"[PREDICT] File not found (tried): {candidates}")
            return JSONResponse(content={"error": f"Audio file not found: {candidates}"}, status_code=404)

        print(f"[PREDICT] Final resolved path: {found_path}"); sys.stdout.flush()

        # Run classifier
        regions = classify_audio(found_path)
        print("[PREDICT] regions:", regions)

        result = {
            "result": _regions_to_ls_results(regions),
            "score": 1.0,
            "model_version": "v1.0-regions",
            "task": task["id"],
        }
        results.append(result)

    return JSONResponse(content={"results": results})


# WEBHOOK (train on submit/update)

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON payload."}, status_code=400)

    print("[WEBHOOK] payload:", payload); sys.stdout.flush()
    action = payload.get("action") or payload.get("event")

    # Handle annotation events → train
    if action in {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"} or (payload.get("annotation") and payload.get("task")):
        task = payload.get("task")
        annotation = payload.get("annotation")

        if not (task and annotation):
            return JSONResponse(content={"error": "Missing task/annotation in webhook"}, status_code=400)

        audio_path = (task.get("data") or {}).get("audio")
        if not audio_path:
            return JSONResponse(content={"error": "Task missing audio path"}, status_code=400)

        # Parse annotation results
        segments = []
        ann_results = annotation.get("result", []) if isinstance(annotation, dict) else []
        for res in ann_results:
            if not isinstance(res, dict): continue
            val = res.get("value", {}) or {}
            if "start" in val and "end" in val:
                labels = val.get("labels") or val.get("choices") or []
                if labels:
                    try:
                        segments.append({
                            "start": float(val["start"]),
                            "end": float(val["end"]),
                            "label": str(labels[0])
                        })
                    except Exception:
                        continue

        # Fallback: single label
        label = None
        if not segments and ann_results:
            v = ann_results[0].get("value", {})
            if isinstance(v.get("labels"), list) and v["labels"]:
                label = v["labels"][0]
            elif isinstance(v.get("choices"), list) and v["choices"]:
                label = v["choices"][0]

        if segments:
            return _train_from_segments(audio_path, segments, background_tasks)
        elif label:
            return _train_from_fallback_label(audio_path, label, background_tasks)
        else:
            return JSONResponse(content={"error": "No valid segments/label in annotation"}, status_code=400)

    return JSONResponse({"status": f"ignored action {action or 'UNKNOWN'}"}, status_code=202)
