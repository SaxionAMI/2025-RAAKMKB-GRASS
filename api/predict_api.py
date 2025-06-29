from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import tempfile
import yaml
import subprocess

# Updated classifier with sliding window logic inside
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

@app.post("/predict_manual")
def predict_manual(input: ManualInput):
    path = input.audio_path
    print("🛠️ Swagger UI manual path:", path)
    sys.stdout.flush()

    if not os.path.exists(path):
        print("❌ Path not found:", path)
        return JSONResponse(content={"error": f"Manual path not found: {path}"}, status_code=404)

    try:
        label = classify_audio(path)
        print("🎯 Manual predicted:", label)
        return JSONResponse(content={"label": label})
    except Exception as e:
        print("❌ Error in manual classification:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return JSONResponse(content={"error": "Invalid JSON payload."}, status_code=400)

    print("🔥 Received payload:", payload)
    sys.stdout.flush()

    # === Direct audio_path use (manual/debug) ===
    if "audio_path" in payload:
        path = payload["audio_path"]
        print("🛠️ Manual mode path:", path)
        sys.stdout.flush()

        if not os.path.exists(path):
            print("❌ Path not found:", path)
            return JSONResponse(content={"error": f"Manual path not found: {path}"}, status_code=404)

        try:
            label = classify_audio(path)
            print("🎯 Manual predicted:", label)
            return JSONResponse(content={"label": label})
        except Exception as e:
            print("❌ Error in manual classification:", e)
            return JSONResponse(content={"error": str(e)}, status_code=500)

    results = []

    # This is your Docker mount point for both containers:
    MOUNT_ROOT = "/data"
    tasks = payload.get("tasks", [])

    if not tasks:
        print("⚠️ No tasks found in payload")
        return JSONResponse(content={"error": "No tasks provided."}, status_code=400)

    for idx, task in enumerate(tasks):
        print(f"🔄 Processing task {idx} with ID {task.get('id')}")
        original_audio_path = task["data"].get("audio", "").replace("\\", "/")
        if not original_audio_path:
            print(f"❌ No 'audio' field in task {idx}")
            return JSONResponse(content={"error": f"No audio field for task {task.get('id')}"}, status_code=400)

        # Remove leading slash and "data/" if present, then always join with /data root
        rel_path = original_audio_path.lstrip("/")
        if rel_path.startswith("data/"):
            rel_path = rel_path[5:]  # strip "data/"
        local_path = os.path.join(MOUNT_ROOT, rel_path)

        # Try various path combinations to resolve audio file
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
            print(f"❌ File not found (tried): {candidate_paths}")
            return JSONResponse(content={"error": f"Audio file not found: {candidate_paths}"}, status_code=404)

        print(f"🔍 Final resolved path: {found_path}")
        sys.stdout.flush()

        try:
            print("🧪 classify_audio called with:", found_path)
            label = classify_audio(found_path)
            print("✅ classify_audio returning:", label)
        except Exception as e:
            print(f"❌ Error classifying task {task.get('id')}:", e)
            return JSONResponse(content={"error": f"Classification error: {str(e)}"}, status_code=500)

        result = {
            "result": [{
                "from_name": "global_sentiment",
                "to_name": "audio",
                "type": "choices",
                "value": {"choices": [label]}
            }],
            "score": 1.0,
            "model_version": "v1.0",
            "task": task["id"]
        }

        print(f"✅ Prediction result: {result}")
        results.append(result)
        sys.stdout.flush()

    if not results:
        print("⚠️ No predictions generated")
        return JSONResponse(content={"error": "No predictions produced."}, status_code=500)

    result_payload = {"results": results}
    print("✅ Final response:", result_payload)
    sys.stdout.flush()
    return JSONResponse(content=result_payload)


@app.post("/train")
async def train(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        print("🔥 Received training payload:", payload)
        sys.stdout.flush()
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return JSONResponse(content={"error": "Invalid JSON payload."}, status_code=400)

    # Extract annotation and task info from Label Studio payload
    annotation = payload.get("annotation")
    task = payload.get("task")

    if not annotation or not task:
        print("❌ Missing annotation or task in payload")
        return JSONResponse(content={"error": "Missing annotation or task in payload"}, status_code=400)

    audio_path = task.get("data", {}).get("audio")
    label = None
    # Defensive extraction of label from annotation results
    results = annotation.get("result", [])
    if results:
        label = results[0].get("value", {}).get("choices", [None])[0]

    if not audio_path or not label:
        print("❌ Missing audio path or label in annotation")
        return JSONResponse(content={"error": "Missing audio path or label in annotation"}, status_code=400)

    print(f"🎯 Incremental training triggered for audio: {audio_path} with label: {label}")

    def run_training_job():
        try:
            # Create temporary config file for Dagster job with annotation data
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_config:
                config_dict = {
                    "ops": {
                        "run_ls_segment_audios": {
                            "config": {
                                "audio_path": audio_path,
                                "label": label
                            }
                        },
                    }
                }
                yaml.dump(config_dict, tmp_config)
                tmp_config_path = tmp_config.name

            print(f"🚀 Launching Dagster LS training job with config: {tmp_config_path}")
            subprocess.run([
                "dagster", "job", "execute",
                "-f", "grass_pipeline_dagster/grass_pipeline_dagster/Label_Studio_Training/jobs_ls.py",
                "-j", "ls_training_job",
                "-c", tmp_config_path
            ], check=True)
            print("✅ Training job triggered.")
            sys.stdout.flush()

        except Exception as e:
            print(f"❌ Failed to launch Dagster job: {e}")

        finally:
            # Cleanup temp file
            try:
                os.remove(tmp_config_path)
            except Exception:
                pass

    background_tasks.add_task(run_training_job)

    return JSONResponse({"status": "Incremental training started"}, status_code=200)

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    # Alias /webhook to /train to handle LS training webhooks uniformly
    return await train(request, background_tasks)
