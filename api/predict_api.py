from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys

# Updated classifier with sliding window logic inside
from api.audio_classifier import classify_audio

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/setup")
def setup():
    return {"status": "setup complete"}

# === Manual Swagger UI Testing Input ===
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

# === Label Studio Inference Format ===
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

    # === Label Studio task-based inference ===
    results = []

    # This matches the volume mount for BOTH containers: ./label_studio_data:/data
    shared_media_root = "/data/media"
    tasks = payload.get("tasks", [])

    if not tasks:
        print("⚠️ No tasks found in payload")
        return JSONResponse(content={"error": "No tasks provided."}, status_code=400)

    for idx, task in enumerate(tasks):
        print(f"🔄 Processing task {idx} with ID {task.get('id')}")
        relative_audio_path = task["data"].get("audio", "").lstrip("/")
        # Remove leading "data/" if present, so we join below (LS always gives `/data/upload/...`)
        if relative_audio_path.startswith("data/"):
            relative_audio_path = relative_audio_path[len("data/"):]
        # Now reconstruct full path under /data/media/...
        local_path = os.path.join(shared_media_root, *relative_audio_path.split("/")[1:])
        print(f"🔍 Final resolved path: {local_path}")
        sys.stdout.flush()

        if not os.path.exists(local_path):
            print("❌ File not found:", local_path)
            return JSONResponse(content={"error": f"Audio file not found: {local_path}"}, status_code=404)

        try:
            print("🧪 classify_audio called with:", local_path)
            label = classify_audio(local_path)
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
