# grass_pipeline_dagster/jobs.py

from dagster import job, op, OpExecutionContext, In, Out, Nothing

from grass_pipeline_dagster.assets import segmented_spectrograms, segment_new_audio
from grass_pipeline_dagster.prepare_assets import prepare_dataset
from grass_pipeline_dagster.train_assets import train_model
from grass_pipeline_dagster.evaluate_assets import evaluate_model
from grass_pipeline_dagster.log_segment_predictions import log_segment_predictions
from grass_pipeline_dagster.utils import load_settings
import os
import time
import ffmpeg
import aiohttp
import asyncio
from datetime import datetime, timezone
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from grass_pipeline_dagster.utils import load_settings, get_influx_client


# ========= INCREMENTAL JOB (with conditional logic) =========

@op
def run_segmentation(context: OpExecutionContext) -> str:
    result = segment_new_audio()
    context.log.info(f"Segmentation result: {result}")
    return result

@op
def run_prepare_dataset(context: OpExecutionContext, result: str) -> bool:
    if "Segmented" in result and "0 new files" not in result:
        context.log.info("New data found — proceeding with dataset preparation.")
        prepare_dataset()
        return True
    else:
        context.log.info("No new data — skipping prepare_dataset.")
        return False

@op
def run_train_model(context: OpExecutionContext, should_train: bool) -> bool:
    if should_train:
        context.log.info("Proceeding with model training.")
        train_model()
        return True
    else:
        context.log.info("Skipping training step.")
        return False

@op
def run_evaluate_model(context: OpExecutionContext, should_evaluate: bool) -> bool:
    if should_evaluate:
        context.log.info("Proceeding with evaluation step.")
        evaluate_model()
        return True
    else:
        context.log.info("Skipping evaluation step.")
        return False

@op
def run_log_predictions(context: OpExecutionContext, should_log: bool):
    if should_log:
        context.log.info("Logging predictions to InfluxDB.")
        log_segment_predictions()
    else:
        context.log.info("Skipping prediction logging step.")

@job
def segment_incremental_job():
    result = run_segmentation()
    should_train = run_prepare_dataset(result)
    should_evaluate = run_train_model(should_train)
    should_log = run_evaluate_model(should_evaluate)
    run_log_predictions(should_log)
    
# ========= STREAMING INFERENCE (listen to audio stream, classify, log) =========    
# RTSP/stream
#    ↓
# segmenter (streaming)
#    ↓
# async FastAPI predict_api inference   ← reuse existing code
#    ↓
# InfluxDB logging

async def send_to_predict_api(session, audio_path: str):
    settings = load_settings()
    PREDICT_API_URL = settings.get("predict_api_url")
    
    payload = {"audio_path": audio_path}
    async with session.post(PREDICT_API_URL, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"Predict API error {resp.status}: {text}")
        return await resp.json()

@op
def segmented_spectrograms_stream(context: OpExecutionContext):
    logger = context.log
    settings = load_settings()

    RTSP_URL = settings.get("rtsp_stream_url")
    segment_duration = float(settings.get("segment_duration", 1.0))
    segment_overlap = float(settings.get("segment_overlap", 0.25))
    SHARED_AUDIO_DIR = settings.get("shared_audio_data_dir", "/data/stream_segments")

    os.makedirs(SHARED_AUDIO_DIR, exist_ok=True)

    influx = get_influx_client()
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    
    label_to_numeric = {
    "approval": 0,
    "neutral": 1,
    "disapproval": 2
    }

    logger.info(f"RTSP source: {RTSP_URL}")
    logger.info(f"Segment duration={segment_duration}s overlap={segment_overlap}s")
    logger.info(f"Shared audio dir: {SHARED_AUDIO_DIR}")

    async def loop():
        segment_counter = 0

        async with aiohttp.ClientSession() as session:
            while True:
                segment_path = os.path.join(
                    SHARED_AUDIO_DIR,
                    f"segment_{segment_counter:010d}.wav"
                )

                try:
                    # 1. Capture segment from RTSP
                    (
                        ffmpeg
                        .input(RTSP_URL, rtsp_transport='tcp', t=segment_duration)
                        .output(
                            segment_path,
                            format='wav',
                            acodec='pcm_s16le',
                            ar='44100',
                            ac=1
                        )
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )

                    logger.info(f"Captured segment: {segment_path}")

                    # 2. Call predict API
                    result = await send_to_predict_api(session, segment_path)
                    logger.info(f"results from api: {result}")

                    regions = result.get("regions", [])       # start/end/label
                    scores_list = result.get("scores", [])    # per-segment class probabilities
                    final_label = result.get("final_label", None)
                    # 3. Log to InfluxDB
                    for i, r in enumerate(regions):
                        label = r["label"]
                        scores = scores_list[i] if i < len(scores_list) else {}
                        final_label_id  = label_to_numeric.get(final_label.lower(), -1)
                        point = (
                            Point("stream_predictions")
                            .tag("predicted_label", label.lower())
                            .tag("source", "rtsp_stream_1")
                            .field("final_label_id", int(final_label_id))
                            .field("segment_start", float(r["start"]))
                            .field("segment_end", float(r["end"]))
                            .field("file", os.path.basename(segment_path))
                            # Add class scores (default to 0.0 if missing)
                            .field("score_approval", float(scores.get("approval", 0.0)))
                            .field("score_disapproval", float(scores.get("disapproval", 0.0)))
                            .field("score_neutral", float(scores.get("neutral", 0.0)))
                            .time(datetime.now(timezone.utc))
                        )
                        write_api.write(bucket="predictions", org="grass", record=point)
                    logger.info(f"Logged {len(regions)} regions from segment {segment_counter}")

                except Exception as e:
                    logger.error(f"Streaming inference failed: {e}")

                finally:
                    # Optional: cleanup
                    try:
                        os.remove(segment_path)
                    except Exception:
                        pass

                segment_counter += 1
                await asyncio.sleep(max(0.0, segment_duration - segment_overlap))

    asyncio.run(loop())

    
@job
def streaming_inference_job():
    segmented_spectrograms_stream()

# ========= FULL MANUAL JOB (no conditions, sequential) =========

@op(out=Out(Nothing))
def run_segmented_spectrograms_manual():
    segmented_spectrograms()

@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def run_prepare_dataset_manual():
    prepare_dataset()

@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def run_train_model_manual():
    train_model()

@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def run_evaluate_model_manual():
    evaluate_model()

@op(ins={"start": In(Nothing)})
def run_log_predictions_manual():
    log_segment_predictions()

@job
def full_pipeline_job():
    s1 = run_segmented_spectrograms_manual()
    s2 = run_prepare_dataset_manual(start=s1)
    s3 = run_train_model_manual(start=s2)
    s4 = run_evaluate_model_manual(start=s3)
    run_log_predictions_manual(start=s4)
