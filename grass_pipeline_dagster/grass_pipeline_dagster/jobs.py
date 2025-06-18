# grass_pipeline_dagster/jobs.py

from dagster import job, op, OpExecutionContext, In, Out, Nothing

from grass_pipeline_dagster.assets import segmented_spectrograms, segment_new_audio
from grass_pipeline_dagster.prepare_assets import prepare_dataset
from grass_pipeline_dagster.train_assets import train_model
from grass_pipeline_dagster.evaluate_assets import evaluate_model
from grass_pipeline_dagster.log_segment_predictions import log_segment_predictions


# ========= INCREMENTAL JOB (with conditional logic) =========

@op
def run_segmentation(context: OpExecutionContext) -> str:
    result = segment_new_audio()
    context.log.info(f"Segmentation result: {result}")
    return result

@op
def run_prepare_dataset(context: OpExecutionContext, result: str) -> bool:
    if "Segmented" in result and "0 new files" not in result:
        context.log.info("✅ New data found — proceeding with dataset preparation.")
        prepare_dataset()
        return True
    else:
        context.log.info("⏭️ No new data — skipping prepare_dataset.")
        return False

@op
def run_train_model(context: OpExecutionContext, should_train: bool) -> bool:
    if should_train:
        context.log.info("🎯 Proceeding with model training.")
        train_model()
        return True
    else:
        context.log.info("⏭️ Skipping training step.")
        return False

@op
def run_evaluate_model(context: OpExecutionContext, should_evaluate: bool) -> bool:
    if should_evaluate:
        context.log.info("📊 Proceeding with evaluation step.")
        evaluate_model()
        return True
    else:
        context.log.info("⏭️ Skipping evaluation step.")
        return False

@op
def run_log_predictions(context: OpExecutionContext, should_log: bool):
    if should_log:
        context.log.info("📌 Logging predictions to InfluxDB.")
        log_segment_predictions()
    else:
        context.log.info("⏭️ Skipping prediction logging step.")

@job
def segment_incremental_job():
    result = run_segmentation()
    should_train = run_prepare_dataset(result)
    should_evaluate = run_train_model(should_train)
    should_log = run_evaluate_model(should_evaluate)
    run_log_predictions(should_log)


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
