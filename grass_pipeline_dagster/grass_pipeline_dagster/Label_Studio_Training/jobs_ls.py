# grass_pipeline_dagster/Label_Studio_Training/jobs_ls.py

from dagster import job, op, Out, OpExecutionContext, String, Noneable

from grass_pipeline_dagster.Label_Studio_Training.ls_segment_audios import (
    ls_segment_audios_incremental,  # incremental segmentation
)

from grass_pipeline_dagster.Label_Studio_Training.train_on_ls_data import (
    train_on_ls_data,  # updated to accept segments list
)


@op(
    config_schema={
        "audio_path": Noneable(String),
        "label": Noneable(String),
    },
    out=Out(),
)
def run_ls_segment_and_train(context: OpExecutionContext):
    audio_path = context.op_config.get("audio_path")
    label = context.op_config.get("label")

    if not audio_path or not label:
        raise ValueError("Both 'audio_path' and 'label' must be provided in config.")

    context.log.info(f"Starting incremental LS pipeline for: {audio_path} (label={label})")

    # Step 1: Segment the audio
    segments = ls_segment_audios_incremental(audio_path=audio_path, label=label)

    if not segments:
        context.log.warning("No segments returned from incremental segmentation.")
        return

    context.log.info(f"{len(segments)} segments generated. Starting training...")

    # Step 2: Train the model on these segments
    model_path = train_on_ls_data(segments)

    context.log.info(f"Training complete. Model saved to: {model_path}")


@job
def ls_training_job():
    run_ls_segment_and_train()
