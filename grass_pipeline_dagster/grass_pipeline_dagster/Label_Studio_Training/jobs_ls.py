from dagster import job, op, Out, OpExecutionContext, String, Noneable

from grass_pipeline_dagster.Label_Studio_Training.ls_segment_audios import (
    ls_segment_audios_incremental,  # now accepts optional segments list
)

from grass_pipeline_dagster.Label_Studio_Training.train_on_ls_data import (
    train_on_ls_data,
)


@op(
    config_schema={
        "audio_path": Noneable(String),
        "label": Noneable(String),
        # webhook may pass 'segments' (list of dicts) — accept as optional
        "segments": Noneable(list),
    },
    out=Out(),
)
def run_ls_segment_and_train(context: OpExecutionContext):
    """
    Run incremental LS pipeline. Config options:
      - audio_path (required)
      - segments (optional): list of dicts with {start, end, label} (from LS annotation export)
      - label (optional): fallback single label for whole audio
    If segments provided, will generate spectrograms for those segments and feed them to training.
    Otherwise will segment the full audio and use `label`.
    """
    audio_path = context.op_config.get("audio_path")
    label = context.op_config.get("label")
    provided_segments = context.op_config.get("segments")

    if not audio_path:
        raise ValueError("Config must include 'audio_path'.")

    # Logging context
    if provided_segments:
        context.log.info(
            f"Using provided segments for audio: {audio_path} (count={len(provided_segments)})"
        )
    else:
        if not label:
            raise ValueError("When 'segments' not provided, 'label' must be provided.")
        context.log.info(f"Starting incremental LS pipeline for: {audio_path} (label={label})")

    # Generate spectrogram image paths + labels.
    # ls_segment_audios_incremental supports both modes:
    #   - (audio_path, label) -> segments generated and labeled with label
    #   - (audio_path, None, segments=provided_segments) -> uses provided timed segments
    try:
        if provided_segments:
            segments_list = ls_segment_audios_incremental(
                audio_path=audio_path, label=None, segments=provided_segments
            )
        else:
            segments_list = ls_segment_audios_incremental(audio_path=audio_path, label=label)
    except Exception as e:
        context.log.error(f"Segmentation step failed: {e}")
        raise

    if not segments_list:
        context.log.warning("No spectrograms produced from provided segments.")
        return

    context.log.info(f"{len(segments_list)} segment spectrograms generated. Starting training...")

    # Step 2: Train the model on these (image_path, label) items
    try:
        model_path = train_on_ls_data(segments_list)
    except Exception as e:
        context.log.error(f"Training failed: {e}")
        raise

    context.log.info(f"Training complete. Model saved to: {model_path}")


@job
def ls_training_job():
    run_ls_segment_and_train()
