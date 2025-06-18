from dagster import Definitions

from grass_pipeline_dagster.assets import segmented_spectrograms, segment_new_audio
from grass_pipeline_dagster.prepare_assets import prepare_dataset
from grass_pipeline_dagster.train_assets import train_model
from grass_pipeline_dagster.evaluate_assets import evaluate_model
from grass_pipeline_dagster.log_segment_predictions import log_segment_predictions
from grass_pipeline_dagster.jobs import segment_incremental_job, full_pipeline_job

defs = Definitions(
    assets=[
        segmented_spectrograms,
        segment_new_audio,
        prepare_dataset,
        train_model,
        evaluate_model,
        log_segment_predictions,
    ],
    jobs=[
        segment_incremental_job,  # ✅ conditional flow
        full_pipeline_job         # ✅ manual full pipeline
    ]
)
