from dagster import job, op, Out, In, Nothing, OpExecutionContext, String, Noneable

from grass_pipeline_dagster.Label_Studio_Training.ls_segment_audios import (
    ls_segment_audios,               # asset (full segmentation)
    ls_segment_audios_incremental,  # incremental function
)
from grass_pipeline_dagster.Label_Studio_Training.generate_ls_data import generate_ls_data
from grass_pipeline_dagster.Label_Studio_Training.train_on_ls_data import train_on_ls_data

@op(config_schema={
    "audio_path": Noneable(String),
    "label": Noneable(String),
}, out=Out(Nothing))
def run_ls_segment_audios(context: OpExecutionContext):
    audio_path = context.op_config.get("audio_path")
    label = context.op_config.get("label")
    if audio_path and label:
        context.log.info(f"Incremental segmentation: {audio_path} with label: {label}")
        ls_segment_audios_incremental(audio_path=audio_path, label=label)
    else:
        context.log.info("Full segmentation (no audio_path or label provided)")
        ls_segment_audios()

@op(ins={"start": In(Nothing)}, out=Out(Nothing))
def run_generate_ls_data():
    generate_ls_data()

@op(ins={"start": In(Nothing)})
def run_train_on_ls_data():
    train_on_ls_data()

@job
def ls_training_job():
    seg = run_ls_segment_audios()
    csvs = run_generate_ls_data(start=seg)
    run_train_on_ls_data(start=csvs)
