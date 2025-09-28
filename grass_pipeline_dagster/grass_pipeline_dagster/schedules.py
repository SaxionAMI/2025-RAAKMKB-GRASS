# schedules.py

from dagster import ScheduleDefinition
from grass_pipeline_dagster.jobs import segment_incremental_job

segment_incremental_schedule = ScheduleDefinition(
    job=segment_incremental_job,
    cron_schedule="0 0 * * *",  # Every day at 00:00 UTC
    name="segment_incremental_schedule"
)
