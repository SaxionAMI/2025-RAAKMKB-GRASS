from dagster import asset, get_dagster_logger
import os
import glob

from grass_pipeline_dagster.utils import load_settings, get_minio_client
from grass_pipeline_dagster.audio_utils.segmentation import segment_audio_files


@asset
def segmented_spectrograms():
    logger = get_dagster_logger()
    logger.info("Running segmented_spectrograms()")

    # Load settings
    settings = load_settings()
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", settings["input_dir"]))

    # Get all .wav and .mp3 files recursively
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3'))]

    if not audio_files:
        raise FileNotFoundError(f"No .wav or .mp3 files found in directory: {input_dir}")

    logger.info(f"Found {len(audio_files)} audio files in: {input_dir}")

    # Delegate to reusable helper
    total_saved = segment_audio_files(audio_files)

    return f"Saved {total_saved} spectrograms"


@asset
def segment_new_audio():
    logger = get_dagster_logger()
    logger.info("Running segment_new_audio() with detailed MinIO check")

    # Load settings and initialize MinIO
    settings = load_settings()
    minio_client = get_minio_client()
    bucket_name = "grass-pipeline"

    # Locate input directory
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", settings["input_dir"]))
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3'))]

    if not audio_files:
        logger.warning(f"No audio files found in input_dir: {input_dir}")
        return "No audio files found."

    logger.info(f"Found {len(audio_files)} audio files to check")
    new_files = []

    for audio_path in audio_files:
        label = os.path.basename(os.path.dirname(audio_path)).strip().replace(" ", "_")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        minio_prefix = f"spectrograms/{label}/{base_name}_segment_"

        response = minio_client.list_objects(Bucket=bucket_name, Prefix=minio_prefix)
        matched_keys = []

        if "Contents" in response:
            matched_keys = [obj["Key"] for obj in response["Contents"]]

        logger.debug(f"Checking prefix: {minio_prefix}")
        logger.debug(f"Matched MinIO keys: {matched_keys}")

        if not matched_keys:
            logger.info(f"New audio detected: {audio_path}")
            new_files.append(audio_path)
        else:
            logger.debug(f"Already segmented: {audio_path} → Found {len(matched_keys)} segment(s)")

    logger.info(f" {len(new_files)} new audio file(s) found for segmentation")

    if not new_files:
        logger.info("No new audio files to segment.")
        return "No new files found."

    logger.info("List of new audio files:")
    for path in new_files:
        logger.info(f"   • {path}")

    total_saved = segment_audio_files(new_files)

    return f"Segmented {len(new_files)} new files → {total_saved} spectrograms"
