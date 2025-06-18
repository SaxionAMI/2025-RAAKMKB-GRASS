from dagster import multi_asset, get_dagster_logger, Output, AssetOut
import os
import glob
from typing import List
from grass_pipeline_dagster.utils import load_settings, get_minio_client


@multi_asset(
    outs={"new_audio_files": AssetOut()}
)
def check_new_audio_data() -> Output[List[str]]:
    logger = get_dagster_logger()
    logger.info("🔎 Checking MinIO for unsegmented audio files...")

    settings = load_settings()
    minio_client = get_minio_client()
    bucket_name = "grass-pipeline"

    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", settings["input_dir"]))
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3'))]

    if not audio_files:
        logger.warning(f"⚠️ No audio files found in input_dir: {input_dir}")
        return Output(value=[], output_name="new_audio_files")

    logger.info(f"📁 Found {len(audio_files)} audio files to check")
    new_files = []

    for audio_path in audio_files:
        label = os.path.basename(os.path.dirname(audio_path)).strip().replace(" ", "_")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        minio_prefix = f"spectrograms/{label}/{base_name}_segment_"

        response = minio_client.list_objects(Bucket=bucket_name, Prefix=minio_prefix)
        matched_keys = []

        if "Contents" in response:
            matched_keys = [obj["Key"] for obj in response["Contents"]]

        logger.debug(f"🔍 Checking prefix: {minio_prefix}")
        logger.debug(f"🔍 Matched MinIO keys: {matched_keys}")

        if not matched_keys:
            logger.info(f"🆕 New audio detected: {audio_path}")
            new_files.append(audio_path)
        else:
            logger.debug(f"✅ Already segmented: {audio_path} → Found {len(matched_keys)} segment(s)")

    logger.info(f"📊 {len(new_files)} new audio file(s) found for segmentation")

    if new_files:
        logger.info("📄 List of new audio files:")
        for path in new_files:
            logger.info(f"   • {path}")

    return Output(value=new_files, output_name="new_audio_files")
