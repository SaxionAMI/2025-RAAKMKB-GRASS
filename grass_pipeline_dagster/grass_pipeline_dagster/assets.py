from dagster import asset, get_dagster_logger
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torchaudio
import tempfile
import gc

from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
from grass_pipeline_dagster.utils import load_settings, get_minio_client


@asset
def segmented_spectrograms():
    logger = get_dagster_logger()
    logger.info("🔁 Running segmented_spectrograms()")
    settings = load_settings()
    minio_client = get_minio_client()
    bucket_name = "grass-pipeline"

    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", settings["input_dir"]))
    segment_duration = float(settings["segment_duration"])
    segment_overlap = float(settings["segment_overlap"])

    output_dir = os.path.abspath(os.path.join(os.getcwd(), "dagster_output", "spectrogram_images"))
    os.makedirs(output_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        raise FileNotFoundError(f"No .wav or .mp3 files found in directory: {input_dir}")

    logger.info(f"Found {len(audio_files)} audio files in: {input_dir}")
    total_saved = 0

    for file_index, audio_path in enumerate(audio_files):
        label = os.path.basename(os.path.dirname(audio_path)).strip().replace(" ", "_")
        original_name = os.path.splitext(os.path.basename(audio_path))[0]

        logger.info(f"Processing file {file_index + 1}/{len(audio_files)}: {audio_path}")

        # Load audio (mp3 or wav)
        if audio_path.lower().endswith(".mp3"):
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_np = waveform[0].numpy()  # mono
        else:
            sample_rate, audio_np = wavfile.read(audio_path)
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]  # mono

        segments = segment_audio(audio_np, sample_rate, window_sec=segment_duration, hop_sec=segment_overlap)
        logger.info(f"Generated {len(segments)} segments from this file.")

        for seg_index, segment in enumerate(segments):
            segment = segment.astype(np.float32)
            segment = segment - np.mean(segment)
            segment = segment / (np.std(segment) + 1e-8)

            spec = compute_spectrogram(segment, sample_rate)
            log_spec = 10 * np.log10(spec + 1e-10)
            log_spec = log_spec - np.max(log_spec)
            log_spec = np.clip(log_spec, -80, 0)

            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            image_filename = f"{original_name}_segment_{seg_index}.png"
            image_path = os.path.join(label_dir, image_filename)

            plt.figure(figsize=(4, 4), frameon=False)
            ax = plt.gca()
            ax.axis("off")
            plt.imshow(
                log_spec,
                origin='lower',
                aspect='auto',
                cmap='jet',
                extent=[0, segment_duration, 0, sample_rate / 2]
            )
            plt.tight_layout()
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            with open(image_path, "rb") as f:
                minio_key = f"spectrograms/{label}/{image_filename}"
                minio_client.upload_fileobj(f, bucket_name, minio_key)
                logger.info(f"Uploaded to MinIO: {minio_key}")

            total_saved += 1
            del log_spec, spec, segment
            gc.collect()

            logger.info(f"Saved spectrogram image: {image_path}")

    logger.info(f"Completed spectrogram extraction: {total_saved} segments total.")
    return f"Saved {total_saved} spectrograms"
