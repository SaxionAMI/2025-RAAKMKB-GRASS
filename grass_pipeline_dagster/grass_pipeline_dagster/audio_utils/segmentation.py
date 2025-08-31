# grass_pipeline_dagster/audio_utils/segmentation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.io import wavfile
import torchaudio
from dagster import get_dagster_logger

from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
# NEW: import the module itself so we can see its _HAS_CPP flag
from cpp_processing import spectrogram as _spec

from grass_pipeline_dagster.utils import load_settings, get_minio_client


def segment_audio_files(audio_files: list) -> int:
    """
    Segments input audio files into overlapping 1D segments and saves spectrogram images.
    NOTE: We treat settings['segment_overlap'] as overlap in seconds.
          We compute hop = segment_duration - segment_overlap before calling the segmenter.

    Returns:
        int: total number of spectrogram images saved (and uploaded to MinIO)
    """
    logger = get_dagster_logger()

    # --- Backend announcement (once per run) ---
    _has_cpp = bool(getattr(_spec, "_HAS_CPP", False))
    _cpp_path = None
    try:
        import audio_cpp as _audio_cpp_mod  # pybind11 extension; .pyd on Windows, .so on Linux
        _cpp_path = getattr(_audio_cpp_mod, "__file__", None)
    except Exception:
        _cpp_path = None

    if _has_cpp:
        if _cpp_path:
            logger.info(f"Spectrogram backend: C++/FFTW via { _cpp_path }")
        else:
            logger.info("Spectrogram backend: C++/FFTW (audio_cpp path unavailable)")
    else:
        logger.info("Spectrogram backend: NumPy/np.fft (Python fallback).")
        if _cpp_path:
            logger.warning(
                "audio_cpp is importable (%s) but compute_spectrogram() is not using it "
                "(cpp_processing.spectrogram._HAS_CPP == False). Check module placement / imports.",
                _cpp_path,
            )

    settings = load_settings()
    minio_client = get_minio_client()
    bucket_name = "grass-pipeline"

    segment_duration = float(settings["segment_duration"])      # e.g., 1.0
    overlap_sec = float(settings["segment_overlap"])            # e.g., 0.25 (desired overlap SECONDS)
    # Convert desired overlap to hop (step) seconds for the segmenter
    hop_sec = max(1e-6, segment_duration - overlap_sec)         # e.g., 0.75

    # Guardrails + traceability
    assert 0 <= overlap_sec < segment_duration, (
        f"segment_overlap ({overlap_sec}) must be in [0, {segment_duration})"
    )
    logger.info(
        f"[Segmentation Params] window={segment_duration:.2f}s, "
        f"overlap={overlap_sec:.2f}s, hop={hop_sec:.2f}s"
    )

    output_dir = os.path.abspath(os.path.join(os.getcwd(), "dagster_output", "spectrogram_images"))
    os.makedirs(output_dir, exist_ok=True)

    total_saved = 0

    for file_index, audio_path in enumerate(audio_files):
        label = os.path.basename(os.path.dirname(audio_path)).strip().replace(" ", "_")
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        logger.info(f" Processing file {file_index + 1}/{len(audio_files)}: {audio_path}")

        try:
            if audio_path.lower().endswith(".mp3"):
                waveform, sample_rate = torchaudio.load(audio_path)
                audio_np = waveform[0].numpy()
            else:
                sample_rate, audio_np = wavfile.read(audio_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]
        except Exception as e:
            logger.warning(f"Failed to read {audio_path}: {e}")
            continue

        # Normalize audio before segmentation to reduce numerical variance
        audio_np = audio_np.astype(np.float32)
        audio_np = (audio_np - np.mean(audio_np)) / (np.std(audio_np) + 1e-8)

        # >>> IMPORTANT: call segmenter with hop_sec (computed from desired overlap) <<<
        segments = segment_audio(audio_np, sample_rate, segment_duration, hop_sec)
        logger.info(f" {len(segments)} segments created.")

        for seg_index, segment in enumerate(segments):
            # (segment already normalized above, but keep per-segment safety)
            segment = (segment - np.mean(segment)).astype(np.float32)
            segment /= (np.std(segment) + 1e-8)

            spec = compute_spectrogram(segment, sample_rate)
            log_spec = 10 * np.log10(spec + 1e-10)
            log_spec -= np.max(log_spec)
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

            # Upload to MinIO
            with open(image_path, "rb") as f:
                minio_key = f"spectrograms/{label}/{image_filename}"
                minio_client.upload_fileobj(f, bucket_name, minio_key)
                logger.info(f" Uploaded to MinIO: {minio_key}")

            logger.info(f" Saved spectrogram image: {image_path}")

            total_saved += 1
            del log_spec, spec, segment
            gc.collect()

    logger.info(f" Completed spectrogram extraction: {total_saved} segments total.")
    return total_saved
