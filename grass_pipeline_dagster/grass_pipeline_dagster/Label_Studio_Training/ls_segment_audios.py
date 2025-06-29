from dagster import asset, get_dagster_logger
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.io import wavfile
import torchaudio

from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
from grass_pipeline_dagster.utils import load_settings


AUDIO_BASE = Path("/data/media/upload/1")  # <- Single source of truth for all lookups

def resolve_audio_path(input_path):
    """
    Always resolves the real, container-accessible path for an audio file,
    given any Label Studio or API-supplied path.
    """
    # Handle if input_path is a Path object or string
    input_path = Path(input_path)
    fname = input_path.name
    # Compose full path
    true_path = AUDIO_BASE / fname
    if true_path.exists():
        return true_path
    # If not found, as a fallback, try absolute
    if input_path.is_absolute() and input_path.exists():
        return input_path
    raise FileNotFoundError(f"Audio file not found at: {true_path}")


@asset
def ls_segment_audios():
    """
    For every labeled audio in the latest LS export:
    - Segments it into chunks using your segmentation logic
    - Generates a spectrogram for each chunk
    - Saves all spectrograms to /app/dagster_output/label_studio/spectrogram_images/<label>/
    """
    logger = get_dagster_logger()
    LS_EXPORT_DIR = Path("/data/export")
    LS_SPECTROGRAM_DIR = Path("/app/dagster_output/label_studio/spectrogram_images")

    settings = load_settings()
    segment_duration = float(settings["segment_duration"])
    segment_overlap = float(settings["segment_overlap"])

    # Only consider .json files that are NOT -info.json
    export_files = sorted(
        [f for f in LS_EXPORT_DIR.glob("*.json") if not str(f).endswith("-info.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not export_files:
        logger.error("No LS task export files found in /data/export/ (skipping -info.json files)")
        return
    export_path = export_files[0]

    # --- Robust LS export loading (list or dict) ---
    with open(export_path, "r") as f:
        ls_data = json.load(f)
    logger.info(f"Using LS export: {export_path}")

    if isinstance(ls_data, dict):
        # For future-proofing, also check "tasks" or search for first list value
        tasks = ls_data.get("tasks")
        if not tasks:
            for v in ls_data.values():
                if isinstance(v, list):
                    tasks = v
                    break
        if not tasks:
            logger.error("No tasks found in LS export dict.")
            return
    elif isinstance(ls_data, list):
        tasks = ls_data
    else:
        logger.error("Unrecognized LS export format.")
        return

    for task in tasks:
        try:
            audio_rel_path = Path(task["data"]["audio"])
            audio_id = (task.get("file_upload") or audio_rel_path.stem).replace(".wav", "")
            label = task["annotations"][0]["result"][0]["value"]["choices"][0].lower()
        except Exception as e:
            logger.warning(f"Skipping task due to missing fields: {e}")
            continue

        # Use our robust resolver (no more AUDIO_BASES list)
        try:
            audio_path = resolve_audio_path(audio_rel_path)
        except FileNotFoundError as e:
            logger.warning(f"Audio file does not exist for {audio_rel_path.name}: {e}")
            continue

        out_folder = LS_SPECTROGRAM_DIR / label
        out_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎧 Processing LS-labeled audio: {audio_path} → label={label}")

        try:
            if audio_path.suffix.lower() == ".mp3":
                waveform, sample_rate = torchaudio.load(str(audio_path))
                audio_np = waveform[0].numpy()
            else:
                sample_rate, audio_np = wavfile.read(str(audio_path))
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]
        except Exception as e:
            logger.warning(f"⚠️ Failed to read {audio_path}: {e}")
            continue

        segments = segment_audio(audio_np, sample_rate, segment_duration, segment_overlap)
        logger.info(f"✂️ {len(segments)} segments created for {audio_path}")

        for seg_index, segment in enumerate(segments):
            segment = (segment - np.mean(segment)).astype(np.float32)
            segment /= (np.std(segment) + 1e-8)

            spec = compute_spectrogram(segment, sample_rate)
            log_spec = 10 * np.log10(spec + 1e-10)
            log_spec -= np.max(log_spec)
            log_spec = np.clip(log_spec, -80, 0)

            image_filename = f"{audio_id}_segment_{seg_index}.png"
            image_path = out_folder / image_filename

            plt.figure(figsize=(4, 4), frameon=False)
            ax = plt.gca()
            ax.axis("off")
            plt.imshow(
                log_spec,
                origin="lower",
                aspect="auto",
                cmap="jet",
                extent=[0, segment_duration, 0, sample_rate / 2],
            )
            plt.tight_layout()
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            logger.info(f"📸 LS: Saved spectrogram image: {image_path}")

            del log_spec, spec, segment
            gc.collect()

    logger.info("✅ LS-driven segmentation asset complete! All outputs in label_studio/spectrogram_images/.")


def ls_segment_audios_incremental(audio_path: str, label: str):
    """
    Segment a single audio file with a given label,
    generate spectrogram images, and save them.
    """
    logger = get_dagster_logger()
    LS_SPECTROGRAM_DIR = Path("/app/dagster_output/label_studio/spectrogram_images")
    
    settings = load_settings()
    segment_duration = float(settings["segment_duration"])
    segment_overlap = float(settings["segment_overlap"])

    # Always resolve to /data/media/upload/1/filename.ext
    try:
        audio_path = resolve_audio_path(audio_path)
    except FileNotFoundError as e:
        logger.error(f"⚠️ {e}")
        return

    out_folder = LS_SPECTROGRAM_DIR / label.lower()
    out_folder.mkdir(parents=True, exist_ok=True)
    audio_id = audio_path.stem

    try:
        if audio_path.suffix.lower() == ".mp3":
            waveform, sample_rate = torchaudio.load(str(audio_path))
            audio_np = waveform[0].numpy()
        else:
            sample_rate, audio_np = wavfile.read(str(audio_path))
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
    except Exception as e:
        logger.error(f"⚠️ Failed to read {audio_path}: {e}")
        return

    segments = segment_audio(audio_np, sample_rate, segment_duration, segment_overlap)
    logger.info(f"✂️ {len(segments)} segments created for {audio_path}")

    for seg_index, segment in enumerate(segments):
        segment = (segment - np.mean(segment)).astype(np.float32)
        segment /= (np.std(segment) + 1e-8)

        spec = compute_spectrogram(segment, sample_rate)
        log_spec = 10 * np.log10(spec + 1e-10)
        log_spec -= np.max(log_spec)
        log_spec = np.clip(log_spec, -80, 0)

        image_filename = f"{audio_id}_segment_{seg_index}.png"
        image_path = out_folder / image_filename

        plt.figure(figsize=(4, 4), frameon=False)
        ax = plt.gca()
        ax.axis("off")
        plt.imshow(
            log_spec,
            origin="lower",
            aspect="auto",
            cmap="jet",
            extent=[0, segment_duration, 0, sample_rate / 2],
        )
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        logger.info(f"📸 LS: Saved spectrogram image: {image_path}")

        del log_spec, spec, segment
        gc.collect()

    logger.info(f"✅ Incremental segmentation complete for: {audio_path}")

