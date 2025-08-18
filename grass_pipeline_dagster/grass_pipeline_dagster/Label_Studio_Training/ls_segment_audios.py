from dagster import asset, get_dagster_logger
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.io import wavfile
import torchaudio
import os
from typing import List, Tuple

from cpp_processing.segmenter import segment_audio
from cpp_processing.spectrogram import compute_spectrogram
from grass_pipeline_dagster.utils import load_settings

# Legacy base (kept for backward compatibility)
AUDIO_BASE = Path("/data/media/upload/1")

# Some fallback roots we will try
FALLBACK_BASES = [
    Path("/data/media/upload"),
    Path("/data/upload"),
    Path("/data/media"),
    Path("/data"),
]

# Env vars to try if set in container config
ENV_MEDIA_VARS = ["LABEL_STUDIO_MEDIA_DIR", "LS_MEDIA_DIR", "LOCAL_LABEL_STUDIO_MEDIA"]


def resolve_audio_path(input_path):
    """
    Robust path resolution for audio files coming from Label Studio or API.

    Tries, in order:
      1. If provided path is absolute and exists -> return
      2. AUDIO_BASE/<basename>
      3. fallback bases (AUDIO_BASE siblings) + basename
      4. preserve relative path under common roots (/data, /data/media, /data/upload)
      5. environment configured media roots
      6. project-local label_studio_data/media/* (useful for local dev)
      7. rglob search under common roots
      8. finally try the as-given path
    Raises FileNotFoundError listing tried candidates if not found.
    """
    logger = get_dagster_logger()
    p = Path(str(input_path))
    fname = p.name
    tried = []

    logger.debug(f"resolve_audio_path: resolving {input_path} -> basename {fname}")

    # 1) absolute as-given
    if p.is_absolute():
        tried.append(str(p))
        if p.exists():
            logger.debug(f"resolve_audio_path: input is absolute and exists: {p}")
            return p

    # 2) AUDIO_BASE / basename
    cand = AUDIO_BASE / fname
    tried.append(str(cand))
    if cand.exists():
        logger.debug(f"resolve_audio_path: found at AUDIO_BASE {cand}")
        return cand

    # 3) fallback bases
    for base in FALLBACK_BASES:
        cand = base / fname
        tried.append(str(cand))
        if cand.exists():
            logger.debug(f"resolve_audio_path: found at fallback base {cand}")
            return cand

    # 4) preserve relative path under known roots
    rel = str(p).lstrip("/")
    candidate_roots = [Path("/data"), Path("/data/media"), Path("/data/media/upload"), Path("/data/upload")]
    for root in candidate_roots:
        cand = root / rel
        tried.append(str(cand))
        if cand.exists():
            logger.debug(f"resolve_audio_path: found at {cand}")
            return cand

    # 5) env-configured roots
    for var in ENV_MEDIA_VARS:
        val = os.environ.get(var)
        if not val:
            continue
        root = Path(val)
        cand1 = root / fname
        cand2 = root / rel
        tried.extend([str(cand1), str(cand2)])
        if cand1.exists():
            logger.debug(f"resolve_audio_path: found at env root {var}: {cand1}")
            return cand1
        if cand2.exists():
            logger.debug(f"resolve_audio_path: found at env root with rel {var}: {cand2}")
            return cand2

    # 6) project-local label_studio_data layout
    cwd = Path.cwd()
    local_candidates = [
        cwd / "label_studio_data" / "media" / "upload" / fname,
        cwd / "label_studio_data" / "media" / fname,
        cwd / "label_studio_data" / "media" / rel,
    ]
    for cand in local_candidates:
        tried.append(str(cand))
        if cand.exists():
            logger.debug(f"resolve_audio_path: found at project-local candidate {cand}")
            return cand

    # 7) quick rglob under common roots
    rglob_roots = [Path("/data/media/upload"), Path("/data/upload"), cwd / "label_studio_data" / "media"]
    for root in rglob_roots:
        if root.exists():
            logger.debug(f"resolve_audio_path: searching {root} for {fname}")
            for match in root.rglob(fname):
                tried.append(str(match))
                if match.exists():
                    logger.debug(f"resolve_audio_path: found by rglob {match}")
                    return match

    # 8) final fallback: as-given
    cand = Path(str(input_path))
    tried.append(str(cand))
    if cand.exists():
        logger.debug(f"resolve_audio_path: final fallback exists: {cand}")
        return cand

    # If we get here, nothing matched
    logger.error(f"Audio file not found at expected locations for: {input_path}. Tried: {tried}")
    raise FileNotFoundError(f"Audio file not found at expected locations for: {input_path}. Tried: {tried}")


@asset
def ls_segment_audios():
    """
    Legacy asset: read latest LS export and produce spectrograms for labeled audio tasks.
    (left intact for Dagster asset usage)
    """
    logger = get_dagster_logger()
    LS_EXPORT_DIR = Path("/data/export")
    LS_SPECTROGRAM_DIR = Path("/app/dagster_output/label_studio/spectrogram_images")

    settings = load_settings()
    segment_duration = float(settings["segment_duration"])
    segment_overlap = float(settings["segment_overlap"])

    export_files = sorted(
        [f for f in LS_EXPORT_DIR.glob("*.json") if not str(f).endswith("-info.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not export_files:
        logger.error("No LS task export files found in /data/export/")
        return

    export_path = export_files[0]
    with open(export_path, "r") as f:
        ls_data = json.load(f)
    logger.info(f"Using LS export: {export_path}")

    tasks = []
    if isinstance(ls_data, dict):
        tasks = ls_data.get("tasks") or next((v for v in ls_data.values() if isinstance(v, list)), [])
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

        try:
            audio_path = resolve_audio_path(audio_rel_path)
        except FileNotFoundError as e:
            logger.warning(f"Audio file not found: {e}")
            continue

        out_folder = LS_SPECTROGRAM_DIR / label
        out_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing LS-labeled audio: {audio_path} → label={label}")

        try:
            if audio_path.suffix.lower() == ".mp3":
                waveform, sample_rate = torchaudio.load(str(audio_path))
                audio_np = waveform[0].numpy()
            else:
                sample_rate, audio_np = wavfile.read(str(audio_path))
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]
        except Exception as e:
            logger.warning(f"Failed to read {audio_path}: {e}")
            continue

        segments = segment_audio(audio_np, sample_rate, segment_duration, segment_overlap)
        logger.info(f"{len(segments)} segments created for {audio_path}")

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

            logger.info(f"LS: Saved spectrogram image: {image_path}")

            del log_spec, spec, segment
            gc.collect()

    logger.info("LS-driven segmentation asset complete! All outputs in label_studio/spectrogram_images.")


def _save_segment_spectrogram_and_return(path_folder: Path, audio_id: str, seg_index: int, segment: np.ndarray, sample_rate: int, segment_duration: float) -> str:
    """
    Helper: compute spectrogram, save to image path, return str(image_path)
    """
    # normalize
    segment = (segment - np.mean(segment)).astype(np.float32)
    segment /= (np.std(segment) + 1e-8)

    spec = compute_spectrogram(segment, sample_rate)
    log_spec = 10 * np.log10(spec + 1e-10)
    log_spec -= np.max(log_spec)
    log_spec = np.clip(log_spec, -80, 0)

    image_filename = f"{audio_id}_segment_{seg_index}.png"
    image_path = path_folder / image_filename

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

    del log_spec, spec
    gc.collect()

    return str(image_path)


def ls_segment_audios_incremental(audio_path: str, label: str = None, segments: List[dict] = None) -> List[Tuple[str, str]]:
    """
    2 modes:
      - If `segments` is None: behave like previous incremental segmentation:
           segment audio using segment_audio(...) and label each with `label`
           (so label must be provided).
      - If `segments` is provided: segments is a list of dicts with {start, end, label}
           -> produce spectrogram images only for those ranges and use their labels.

    Returns list of tuples: (image_path, label)
    """
    logger = get_dagster_logger()
    LS_SPECTROGRAM_DIR = Path("/app/dagster_output/label_studio/spectrogram_images")
    segment_paths: List[Tuple[str, str]] = []

    settings = load_settings()
    segment_duration = float(settings["segment_duration"])
    # segment_overlap not needed when segments provided but load for completeness
    segment_overlap = float(settings["segment_overlap"])

    # Resolve audio path robustly
    try:
        audio_path = resolve_audio_path(audio_path)
    except FileNotFoundError as e:
        logger.error(f"{e}")
        return []

    out_base = LS_SPECTROGRAM_DIR
    audio_id = audio_path.stem

    # Read full audio
    try:
        if audio_path.suffix.lower() == ".mp3":
            waveform, sample_rate = torchaudio.load(str(audio_path))
            audio_np = waveform[0].numpy()
        else:
            sample_rate, audio_np = wavfile.read(str(audio_path))
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
    except Exception as e:
        logger.error(f"Failed to read {audio_path}: {e}")
        return []

    # Mode A: provided segments (use start/end times)
    if segments:
        logger.info(f"Using provided {len(segments)} segments for audio: {audio_path}")
        # Ensure segments are sorted by start time
        try:
            sorted_segments = sorted(segments, key=lambda s: float(s.get("start", 0)))
        except Exception:
            sorted_segments = segments

        out_folder_map = {}
        for seg_index, seg in enumerate(sorted_segments):
            try:
                start = float(seg["start"])
                end = float(seg["end"])
                seg_label = str(seg.get("label") or seg.get("choice") or seg.get("value") or "").lower()
            except Exception as e:
                logger.warning(f"Skipping malformed segment entry: {seg} ({e})")
                continue

            # create folder per label
            out_folder = out_base / (seg_label or "unknown")
            out_folder.mkdir(parents=True, exist_ok=True)

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            # bounds check
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_np), end_sample)
            if end_sample <= start_sample:
                logger.warning(f"Invalid segment samples for {seg}: start_sample={start_sample}, end_sample={end_sample}")
                continue

            segment_array = audio_np[start_sample:end_sample]
            # Use requested duration for plotting's X extent:
            duration_for_extent = (end - start) if (end - start) > 0 else segment_duration

            image_path = _save_segment_spectrogram_and_return(out_folder, audio_id, seg_index, segment_array, sample_rate, duration_for_extent)
            logger.info(f"LS: Saved spectrogram image for provided segment: {image_path}")
            segment_paths.append((image_path, seg_label))

        logger.info(f"Incremental (provided segments) complete for: {audio_path}")
        return segment_paths

    # Mode B: no provided segments -> segment whole audio using segmentation logic and use single label
    if not label:
        logger.error("No 'label' provided for full-audio segmentation mode.")
        return []

    out_folder = out_base / label.lower()
    out_folder.mkdir(parents=True, exist_ok=True)

    # Segment via existing algorithm
    segments_auto = segment_audio(audio_np, sample_rate, segment_duration, segment_overlap)
    logger.info(f"{len(segments_auto)} segments created for {audio_path}")

    for seg_index, segment in enumerate(segments_auto):
        image_path = _save_segment_spectrogram_and_return(out_folder, audio_id, seg_index, segment, sample_rate, segment_duration)
        logger.info(f"LS: Saved spectrogram image: {image_path}")
        segment_paths.append((image_path, label.lower()))

    logger.info(f"Incremental segmentation complete for: {audio_path}")
    return segment_paths
