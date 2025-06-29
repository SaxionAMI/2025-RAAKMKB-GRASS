from dagster import asset, get_dagster_logger
from pathlib import Path
import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

@asset
def generate_ls_data() -> dict:
    """
    Build train/test CSVs using ONLY LS-segmented spectrograms,
    based on latest LS annotation export.
    Output is always under /app/dagster_output/label_studio/ls_dataset as ls_train.csv and ls_test.csv.
    """
    logger = get_dagster_logger()
    # Path setup (all inside Docker)
    LS_EXPORT_DIR = Path("/data/export")
    LS_SPEC_ROOT = Path("/app/dagster_output/label_studio/spectrogram_images")
    LS_OUT_PARENT = Path("/app/dagster_output/label_studio")
    LS_OUT_DIR = LS_OUT_PARENT / "ls_dataset"
    LS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only consider .json files that are NOT -info.json
    export_files = sorted(
        [f for f in LS_EXPORT_DIR.glob("*.json") if not str(f).endswith("-info.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not export_files:
        logger.error("No LS task export files found in /data/export/ (skipping -info.json files)")
        raise RuntimeError("No LS export file found!")
    export_path = export_files[0]
    logger.info(f"Using LS export: {export_path}")

    # Parse tasks: support both dict and list formats
    with open(export_path, "r") as f:
        ls_data = json.load(f)
    if isinstance(ls_data, dict):
        tasks = ls_data.get("tasks")
        if not tasks:
            for v in ls_data.values():
                if isinstance(v, list):
                    tasks = v
                    break
        if not tasks:
            logger.error("No tasks found in LS export dict.")
            raise RuntimeError("Failed to parse LS export (no tasks)!")
    elif isinstance(ls_data, list):
        tasks = ls_data
    else:
        logger.error("Unrecognized LS export format.")
        raise RuntimeError("Failed to parse LS export!")

    records = []
    for task in tasks:
        try:
            audio_id = (task.get("file_upload") or Path(task["data"]["audio"]).stem).replace(".wav", "")
            label = task["annotations"][0]["result"][0]["value"]["choices"][0].lower()
        except Exception:
            logger.warning(f"Skipped a task due to missing fields: {task}")
            continue

        # Find all segments for this audio, for the given label, in LS-only folder
        seg_glob = LS_SPEC_ROOT / label / f"{audio_id}_segment_*.png"
        segs = glob.glob(str(seg_glob))
        if not segs:
            logger.warning(
                f"Audio '{audio_id}' with label '{label}' has NO LS spectrogram segments in {seg_glob.parent}"
            )
        for seg_path in segs:
            records.append({
                "path": seg_path,
                "label": label,
                "clip_id": audio_id
            })

    if not records:
        raise RuntimeError("No segments matched LS labels. Check LS export and LS spectrogram folders.")

    df = pd.DataFrame(records)
    logger.info(f"LS: Total segments = {len(df)}, Unique clips = {df['clip_id'].nunique()}")

    # Stratified split at clip level
    clip_df = df.groupby("clip_id").first().reset_index()[["clip_id", "label"]]
    train_clip_ids, test_clip_ids = train_test_split(
        clip_df["clip_id"],
        test_size=0.15,
        stratify=clip_df["label"],
        random_state=42
    )
    train_df = df[df["clip_id"].isin(train_clip_ids)].drop(columns=["clip_id"])
    test_df = df[df["clip_id"].isin(test_clip_ids)].drop(columns=["clip_id"])

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Output paths (LS-only)
    train_path = LS_OUT_DIR / "ls_train.csv"
    test_path = LS_OUT_DIR / "ls_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"LS Train set: {len(train_df)} segments from {len(train_clip_ids)} clips → {train_path}")
    logger.info(f"LS Test set: {len(test_df)} segments from {len(test_clip_ids)} clips → {test_path}")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_size": len(train_df),
        "test_size": len(test_df)
    }
