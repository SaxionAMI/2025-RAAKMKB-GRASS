# tests/test_segmented_spectrograms.py
import sys
from types import ModuleType
from pathlib import Path
import importlib.util
import pytest
from unittest.mock import MagicMock


# ---- helpers: stub package, utils, and audio segmentation; load module from file ----
def _stub_utils_in_sysmodules(settings=None):
    """
    Provide stubs so assets.py imports don't pull real deps:
      - grass_pipeline_dagster  (as a package)
      - grass_pipeline_dagster.utils
      - grass_pipeline_dagster.audio_utils.segmentation.segment_audio_files
    """
    # Ensure top-level package exists and acts like a package
    if "grass_pipeline_dagster" not in sys.modules:
        pkg = ModuleType("grass_pipeline_dagster")
        pkg.__path__ = []  # mark as package so submodules can be imported
        sys.modules["grass_pipeline_dagster"] = pkg
    else:
        if not hasattr(sys.modules["grass_pipeline_dagster"], "__path__"):
            sys.modules["grass_pipeline_dagster"].__path__ = []

    # Stub utils
    if settings is None:
        settings = {"input_dir": "audio_input"}
    utils_mod = ModuleType("grass_pipeline_dagster.utils")
    utils_mod.load_settings = lambda: settings
    utils_mod.get_minio_client = lambda: MagicMock()
    sys.modules["grass_pipeline_dagster.utils"] = utils_mod

    # Stub audio_utils package
    if "grass_pipeline_dagster.audio_utils" not in sys.modules:
        au_pkg = ModuleType("grass_pipeline_dagster.audio_utils")
        au_pkg.__path__ = []  # make it package-like
        sys.modules["grass_pipeline_dagster.audio_utils"] = au_pkg
    else:
        if not hasattr(sys.modules["grass_pipeline_dagster.audio_utils"], "__path__"):
            sys.modules["grass_pipeline_dagster.audio_utils"].__path__ = []

    # Stub audio_utils.segmentation with a default segment_audio_files
    seg_mod = ModuleType("grass_pipeline_dagster.audio_utils.segmentation")
    seg_mod.segment_audio_files = lambda files: 0
    sys.modules["grass_pipeline_dagster.audio_utils.segmentation"] = seg_mod


def _load_assets_module():
    """
    Import grass_pipeline_dagster/grass_pipeline_dagster/assets.py directly from file.
    """
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "grass_pipeline_dagster" / "grass_pipeline_dagster" / "assets.py"
    assert mod_path.exists(), f"assets.py not found at {mod_path}"
    spec = importlib.util.spec_from_file_location("assets_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---- 6 fully mocked unit tests for segmented_spectrograms ----

# 1) Happy path: mixed extensions -> forwards files and returns formatted count
def test_segmented_spectrograms_happy_mixed(monkeypatch):
    _stub_utils_in_sysmodules({"input_dir": "audio_input"})
    m = _load_assets_module()

    fake_files = ["audio_input/class1/a.wav", "audio_input/class2/b.mp3"]
    monkeypatch.setattr(m.glob, "glob", lambda pattern, recursive=False: fake_files)

    calls = {"seen": None}

    def fake_segment(files):
        calls["seen"] = files
        return 10

    # Patch the imported function used inside assets.py
    monkeypatch.setattr(m, "segment_audio_files", fake_segment)

    result = m.segmented_spectrograms()
    assert result == "Saved 10 spectrograms"
    assert calls["seen"] == fake_files


# 2) Verifies settings are used to build the glob pattern; avoid raising by returning a dummy .wav
def test_segmented_spectrograms_uses_input_dir_in_glob(monkeypatch):
    _stub_utils_in_sysmodules({"input_dir": "custom_dir"})
    m = _load_assets_module()

    captured = {"pattern": None}

    def fake_glob(pattern, recursive=False):
        captured["pattern"] = pattern
        # return at least one wav so the asset doesn't raise
        return ["custom_dir/dummy.wav"]

    monkeypatch.setattr(m.glob, "glob", fake_glob)
    monkeypatch.setattr(m, "segment_audio_files", lambda files: 0)

    _ = m.segmented_spectrograms()

    assert captured["pattern"] is not None
    # The function makes an absolute path; just ensure "custom_dir" appears in the pattern
    assert "custom_dir" in captured["pattern"].replace("\\", "/")


# 3) Empty list: the asset raises FileNotFoundError (matches real behavior)
def test_segmented_spectrograms_empty_returns_zero(monkeypatch):
    _stub_utils_in_sysmodules({"input_dir": "audio_input"})
    m = _load_assets_module()

    monkeypatch.setattr(m.glob, "glob", lambda pattern, recursive=False: [])
    monkeypatch.setattr(m, "segment_audio_files", lambda files: 0)

    with pytest.raises(FileNotFoundError):
        _ = m.segmented_spectrograms()


# 4) Pass-through: whatever glob returns (wav/mp3 only) is passed verbatim to segment_audio_files
def test_segmented_spectrograms_forwards_discovered_files(monkeypatch):
    _stub_utils_in_sysmodules({"input_dir": "audio_input"})
    m = _load_assets_module()

    files = [
        "audio_input/c1/x.wav",
        "audio_input/c2/y.mp3",
    ]  # only supported extensions so filter doesn't drop any
    monkeypatch.setattr(m.glob, "glob", lambda pattern, recursive=False: files)

    seen = {"files": None}

    def fake_segment(fs):
        seen["files"] = fs
        return len(fs)

    monkeypatch.setattr(m, "segment_audio_files", fake_segment)
    _ = m.segmented_spectrograms()

    assert seen["files"] == files


# 5) Multiple calls are independent and counts are propagated each time
def test_segmented_spectrograms_multiple_calls(monkeypatch):
    _stub_utils_in_sysmodules({"input_dir": "audio_input"})
    m = _load_assets_module()

    # First call
    monkeypatch.setattr(m.glob, "glob", lambda pattern, recursive=False: ["audio_input/a.wav"])
    monkeypatch.setattr(m, "segment_audio_files", lambda files: 1)
    r1 = m.segmented_spectrograms()
    assert r1 == "Saved 1 spectrograms"

    # Second call (different data)
    monkeypatch.setattr(
        m.glob, "glob",
        lambda pattern, recursive=False: ["audio_input/a.wav", "audio_input/b.mp3", "audio_input/c.wav"]
    )
    monkeypatch.setattr(m, "segment_audio_files", lambda files: 3)
    r2 = m.segmented_spectrograms()
    assert r2 == "Saved 3 spectrograms"


# 6) Large count: return string formatting remains correct
def test_segmented_spectrograms_large_count_format(monkeypatch):
    _stub_utils_in_sysmodules({"input_dir": "audio_input"})
    m = _load_assets_module()

    monkeypatch.setattr(m.glob, "glob", lambda pattern, recursive=False: ["audio_input/a.wav"] * 5)
    monkeypatch.setattr(m, "segment_audio_files", lambda files: 123)

    result = m.segmented_spectrograms()
    assert result == "Saved 123 spectrograms"
