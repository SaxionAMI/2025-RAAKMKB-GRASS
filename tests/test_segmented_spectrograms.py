import pytest
from unittest.mock import patch

from grass_pipeline_dagster.assets import segmented_spectrograms


@patch("grass_pipeline_dagster.assets.segment_audio_files")
@patch("grass_pipeline_dagster.assets.glob.glob")
@patch("grass_pipeline_dagster.assets.load_settings")
def test_segmented_spectrograms(mock_load_settings, mock_glob, mock_segment_audio):
    # Mock settings
    mock_load_settings.return_value = {"input_dir": "audio_input"}

    # Simulate audio files
    mock_glob.return_value = [
        "audio_input/class1/audio1.wav",
        "audio_input/class2/audio2.mp3"
    ]

    # Simulate segmentation output
    mock_segment_audio.return_value = 10

    # Run asset function
    result = segmented_spectrograms()

    # Check result
    assert result == "Saved 10 spectrograms"
    mock_segment_audio.assert_called_once_with(mock_glob.return_value)
