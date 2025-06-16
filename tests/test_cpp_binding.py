import numpy as np
from audio_input.audio_loader import load_audio, list_labeled_audio_files
import cpp_processing.audio_cpp as cpp

def test_cpp_spectrogram():
    # Load one sample audio file
    file_path, label = list_labeled_audio_files("audio_input/normalized")[0]
    print(f"📂 Testing file: {file_path} | Label: {label}")

    audio, sr = load_audio(file_path)

    # Call the C++ segment + spectrogram function
    specs = cpp.segment_and_compute_spectrogram(
        audio.tolist(), sr, 1.0, 0.25, 512, 128
    )

    print(f"✅ Segments returned: {len(specs)}")

    if specs:
        segment_len = len(specs[0])
        print(f"✅ Shape of first spectrogram: {segment_len} values")

        # Estimated dimensions
        expected_bins = 512 // 2 + 1  # 257 frequency bins
        # We'll estimate frames from FFT parameters
        approx_frames = (sr * 1.0 - 512) // 128 + 1
        expected_total = expected_bins * approx_frames

        print("🔍 Expected shape estimate:", f"{expected_bins} × ~{int(approx_frames)} ≈ {int(expected_total)}")

        # Show a preview of the data
        print("🔍 First 10 values of first spectrogram:", specs[0][:10])

        if segment_len < expected_total * 0.9:
            print("⚠️ Spectrogram seems smaller than expected — check FFT parameters.")
    else:
        print("❌ No spectrogram segments returned.")

if __name__ == "__main__":
    test_cpp_spectrogram()
