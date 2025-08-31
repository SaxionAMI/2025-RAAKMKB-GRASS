// cpp_processing/processing.hpp
#pragma once
#include <vector>

/**
 * Segment the full audio (1D float) and compute spectrogram magnitudes for each segment.
 *
 * - Segmentation:
 *     window_sec -> segment length (seconds)
 *     hop_sec    -> step between segment starts (seconds)
 *
 * - STFT:
 *     fft_size   -> FFT window size (samples)
 *     hop_length -> STFT hop (samples)
 *
 * Returns:
 *   vector of segments; each segment is a flattened array containing
 *   log1p(|RFFT|) values laid out frame-major then concatenated.
 *   Reshape each to [num_frames, fft_bins] and transpose if needed.
 */
std::vector<std::vector<float>> segment_and_compute_spectrogram(
    const std::vector<float>& audio,
    int sample_rate,
    float window_sec,
    float hop_sec,
    int fft_size = 512,
    int hop_length = 128
);
