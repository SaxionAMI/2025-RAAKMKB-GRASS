#pragma once
#include <vector>

std::vector<std::vector<float>> segment_and_compute_spectrogram(
    const std::vector<float>& audio,
    int sample_rate,
    float window_sec,
    float hop_sec,
    int fft_size = 512,
    int hop_length = 128
);
