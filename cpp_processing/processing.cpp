// cpp_processing/processing.cpp
#include "processing.hpp"
#include <cmath>
#include <fftw3.h>
#include <vector>
#include <algorithm>

// NOTE:
// - No M_PI dependency.
// - Hamming window uses a literal 2π constant to avoid math defines.

std::vector<std::vector<float>> segment_and_compute_spectrogram(
    const std::vector<float>& audio,
    int sample_rate,
    float window_sec,
    float hop_sec,
    int fft_size,
    int hop_length
) {
    const int window_samples = static_cast<int>(window_sec * sample_rate);
    const int hop_samples    = static_cast<int>(hop_sec  * sample_rate);

    if (window_samples <= 0 || hop_samples <= 0 || static_cast<int>(audio.size()) < window_samples) {
        return {};
    }

    const int num_segments = (static_cast<int>(audio.size()) - window_samples) / hop_samples + 1;

    std::vector<std::vector<float>> all_specs;
    all_specs.reserve(std::max(0, num_segments));

    // ---- FFT setup ----
    const int fft_bins = fft_size / 2 + 1;
    float* in_frame = static_cast<float*>(fftwf_malloc(sizeof(float) * fft_size));
    fftwf_complex* out_frame = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * fft_bins));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(fft_size, in_frame, out_frame, FFTW_MEASURE);

    // ---- Precompute Hamming window (NumPy/SciPy-compatible) ----
    // w[n] = 0.54 - 0.46 * cos(2π n / (N-1))
    const float two_pi = 6.28318530717958647692f;
    std::vector<float> hamming(fft_size);
    for (int n = 0; n < fft_size; ++n) {
        hamming[n] = 0.54f - 0.46f * std::cos(two_pi * static_cast<float>(n) / static_cast<float>(fft_size - 1));
    }

    // Estimate frames per segment to reserve capacity
    const int frames_per_segment = (window_samples - fft_size >= 0)
        ? ((window_samples - fft_size) / hop_length + 1)
        : 0;

    for (int seg = 0; seg < num_segments; ++seg) {
        const int seg_start = seg * hop_samples;
        const int seg_end   = std::min(seg_start + window_samples, static_cast<int>(audio.size()));

        // Copy segment (zero-padded) into local buffer of size window_samples
        std::vector<float> segment(window_samples, 0.0f);
        std::copy(audio.begin() + seg_start, audio.begin() + seg_end, segment.begin());

        std::vector<float> magnitudes;
        magnitudes.reserve(std::max(0, frames_per_segment) * fft_bins);

        for (int frame_start = 0; frame_start + fft_size <= window_samples; frame_start += hop_length) {
            // Apply Hamming and load into FFT input
            for (int i = 0; i < fft_size; ++i) {
                in_frame[i] = segment[frame_start + i] * hamming[i];
            }

            // Execute FFT
            fftwf_execute(plan);

            // Store log1p(|RFFT|) magnitudes
            for (int k = 0; k < fft_bins; ++k) {
                const float re  = out_frame[k][0];
                const float im  = out_frame[k][1];
                const float mag = std::sqrt(re * re + im * im);
                magnitudes.push_back(std::log1p(mag)); // returns double; implicitly casts to float
            }
        }

        all_specs.push_back(std::move(magnitudes));
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in_frame);
    fftwf_free(out_frame);

    return all_specs;
}
