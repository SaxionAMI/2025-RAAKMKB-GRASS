#include "processing.hpp"
#include <cmath>
#include <fftw3.h>
#include <vector>
#include <iostream>
#include <algorithm>

std::vector<std::vector<float>> segment_and_compute_spectrogram(
    const std::vector<float>& audio,
    int sample_rate,
    float window_sec,
    float hop_sec,
    int fft_size,
    int hop_length
) {
    std::cout << "C++ MODULE INVOKED! (debug trace)" << std::endl;

    int window_samples = static_cast<int>(window_sec * sample_rate);   // e.g. 44100
    int hop_samples = static_cast<int>(hop_sec * sample_rate);         // e.g. 11025
    int num_segments = (audio.size() - window_samples) / hop_samples + 1;

    std::cout << " Total audio samples: " << audio.size() << std::endl;
    std::cout << " Window samples: " << window_samples << " | Hop samples: " << hop_samples << std::endl;
    std::cout << " Calculated number of segments: " << num_segments << std::endl;

    std::vector<std::vector<float>> all_specs;

    // FFT setup
    int fft_bins = fft_size / 2 + 1;
    float* in_frame = (float*)fftwf_malloc(sizeof(float) * fft_size);
    fftwf_complex* out_frame = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_bins);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(fft_size, in_frame, out_frame, FFTW_MEASURE);

    for (int seg = 0; seg < num_segments; ++seg) {
        int seg_start = seg * hop_samples;
        int seg_end = std::min(seg_start + window_samples, static_cast<int>(audio.size()));

        std::vector<float> segment(window_samples, 0.0f);
        std::copy(audio.begin() + seg_start, audio.begin() + seg_end, segment.begin());

        std::vector<float> magnitudes;

        int frame_count = 0;
        for (int frame_start = 0; frame_start + fft_size <= segment.size(); frame_start += hop_length) {
            for (int i = 0; i < fft_size; ++i) {
                in_frame[i] = segment[frame_start + i];
            }

            fftwf_execute(plan);

            for (int k = 0; k < fft_bins; ++k) {
                float re = out_frame[k][0];
                float im = out_frame[k][1];
                float mag = std::sqrt(re * re + im * im);
                magnitudes.push_back(std::log1p(mag));
            }
            ++frame_count;
        }

        std::cout << " Segment " << seg
                  << ": size=" << segment.size()
                  << ", FFT frames=" << frame_count
                  << ", total values=" << magnitudes.size() << std::endl;

        all_specs.push_back(magnitudes);
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in_frame);
    fftwf_free(out_frame);

    return all_specs;
}
