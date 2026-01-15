#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "processing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(audio_cpp, m) {
    m.def(
        "segment_and_compute_spectrogram",
        &segment_and_compute_spectrogram,
        py::arg("audio"),
        py::arg("sample_rate"),
        py::arg("window_sec"),
        py::arg("hop_sec"),
        py::arg("fft_size") = 512,
        py::arg("hop_length") = 128,
        "Segment audio and compute log1p(|RFFT|) spectrograms per segment."
    );
}
