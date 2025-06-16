#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "processing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(audio_cpp, m) {
    m.def("segment_and_compute_spectrogram", &segment_and_compute_spectrogram,
        "Segment audio and compute log spectrogram");
}
