# cpp_processing/spectrogram.py
import os, sys
import numpy as np

# ensure we can import the compiled extension that lives in this folder
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from audio_cpp import segment_and_compute_spectrogram as _seg_spec
    _HAS_CPP = True
except Exception:
    _HAS_CPP = False

def compute_spectrogram(segment, sample_rate, n_fft=512, hop_length=128):
    """
    Computes a log-scaled spectrogram [freq_bins x time_frames] for a 1D segment.
    If the C++/FFTW module is available, use it; otherwise fall back to NumPy.
    """
    if _HAS_CPP:
        # Use the C++ fused function with exactly one segment
        window_sec = float(len(segment) / float(sample_rate))
        hop_sec = window_sec  # force a single segment
        mags = _seg_spec(
            segment.astype(np.float32).tolist(),
            int(sample_rate),
            window_sec,
            hop_sec,
            int(n_fft),
            int(hop_length),
        )
        if not mags or not mags[0]:
            return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)

        flat = np.array(mags[0], dtype=np.float32)
        fft_bins = n_fft // 2 + 1
        frames = flat.size // fft_bins
        if frames == 0:
            return np.zeros((fft_bins, 0), dtype=np.float32)

        # C++ already applied log1p to magnitudes; reshape to [freq x time]
        spec2d = flat.reshape(frames, fft_bins).T
        return spec2d

    # -------- NumPy fallback (old behavior) --------
    window = np.hamming(n_fft).astype(np.float32)
    frames = []
    seg = segment.astype(np.float32, copy=False)
    for start in range(0, len(seg) - n_fft + 1, hop_length):
        frame = seg[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        frames.append(spectrum)
    if not frames:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    S = np.stack(frames, axis=0).T  # [freq_bins x time_frames]
    return np.log1p(S).astype(np.float32)
