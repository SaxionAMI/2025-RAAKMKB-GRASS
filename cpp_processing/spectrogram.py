import numpy as np

def compute_spectrogram(segment, sample_rate, n_fft=512, hop_length=128):
    """
    Computes a simple log-scaled spectrogram from a 1D audio segment.

    Args:
        segment (np.ndarray): 1D audio segment (float32)
        sample_rate (int): audio sample rate
        n_fft (int): FFT window size
        hop_length (int): step size between windows

    Returns:
        np.ndarray: 2D log-scaled spectrogram [freq_bins x time_frames]
    """
    window = np.hamming(n_fft)
    frames = []

    for start in range(0, len(segment) - n_fft + 1, hop_length):
        frame = segment[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        frames.append(spectrum)

    S = np.array(frames).T  # [freq_bins x time_frames]
    S_log = np.log1p(S)     # Log scaling

    return S_log
