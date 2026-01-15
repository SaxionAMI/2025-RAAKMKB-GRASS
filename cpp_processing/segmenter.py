import numpy as np

def segment_audio(audio, sample_rate, window_sec=1.0, hop_sec=0.25):
    """
    Splits audio into overlapping segments.

    Args:
        audio (np.ndarray): 1D audio signal
        sample_rate (int): sampling rate of the audio
        window_sec (float): segment length in seconds
        hop_sec (float): overlap step in seconds

    Returns:
        np.ndarray: [num_segments x window_length] array of audio segments
    """
    window_len = int(window_sec * sample_rate)
    hop_len = int(hop_sec * sample_rate)
    
    segments = []
    for start in range(0, len(audio) - window_len + 1, hop_len):
        segment = audio[start:start + window_len]
        segments.append(segment)

    return np.array(segments)
