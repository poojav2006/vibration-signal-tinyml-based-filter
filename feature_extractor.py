"""
feature_extractor.py
====================
Lightweight, TinyML-compatible feature extraction engine for vibration windows.

Configuration:
- Window length: N = 128 samples
- Hop size: 64 samples (50% overlap)
- Sampling rate: Fs = 100.0 Hz
- FFT length: NFFT = 128 points
- Frequency resolution: delta_f = 100.0 / 128 = 0.78125 Hz

The 5 exact features extracted per window:
1. Mean:               mu = (1/N) * sum(x[i])
2. Standard Deviation: sigma = sqrt( (1/N) * sum((x[i] - mu)^2) )
3. RMS:                RMS = sqrt( (1/N) * sum(x[i]^2) )
4. Kurtosis:           kurt = (1/N) * sum((x[i] - mu)^4) / (sigma^4 + eps)
5. Dominant Frequency: f_dom = argmax_{k in [1, N/2-1]} |X[k]| * (Fs / N)

DC Handling:
- Bin k=0 (0 Hz) represents the static DC gravity component (~1.0 g) and is excluded
  so the dominant frequency reflects dynamic vibration/disturbance frequencies.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "kurtosis",
    "dominant_freq"
]

FS_DEFAULT = 100.0
WINDOW_SIZE_DEFAULT = 128
HOP_SIZE_DEFAULT = 64


def extract_features_from_window(
    window: np.ndarray,
    fs: float = FS_DEFAULT
) -> np.ndarray:
    """
    Extract exactly the 5 specified features from a 128-sample window.
    
    Parameters:
        window: 1D numpy array of length 128.
        fs: Sampling rate in Hz (default 100 Hz).
        
    Returns:
        1D numpy array of 5 float features: [mean, std, rms, kurtosis, dominant_freq].
    """
    n = len(window)
    if n == 0:
        return np.zeros(5, dtype=np.float32)
        
    # 1. Mean
    mean_val = float(np.mean(window))
    
    # 2. Standard deviation (using population std N degrees of freedom for TinyML consistency)
    std_val = float(np.std(window, ddof=0))
    
    # 3. Root Mean Square (RMS)
    rms_val = float(np.sqrt(np.mean(window ** 2)))
    
    # 4. Kurtosis (4th statistical moment for shock / impulse detection)
    kurt_val = float(np.mean((window - mean_val) ** 4) / (std_val ** 4 + 1e-9))
    
    # 5. Dominant frequency from 128-point FFT
    # Positive frequencies: bins 1 to (N/2 - 1) (bins 1 to 63 for N=128)
    # Bin 0 is DC and explicitly excluded.
    fft_vals = np.fft.rfft(window, n=n)
    magnitudes = np.abs(fft_vals)
    
    if len(magnitudes) > 1:
        # Exclude DC bin 0, search in bins 1 through len(magnitudes)-1
        k_peak = int(np.argmax(magnitudes[1:])) + 1
        freq_resolution = fs / n
        dom_freq = float(k_peak * freq_resolution)
    else:
        dom_freq = 0.0
        
    return np.array([mean_val, std_val, rms_val, kurt_val, dom_freq], dtype=np.float32)


def extract_fft_spectrum(
    window: np.ndarray,
    fs: float = FS_DEFAULT
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Compute positive frequency spectrum for dashboard FFT visualization.
    
    Returns:
        (freqs, magnitudes, dominant_freq, dominant_bin)
    """
    n = len(window)
    fft_vals = np.fft.rfft(window, n=n)
    magnitudes = (2.0 / n) * np.abs(fft_vals)
    magnitudes[0] /= 2.0  # DC normalization
    
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    
    # Exclude DC bin for dominant frequency determination
    if len(magnitudes) > 1:
        k_peak = int(np.argmax(magnitudes[1:])) + 1
        dom_freq = float(freqs[k_peak])
    else:
        k_peak = 0
        dom_freq = 0.0
        
    return freqs, magnitudes, dom_freq, k_peak


def slice_into_windows(
    signal_100hz: np.ndarray,
    window_size: int = WINDOW_SIZE_DEFAULT,
    hop_size: int = HOP_SIZE_DEFAULT
) -> List[np.ndarray]:
    """
    Slice a continuous signal into overlapping windows.
    Default: 128 samples per window, 64 samples hop (50% overlap).
    """
    num_samples = len(signal_100hz)
    windows = []
    
    for start in range(0, num_samples - window_size + 1, hop_size):
        end = start + window_size
        windows.append(signal_100hz[start:end])
        
    return windows


def extract_features_dataset(
    windows: List[np.ndarray],
    fs: float = FS_DEFAULT
) -> np.ndarray:
    """
    Extract 5-feature matrix from a list of windows.
    Returns array of shape (len(windows), 5).
    """
    matrix = np.zeros((len(windows), 5), dtype=np.float32)
    for i, w in enumerate(windows):
        matrix[i] = extract_features_from_window(w, fs=fs)
    return matrix


if __name__ == '__main__':
    print("--- Testing Feature Extractor ---")
    # Synthetic test signal: 10 Hz sinusoid with DC offset of 1.0 g
    t = np.arange(128) / 100.0
    synth = 1.0 + 0.3 * np.sin(2 * np.pi * 10.0 * t)
    feats = extract_features_from_window(synth, fs=100.0)
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:15s}: {val:.4f}")
    assert abs(feats[0] - 1.0) < 0.05, "Mean error"
    assert abs(feats[4] - 10.0) < 1.0, f"Dominant freq error: {feats[4]}"
    print("Feature extractor tests passed!")
