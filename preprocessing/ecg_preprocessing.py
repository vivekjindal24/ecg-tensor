"""
ECG preprocessing utilities: filtering, resampling, normalization, and beat/window segmentation.
These are lightweight, dependency-minimal implementations intended as a starting point.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample


@dataclass
class SegmentationConfig:
    window_seconds: float = 10.0  # fixed-length window segmentation fallback
    fs_target: int = 500          # target sampling frequency
    overlap: float = 0.0          # fraction overlap between windows [0, 1)


def bandpass_filter(x: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.
    Args:
        x: (channels, time) or (time,) array.
        fs: sampling frequency in Hz.
        low, high: passband edges in Hz.
        order: filter order.
    Returns: filtered array with same shape as x.
    """
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)
    b, a = butter(order, [low_n, high_n], btype='band')
    if x.ndim == 1:
        return filtfilt(b, a, x)
    return np.vstack([filtfilt(b, a, ch) for ch in x])


def notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Apply notch filter at specified mains frequency (50/60 Hz)."""
    b, a = iirnotch(w0=freq/(fs/2), Q=q)
    if x.ndim == 1:
        return filtfilt(b, a, x)
    return np.vstack([filtfilt(b, a, ch) for ch in x])


def denoise_signal(x: np.ndarray, fs: float, use_notch: bool = True, mains_freq: float = 50.0) -> np.ndarray:
    """Convenience denoiser: bandpass + optional notch."""
    y = bandpass_filter(x, fs)
    if use_notch:
        y = notch_filter(y, fs, mains_freq)
    return y


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> Tuple[np.ndarray, float]:
    """Resample along time axis to fs_out. Returns (signal, fs_out)."""
    if np.isclose(fs_in, fs_out):
        return x, fs_out
    factor = fs_out / fs_in
    if x.ndim == 1:
        n_out = int(round(x.shape[0] * factor))
        return resample(x, n_out), fs_out
    else:
        n_out = int(round(x.shape[1] * factor))
        y = np.vstack([resample(ch, n_out) for ch in x])
        return y, fs_out


def normalize_signal(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization along given axis."""
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mu) / sd


def segment_beats(x: np.ndarray, fs: float, cfg: Optional[SegmentationConfig] = None) -> List[np.ndarray]:
    """Simple fixed-window segmentation as a fallback.
    For true beat-level segmentation, integrate an R-peak detector (e.g., neurokit2, wfdb).
    Returns a list of windowed segments of shape (channels, time_window).
    """
    if cfg is None:
        cfg = SegmentationConfig()
    win = int(round(cfg.window_seconds * fs))
    step = int(round(win * (1 - cfg.overlap)))
    if x.ndim == 1:
        x = x[None, :]
    segments = []
    for start in range(0, x.shape[1] - win + 1, step):
        segments.append(x[:, start:start+win].copy())
    return segments
