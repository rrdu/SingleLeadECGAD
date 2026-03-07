# signal_cleaning.py
from __future__ import annotations

from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

###########################################################################
def nan_ratio(signal):
    """Return fraction of NaN values in a 1D signal."""
    signal = np.asarray(signal)
    if len(signal) == 0:
        return 1.0
    return np.isnan(signal).mean()

###########################################################################
def fill_nans(signal):
    """
    Fill NaNs in a 1D signal using linear interpolation.
    If all values are NaN, return the original signal.
    """
    signal = np.asarray(signal, dtype=np.float32).copy()

    if not np.isnan(signal).any():
        return signal

    n = len(signal)
    idx = np.arange(n)
    valid = ~np.isnan(signal)

    if valid.sum() == 0:
        return signal

    signal[~valid] = np.interp(idx[~valid], idx[valid], signal[valid])
    return signal

###########################################################################
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter to a 1D signal."""
    signal = np.asarray(signal, dtype=np.float32)

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)

###########################################################################
def notch_filter(signal, notch_freq, fs, quality_factor=30):
    """Apply notch filter to remove powerline interference."""
    signal = np.asarray(signal, dtype=np.float32)

    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0=w0, Q=quality_factor)

    return filtfilt(b, a, signal).astype(np.float32)

###########################################################################
def zscore_normalize(signal, eps=1e-8):
    """Apply z-score normalization to a 1D signal."""
    signal = np.asarray(signal, dtype=np.float32)
    mean = np.mean(signal)
    std = np.std(signal)

    if std < eps:
        return signal - mean

    return (signal - mean) / (std + eps)

###########################################################################
def minmax_normalize(signal, eps=1e-8):
    """Apply min-max normalization to a 1D signal."""
    signal = np.asarray(signal, dtype=np.float32)
    min_val = np.min(signal)
    max_val = np.max(signal)
    denom = max_val - min_val

    if denom < eps:
        return signal - min_val

    return (signal - min_val) / (denom + eps)

###########################################################################
def normalize_signal(signal, method="zscore", eps=1e-8):
    """Normalize a 1D signal using configured method."""
    if method == "zscore":
        return zscore_normalize(signal, eps=eps)
    elif method == "minmax":
        return minmax_normalize(signal, eps=eps)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

###########################################################################
def is_valid_signal(signal, min_length, max_nan_ratio):
    """
    Check whether a signal passes basic quality control.
    Returns (is_valid, reason).
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        return False, "not_1d"

    if len(signal) < min_length:
        return False, "too_short"

    if nan_ratio(signal) > max_nan_ratio:
        return False, "too_many_nans"

    if np.all(np.isnan(signal)):
        return False, "all_nan"

    return True, "ok"

###########################################################################
def clean_ecg_signal(
    signal,
    fs,
    lowcut,
    highcut,
    notch_freq,
    filter_order=4,
    quality_factor=30,
    apply_bandpass=True,
    apply_notch=True,
    normalize=True,
    normalization_method="zscore",
    eps=1e-8,
):
    """
    Full ECG cleaning pipeline:
    1. ensure float32
    2. fill NaNs
    3. apply bandpass filter
    4. apply notch filter
    5. normalize
    """
    signal = np.asarray(signal, dtype=np.float32).copy()

    signal = fill_nans(signal)

    if apply_bandpass:
        signal = butter_bandpass_filter(
            signal,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
            order=filter_order,
        )

    if apply_notch:
        signal = notch_filter(
            signal,
            notch_freq=notch_freq,
            fs=fs,
            quality_factor=quality_factor,
        )

    if normalize:
        signal = normalize_signal(
            signal,
            method=normalization_method,
            eps=eps,
        )

    return signal.astype(np.float32)

###########################################################################