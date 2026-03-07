#rpeak.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import heartpy as hp
except Exception:
    hp = None

###########################################################################
def detect_rpeaks_heartpy(signal: np.ndarray, fs: int) -> Dict[str, Any]:
    """Detect R-peaks using HeartPy."""
    if hp is None:
        return {
            "success": False,
            "rpeaks": np.array([], dtype=int),
            "reason": "heartpy_not_installed",
        }

    signal = np.asarray(signal, dtype=np.float32).squeeze()
    if signal.ndim != 1:
        return {
            "success": False,
            "rpeaks": np.array([], dtype=int),
            "reason": "signal_not_1d",
        }

    try:
        working_data, _ = hp.process(signal, sample_rate=fs)
        peaks = np.asarray(working_data.get("peaklist", []), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(signal))]
        return {
            "success": True,
            "rpeaks": peaks,
            "reason": "ok",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "rpeaks": np.array([], dtype=int),
            "reason": f"heartpy_failed: {exc}",
        }

###########################################################################
def filter_rpeaks_near_edges(rpeaks: Sequence[int], signal_length: int, pre_samples: int, post_samples: int) -> np.ndarray:
    """Keep only peaks that can support a full beat-centered segment."""
    rpeaks = np.asarray(rpeaks, dtype=int)
    keep = (rpeaks - pre_samples >= 0) & (rpeaks + post_samples <= signal_length)
    return rpeaks[keep]

###########################################################################
def extract_beat_segments(
    signal: np.ndarray,
    rpeaks: Sequence[int],
    pre_samples: int,
    post_samples: int,
) -> np.ndarray:
    """Extract beat-centered snippets around each detected R-peak."""
    signal = np.asarray(signal, dtype=np.float32).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    valid_peaks = filter_rpeaks_near_edges(
        rpeaks=rpeaks,
        signal_length=len(signal),
        pre_samples=pre_samples,
        post_samples=post_samples,
    )

    if len(valid_peaks) == 0:
        return np.empty((0, pre_samples + post_samples), dtype=np.float32)

    beats = [signal[p - pre_samples : p + post_samples] for p in valid_peaks]
    return np.stack(beats).astype(np.float32)

###########################################################################
def detect_and_extract_beats(
    signal: np.ndarray,
    fs: int,
    pre_samples: int,
    post_samples: int,
) -> Dict[str, Any]:
    """Run R-peak detection and extract valid beat segments."""
    result = detect_rpeaks_heartpy(signal, fs=fs)
    peaks = result["rpeaks"]
    valid_peaks = filter_rpeaks_near_edges(peaks, len(signal), pre_samples, post_samples)
    beats = extract_beat_segments(signal, valid_peaks, pre_samples, post_samples)

    result.update(
        {
            "valid_rpeaks": valid_peaks,
            "beats": beats,
            "num_rpeaks": int(len(peaks)),
            "num_valid_rpeaks": int(len(valid_peaks)),
        }
    )
    return result

###########################################################################