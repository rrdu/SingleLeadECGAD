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
def compute_stride(window_size: int, stride: int | None = None, stride_ratio: float | None = 0.5) -> int:
    """Return the stride used for window extraction."""
    if stride is not None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        return int(stride)

    if stride_ratio is None:
        raise ValueError("Provide either stride or stride_ratio")
    if not (0 < stride_ratio <= 1):
        raise ValueError("stride_ratio must be in (0, 1]")

    return max(1, int(round(window_size * stride_ratio)))

###########################################################################
def generate_window_ranges(
    signal_length: int,
    window_size: int,
    stride: int | None = None,
    stride_ratio: float | None = 0.5,
    drop_last: bool = True,
) -> List[Tuple[int, int]]:
    """Generate (start, end) index pairs for a 1D signal."""
    if signal_length <= 0:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if signal_length < window_size:
        return []

    stride_val = compute_stride(window_size, stride=stride, stride_ratio=stride_ratio)
    ranges: List[Tuple[int, int]] = []

    start = 0
    while start + window_size <= signal_length:
        end = start + window_size
        ranges.append((start, end))
        start += stride_val

    if not drop_last and ranges:
        last_end = ranges[-1][1]
        if last_end < signal_length:
            ranges.append((signal_length - window_size, signal_length))

    return ranges

###########################################################################
def extract_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int | None = None,
    stride_ratio: float | None = 0.5,
    drop_last: bool = True,
    return_ranges: bool = True,
) -> tuple[np.ndarray, List[Tuple[int, int]]] | np.ndarray:
    """Slice a 1D signal into fixed-length windows."""
    signal = np.asarray(signal, dtype=np.float32).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    ranges = generate_window_ranges(
        signal_length=len(signal),
        window_size=window_size,
        stride=stride,
        stride_ratio=stride_ratio,
        drop_last=drop_last,
    )

    if not ranges:
        empty = np.empty((0, window_size), dtype=np.float32)
        return (empty, []) if return_ranges else empty

    windows = np.stack([signal[s:e] for s, e in ranges]).astype(np.float32)
    return (windows, ranges) if return_ranges else windows

###########################################################################
def window_record(
    record: Dict[str, Any],
    window_size: int,
    stride: int | None = None,
    stride_ratio: float | None = 0.5,
    drop_last: bool = True,
) -> List[Dict[str, Any]]:
    """Expand one standardized record dict into per-window record dicts."""
    ecg = np.asarray(record["ecg"], dtype=np.float32).squeeze()
    windows, ranges = extract_windows(
        ecg,
        window_size=window_size,
        stride=stride,
        stride_ratio=stride_ratio,
        drop_last=drop_last,
        return_ranges=True,
    )

    out: List[Dict[str, Any]] = []
    for idx, ((start, end), window) in enumerate(zip(ranges, windows)):
        new_record = dict(record)
        new_record["ecg"] = window
        new_record["window_index"] = idx
        new_record["window_start"] = int(start)
        new_record["window_end"] = int(end)
        new_record["window_size"] = int(window_size)
        out.append(new_record)
    return out

###########################################################################