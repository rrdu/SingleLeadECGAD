#metadata_loader.py

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
def build_record_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Create metadata row from a standardized record dict."""
    ecg = np.asarray(record.get("ecg", []))
    return {
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "category": record.get("category"),
        "label": record.get("label"),
        "fs": record.get("fs"),
        "signal_length": int(len(ecg)),
        "file_path": record.get("file_path"),
    }

###########################################################################
def build_window_metadata(record: Dict[str, Any], rpeak_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create metadata row for a windowed record."""
    row = build_record_metadata(record)
    row.update(
        {
            "window_index": record.get("window_index"),
            "window_start": record.get("window_start"),
            "window_end": record.get("window_end"),
            "window_size": record.get("window_size"),
        }
    )

    if rpeak_result is not None:
        row.update(
            {
                "rpeak_success": bool(rpeak_result.get("success", False)),
                "rpeak_reason": rpeak_result.get("reason", "unknown"),
                "num_rpeaks": int(rpeak_result.get("num_rpeaks", 0)),
                "num_valid_rpeaks": int(rpeak_result.get("num_valid_rpeaks", 0)),
            }
        )
    return row

###########################################################################
def records_to_metadata_df(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Convert standardized records into a metadata dataframe."""
    rows = [build_record_metadata(r) for r in records]
    return pd.DataFrame(rows)

###########################################################################
def windows_to_metadata_df(
    windowed_records: Sequence[Dict[str, Any]],
    rpeak_results: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """Convert windowed records into a metadata dataframe."""
    if rpeak_results is None:
        rows = [build_window_metadata(r) for r in windowed_records]
    else:
        rows = [build_window_metadata(r, rp) for r, rp in zip(windowed_records, rpeak_results)]
    return pd.DataFrame(rows)

###########################################################################