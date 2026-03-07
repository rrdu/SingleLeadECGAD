#ecg_extraction.py
import os
import glob
import json
import numpy as np
import pandas as pd

from pathlib import Path

from .dataloaders import get_parent_folder_name, infer_record_id

SAMPLING_RATE = 1000
###########################################################################
def extract_ecg_from_dump_duke(json_data):
    """
    Extract ECG signal from a dump_duke JSON record.
    Returns a 1D numpy array.
    """
    if "raw_data" not in json_data or "ecg" not in json_data["raw_data"]:
        raise KeyError("Expected json_data['raw_data']['ecg'] in dump_duke record.")

    ecg = np.asarray(json_data["raw_data"]["ecg"], dtype=np.float32).squeeze()

    if ecg.ndim != 1:
        raise ValueError(f"Expected 1D ECG signal, got shape {ecg.shape}")

    return ecg

###########################################################################
def extract_ecg_from_csv(df):
    """
    Extract ECG signal from CSV.
    Priority:
    1. Column named 'ECG'
    2. Otherwise first column
    """
    
    if "ECG" in df.columns:
        ecg = df["ECG"]
    else:
        ecg = df.iloc[:, 0]  # first column

    ecg = pd.to_numeric(ecg, errors="coerce").to_numpy(dtype=np.float32).squeeze()

    if ecg.ndim != 1:
        raise ValueError(f"Expected 1D ECG signal, got shape {ecg.shape}")

    return ecg

###########################################################################
def extract_pcg_from_csv(df):
    """
    Extract PCG signal if available.
    """
    if "PCG" not in df.columns:
        return None

    pcg = pd.to_numeric(df["PCG"], errors="coerce").to_numpy(dtype=np.float32).squeeze()

    if pcg.ndim != 1:
        raise ValueError(f"Expected 1D PCG signal, got shape {pcg.shape}")

    return pcg

###########################################################################
def ensure_1d_signal(signal, signal_name="signal"):
    """
    Ensure signal is a 1D numpy array.
    """
    signal = np.asarray(signal).squeeze()

    if signal.ndim != 1:
        raise ValueError(f"Expected {signal_name} to be 1D, got shape {signal.shape}")

    return signal.astype(np.float32)

###########################################################################
def extract_category_from_dump_duke(file_path, json_data=None):
    """
    Extract category for dump_duke from JSON if present, otherwise from parent folder.
    """
    if isinstance(json_data, dict) and "category" in json_data:
        return str(json_data["category"])

    return get_parent_folder_name(file_path)

###########################################################################
def build_record_dict_from_dump_duke(file_path, json_data):
    """
    Standardize one dump_duke JSON record into a common dictionary format.
    """
    ecg = extract_ecg_from_dump_duke(json_data)

    return {
        "record_id": infer_record_id(file_path, json_data),
        "source": "dump_duke",
        "category": extract_category_from_dump_duke(file_path, json_data),
        "label": "ID",
        "ecg": ecg,
        "pcg": None,
        "fs": SAMPLING_RATE,
        "file_path": str(file_path),
    }

###########################################################################
def build_record_dict_from_csv(file_path, df, source_name, label):
    """
    Standardize one CSV-based record into a common dictionary format.
    """
    ecg = extract_ecg_from_csv(df)
    pcg = extract_pcg_from_csv(df)

    return {
        "record_id": infer_record_id(file_path),
        "source": source_name,
        "category": source_name if label == "OOD" else "ID",
        "label": label,
        "ecg": ecg,
        "pcg": pcg,
        "fs": SAMPLING_RATE,
        "file_path": str(file_path),
    }

###########################################################################