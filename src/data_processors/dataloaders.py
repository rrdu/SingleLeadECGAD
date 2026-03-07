#dataloaders.py
import os
import glob
import json
import numpy as np
import pandas as pd

from pathlib import Path
###########################################################################
def list_json_files(root_dir):
    """Recursively list all JSON files under a directory."""
    root_dir = Path(root_dir)
    return sorted(root_dir.rglob("*.json"))

###########################################################################
def list_csv_files(root_dir):
    """Recursively list all CSV files under a directory."""
    root_dir = Path(root_dir)
    return sorted(root_dir.rglob("*.csv"))

###########################################################################
def load_json_file(file_path):
    """Load a JSON file and return its contents as a dict."""
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        return json.load(f)

###########################################################################
def load_csv_file(file_path):
    """Load a CSV file and return a pandas DataFrame."""
    file_path = Path(file_path)
    return pd.read_csv(file_path)
###########################################################################
def get_file_stem(file_path):
    """Return filename without extension."""
    return Path(file_path).stem

###########################################################################
def get_parent_folder_name(file_path):
    """Return immediate parent folder name."""
    return Path(file_path).parent.name

###########################################################################
def infer_source_name(file_path, id_dump_duke_dir, id_raw_data_dir, ood_data_dir):
    """Infer dataset source from file path."""
    file_path = Path(file_path)
    if Path(id_dump_duke_dir) in file_path.parents:
        return "dump_duke"
    elif Path(id_raw_data_dir) in file_path.parents:
        return "raw_data_add_v1"
    elif Path(ood_data_dir) in file_path.parents:
        return "ECG_OOD"
    return "unknown"

###########################################################################
def infer_record_id(file_path, file_data=None):
    """
    Infer a record_id from file contents if available, otherwise from filename.
    """
    if isinstance(file_data, dict) and "record_id" in file_data:
        return str(file_data["record_id"])
    return get_file_stem(file_path)

###########################################################################
def summarize_file_inventory(id_dump_duke_dir, id_raw_data_dir, ood_data_dir):
    """Return a quick summary of file counts in each source."""
    summary = {
        "dump_duke_json": len(list_json_files(id_dump_duke_dir)),
        "raw_data_add_v1_csv": len(list_csv_files(id_raw_data_dir)),
        "ECG_OOD_csv": len(list_csv_files(ood_data_dir)),
    }
    return pd.Series(summary)

###########################################################################