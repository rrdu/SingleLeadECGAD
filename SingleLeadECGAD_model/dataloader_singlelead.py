import numpy as np
import heartpy as hp
import random
import pandas as pd
import torch


class DataSet:
    def __init__(self, x_path, meta_path=None, attr_dim=0, num_classes=116,
                 left_margin=140, right_margin=340):
        self.X = np.load(x_path)
        self.meta = pd.read_csv(meta_path) if meta_path is not None else None
        self.attr_dim = attr_dim
        self.num_classes = num_classes
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.beat_len = left_margin + right_margin

    def checkR(self, ecg):
        try:
            working_data, measures = hp.process(ecg, 500.0)
            peak_list = working_data["peaklist"]
            return peak_list
        except Exception:
            return []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ecg_instance = self.X[index]

        if not isinstance(ecg_instance, np.ndarray):
            ecg_instance = np.array(ecg_instance)

        if ecg_instance.ndim == 1:
            ecg_instance = ecg_instance[:, np.newaxis]

        if ecg_instance.ndim == 2 and ecg_instance.shape[0] == 1 and ecg_instance.shape[1] > 1:
            ecg_instance = ecg_instance.T

        ecg_instance = ecg_instance.astype(np.float32)
        signal_len = ecg_instance.shape[0]

        r_index_list = self.checkR(ecg_instance[:, 0])
        valid_peaks = [
            p for p in r_index_list
            if p >= self.left_margin and p + self.right_margin <= signal_len
        ]

        if len(valid_peaks) == 0:
            r_idx = signal_len // 2
        else:
            r_idx = random.choice(valid_peaks)

        if signal_len < self.beat_len:
            pad_amount = self.beat_len - signal_len
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            beat = np.pad(ecg_instance, ((pad_left, pad_right), (0, 0)), mode="edge")
        else:
            start = r_idx - self.left_margin
            end = r_idx + self.right_margin

            if start < 0 or end > signal_len:
                start = max(0, signal_len // 2 - self.left_margin)
                end = start + self.beat_len

            beat = ecg_instance[start:end, :]

        beat = beat.astype(np.float32)

        if self.meta is not None and "attribute" in self.meta.columns:
            attribute = self.meta.iloc[index]["attribute"]
            attribute = torch.tensor(np.array(eval(attribute), dtype=np.float32))
        else:
            attribute = torch.zeros(self.attr_dim, dtype=torch.float32)

        if self.meta is not None and "target" in self.meta.columns:
            target = self.meta.iloc[index]["target"]
            target = torch.tensor(np.array(eval(target), dtype=np.float32))
        else:
            target = torch.zeros(self.num_classes, dtype=torch.float32)

        beat = torch.tensor(beat, dtype=torch.float32)               # local_ecg
        ecg_instance = torch.tensor(ecg_instance, dtype=torch.float32)  # global_ecg

        return beat, ecg_instance, attribute, target