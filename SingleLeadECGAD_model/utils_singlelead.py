import time
import numpy as np
import torch
import copy
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def time_string():
    iso_time_format = "%Y-%m-%d %X"
    return "[{}]".format(time.strftime(iso_time_format, time.localtime()))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def _safe_minmax_scale(seq):
    """
    Scale a 1D numpy array or tensor to [-1, 1] safely.
    If max == min, return zeros of the same shape.
    """
    seq_min = seq.min()
    seq_max = seq.max()
    denom = seq_max - seq_min

    if isinstance(seq, np.ndarray):
        if denom == 0:
            return np.zeros_like(seq, dtype=np.float32)
        return (2 * (seq - seq_min) / denom - 1).astype(np.float32)
    else:
        if torch.all(denom == 0):
            return torch.zeros_like(seq)
        return 2 * (seq - seq_min) / denom - 1


def normalize(X_train_ori):
    """
    Normalize a batch of ECG signals to [-1, 1] per sample, per channel.

    Expected input shapes:
      - (N, T, C)
      - (N, T)  -> treated as single-channel and expanded internally
    """
    X_train = copy.deepcopy(X_train_ori)

    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train, dtype=np.float32)

    if X_train.ndim == 2:
        X_train = X_train[..., np.newaxis]  # (N, T) -> (N, T, 1)

    num_samples, _, num_channels = X_train.shape

    for i in range(num_samples):
        for c in range(num_channels):
            seq = X_train[i, :, c]
            X_train[i, :, c] = _safe_minmax_scale(seq)

    return X_train.astype(np.float32)


def beat_normalize(X_train_ori):
    """
    Normalize a single ECG segment/beat to [-1, 1] per channel.

    Expected input shapes:
      - (T, C)
      - (T,) -> treated as single-channel
    """
    X_train = copy.deepcopy(X_train_ori)

    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train, dtype=np.float32)

    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]  # (T,) -> (T, 1)

    _, num_channels = X_train.shape

    for c in range(num_channels):
        seq = X_train[:, c]
        X_train[:, c] = _safe_minmax_scale(seq)

    return X_train.astype(np.float32)


def generate_trend(ecg):
    """
    Generate a trend signal for ECG input.

    Expected input:
      ecg shape = (B, T, C)

    Output:
      trend shape = (B, T, C)

    Notes:
    - Works for any number of channels, including single-lead ECG.
    - Works on CPU or GPU automatically by using ecg.device.
    - Works for variable window lengths.
    """
    if ecg.ndim != 3:
        raise ValueError(f"Expected ecg to have shape (B, T, C), got {ecg.shape}")

    device = ecg.device
    dtype = ecg.dtype
    _, signal_len, num_channels = ecg.shape

    # Moving average filter
    avg_kernel_size = 10
    avg_filter = torch.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=avg_kernel_size,
        stride=1,
        padding=0,
        bias=False
    ).to(device=device, dtype=dtype)

    avg_kernel = torch.ones((1, 1, avg_kernel_size), device=device, dtype=dtype) / avg_kernel_size
    avg_filter.weight.data = avg_kernel
    avg_filter.weight.requires_grad = False

    # Difference filter
    dif_filter = torch.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        stride=1,
        padding=0,
        bias=False
    ).to(device=device, dtype=dtype)

    dif_kernel = torch.tensor([-1.0, 1.0], device=device, dtype=dtype).view(1, 1, 2)
    dif_filter.weight.data = dif_kernel
    dif_filter.weight.requires_grad = False

    trend_channels = []

    for c in range(num_channels):
        # ecg[:, :, c] -> (B, T), unsqueeze to (B, 1, T)
        x = ecg[:, :, c].unsqueeze(1)

        # Smooth then differentiate
        x = avg_filter(x)
        x = dif_filter(x)

        # Pad back close to original temporal length
        current_len = x.shape[-1]
        total_pad = signal_len - current_len

        if total_pad < 0:
            # In the unlikely event output is longer, trim it
            x = x[:, :, :signal_len]
        elif total_pad > 0:
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            x = F.pad(x, (pad_left, pad_right), mode="constant", value=0)

        # (B, 1, T) -> (B, T, 1)
        x = x.transpose(1, 2)
        trend_channels.append(x)

    result = torch.cat(trend_channels, dim=-1)  # (B, T, C)

    # Normalize each sample independently to [-1, 1]
    normalized = []
    for b in range(result.shape[0]):
        sample = result[b]  # (T, C)
        sample_min = sample.min()
        sample_max = sample.max()
        denom = sample_max - sample_min

        if torch.all(denom == 0):
            sample = torch.zeros_like(sample)
        else:
            sample = 2 * (sample - sample_min) / denom - 1

        normalized.append(sample)

    result = torch.stack(normalized, dim=0)
    return result