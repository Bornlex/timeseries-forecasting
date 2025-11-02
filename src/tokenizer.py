from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple


def scale_and_quantize(
        time_series: np.ndarray,
        low_limit: float,
        high_limit: float,
        num_bins: int,
        pad_token_id: int,
        scale: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    epsilon = 1e-8

    nan_mask = np.isnan(time_series)

    if scale is None:
        scale = np.nanmean(np.abs(time_series)) + epsilon
        if np.isnan(scale):
            scale = 1.0 + epsilon

    scaled_series = time_series / scale
    scaled_series_filled = np.nan_to_num(scaled_series, nan=0.0)

    if low_limit >= high_limit:
        high_limit = low_limit + epsilon

    bin_width = (high_limit - low_limit) / num_bins

    clipped_series = np.clip(scaled_series_filled, low_limit, high_limit)
    token_ids = np.floor((clipped_series - low_limit) / bin_width).astype(np.int64)
    token_ids = np.clip(token_ids, 0, num_bins - 1)

    token_ids[nan_mask] = pad_token_id

    return token_ids, scale


def find_scaling_limits(data_list: list, context_length: int, horizon: int = 1):
    global_min = np.inf
    global_max = -np.inf
    epsilon = 1e-8

    for item in data_list:
        series = np.array(item['target'][0]).astype(np.float32)

        if len(series) < context_length + horizon:
            continue

        for i in range(len(series) - context_length - horizon + 1):
            x = series[i : i + context_length]
            y = series[i + context_length + horizon - 1]

            scale = np.mean(np.abs(x)) + epsilon

            x_scaled = x / scale
            y_scaled = y / scale

            current_min = min(np.min(x_scaled), y_scaled)
            current_max = max(np.max(x_scaled), y_scaled)

            if current_min < global_min: global_min = current_min
            if current_max > global_max: global_max = current_max

    return global_min, global_max


class ChronosDataset(Dataset):
    def __init__(
            self,
            data_list: list,
            window_pointers: list,
            context_length: int,
            horizon: int,
            low_limit: float,
            high_limit: float,
            num_bins: int,
            pad_token_id: int
    ):
        self.data_list = data_list
        self.window_pointers = window_pointers
        self.context_length = context_length
        self.horizon = horizon
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.num_bins = num_bins
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.window_pointers)

    def __getitem__(self, idx):
        series_idx, start_idx = self.window_pointers[idx]
        series = self.data_list[series_idx]['target'][0]

        x_tokens, y_tokens, _ = tokenized_window_from_series(
            series,
            start_idx,
            self.context_length,
            self.low_limit,
            self.high_limit,
            self.num_bins,
            self.pad_token_id
        )

        x_tensor = torch.from_numpy(x_tokens).long()
        y_tensor = torch.from_numpy(y_tokens).long()

        return x_tensor, y_tensor


def make_window_pointers(data_list: list, context_length: int) -> List[Tuple[int, int]]:
    """
    Build sliding-window pointers for a dataset of time series.

    Each returned tuple is (series_index, start_index) where `start_index` is the
    beginning of a window that contains `context_length` context steps plus one
    target step (window length = context_length + 1). For a series of length L,
    valid start indices are 0..(L - context_length - 1).

    Args:
        data_list (list): List of items where each item contains a time series at
            `item['target'][0]`.
        context_length (int): Number of context steps in each window.

    Returns:
        List[Tuple[int, int]]: List of (series_index, start_index) pointers.
    """
    pointers = []

    for series_idx, item in enumerate(data_list):
        series_len = len(item['target'][0])

        for i in range(series_len - context_length):
            pointers.append((series_idx, i))

    return pointers


def tokenized_window_from_series(
        series: List[float],
        start_idx: int,
        context_length: int,
        low_limit: float,
        high_limit: float,
        num_bins: int,
        pad_token_id: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return (x_tokens, y_tokens, scale) as numpy arrays for the window starting at start_idx.
    """
    window_end = start_idx + context_length + 1
    full_window_np = np.array(series[start_idx : window_end]).astype(np.float32)

    scale_context_np = full_window_np[:-1]

    epsilon = 1e-8
    scale = np.nanmean(np.abs(scale_context_np)) + epsilon

    if np.isnan(scale):
        scale = 1.0 + epsilon

    all_tokens, used_scale = scale_and_quantize(
        full_window_np,
        low_limit,
        high_limit,
        num_bins,
        pad_token_id,
        scale=scale
    )

    x_tokens = all_tokens[:-1]
    y_tokens = all_tokens[1:]

    return x_tokens, y_tokens, used_scale


def setup_data(num_series: int, context_length: int, horizon: int, num_bins: int, batch_size: int):
    raw_dataset = load_dataset(
        "theforecastingcompany/GiftEvalPretrain",
        split="train",
        streaming=True
    )

    data_list = list(raw_dataset.take(num_series))
    validation_list = list(raw_dataset.skip(num_series).take(100))
    validation_list = [item['target'][0] for item in validation_list]

    low_limit, high_limit = -15.0, 15.0
    pad_token_id = num_bins

    window_pointers = make_window_pointers(data_list, context_length)

    dataset = ChronosDataset(
        data_list,
        window_pointers,
        context_length,
        horizon,
        low_limit,
        high_limit,
        num_bins,
        pad_token_id
    )

    train_loader = DataLoader(dataset, batch_size=batch_size)

    return train_loader, validation_list
