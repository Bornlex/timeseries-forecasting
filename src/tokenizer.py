from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple


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


def find_scaling_limits(scaled_series: np.ndarray, num_bins: int) -> Tuple[float, float]:
    finite_values = scaled_series[np.isfinite(scaled_series)]

    if finite_values.size == 0:
        return -1.0, 1.0

    low_limit = np.percentile(finite_values, 1)
    high_limit = np.percentile(finite_values, 99)

    if low_limit >= high_limit:
        high_limit = low_limit + 1.0

    return low_limit, high_limit


def make_window_pointers(data_list: list, context_length: int) -> List[Tuple[int, int]]:
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


def setup_data(
        dataset_path: str,
        num_series: int,
        context_length: int,
        horizon: int,
        num_bins: int,
        batch_size: int,
        low_limit: float = -1000,
        high_limit: float = 1000,
        seed: int = 42
):
    def is_valid_series(item, min_val, max_val):
        series = np.array(item['target'][0])

        if np.any(np.isnan(series)):
            return False

        if np.any(series < min_val):
            return False

        if np.any(series > max_val):
            return False

        return True

    raw_dataset = load_dataset(
        "json",
        data_files=dataset_path,
        split="train",
        streaming=True,
    ).shuffle(seed=seed).filter(
        lambda x: is_valid_series(x, min_val=low_limit, max_val=high_limit)
    )

    data_list = list(raw_dataset.take(num_series))
    validation_list = list(raw_dataset.shuffle(seed=seed + 1).take(100))
    validation_list = [item['target'][0] for item in validation_list]

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


if __name__ == '__main__':
    train_loader, validation_list = setup_data(
        'dataset.jsonl',
        num_series=100,
        context_length=30,
        horizon=1,
        num_bins=255,
        batch_size=32,
        seed=42
    )

    for x_batch, y_batch in train_loader:
        print("X batch shape:", x_batch.shape)
        print("Y batch shape:", y_batch.shape)
        break
