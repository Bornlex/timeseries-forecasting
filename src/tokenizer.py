from dataclasses import dataclass
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple


@dataclass
class TokenizerConfig:
    n_tokens: int
    n_special_tokens: int
    context_length: int
    prediction_length: int
    pad_token_id: int


class Tokenizer:
    def __init__(self, low_limit: float, high_limit: float, config: TokenizerConfig) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def _input_transform(
        self, context: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = context.to(dtype=torch.float32)

        if scale is None:
            scale = torch.nanmean(torch.abs(context), dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                right=True,
            )
            + self.config.n_special_tokens
        )

        token_ids.clamp_(0, self.config.n_tokens - 1)

        return token_ids, scale

    def context_input_transform(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, scale = self._input_transform(context=context)

        return token_ids, scale

    def label_input_transform(self, label: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        length = label.shape[-1]

        assert length == self.config.prediction_length
        token_ids, _ = self._input_transform(context=label, scale=scale)

        return token_ids

    def output_transform(self, samples: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )

        return self.centers[indices] * scale_unsqueezed


class ChronosDataset(Dataset):
    def __init__(
            self,
            data_list: list,
            window_pointers: list,
            tokenizer: Tokenizer,
    ):
        self.data_list = data_list
        self.window_pointers = window_pointers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.window_pointers)

    def __getitem__(self, idx):
        series_idx, start_idx = self.window_pointers[idx]
        series = self.data_list[series_idx]['target'][0]
        series_as_tensor = torch.tensor(series, dtype=torch.float32)

        tokens, scale = self.tokenizer.context_input_transform(series_as_tensor)
        x_tensor = tokens[:-1]
        y_tensor = tokens[1:]

        return x_tensor, y_tensor, scale


def make_window_pointers(data_list: list, context_length: int) -> List[Tuple[int, int]]:
    pointers = []

    for series_idx, item in enumerate(data_list):
        series_len = len(item['target'][0])

        for i in range(series_len - context_length):
            pointers.append((series_idx, i))

    return pointers


def setup_data(
        dataset_path: str,
        num_series: int,
        context_length: int,
        horizon: int,
        num_bins: int,
        batch_size: int,
        low_limit: float = -15,
        high_limit: float = 15,
        seed: int = 42
):
    def is_valid_series(item):
        series = np.array(item['target'][0])

        if np.any(np.isnan(series)):
            return False

        if not (np.all((series >= -50) & (series <= 50))):
            return False

        return True

    raw_dataset = load_dataset(
        "json",
        data_files=dataset_path,
        split="train",
        streaming=True,
    ).shuffle(seed=seed).filter(
        lambda x: is_valid_series(x)
    )

    tokenizer_config = TokenizerConfig(
        n_tokens=num_bins + 1,
        n_special_tokens=1,
        context_length=context_length,
        prediction_length=horizon,
        pad_token_id=num_bins
    )
    tokenizer = Tokenizer(
        low_limit=low_limit,
        high_limit=high_limit,
        config=tokenizer_config
    )

    data_list = list(raw_dataset.take(num_series))
    validation_list = list(raw_dataset.shuffle(seed=seed + 1).take(100))
    validation_list = [item['target'][0] for item in validation_list]

    window_pointers = make_window_pointers(data_list, context_length)

    dataset = ChronosDataset(
        data_list,
        window_pointers,
        tokenizer
    )

    train_loader = DataLoader(dataset, batch_size=batch_size)

    return train_loader, validation_list, tokenizer


if __name__ == '__main__':
    train_loader, validation_list, tokenizer = setup_data(
        'dataset.jsonl',
        num_series=100,
        context_length=256,
        horizon=1,
        num_bins=4096,
        batch_size=32,
        seed=42
    )

    series = torch.from_numpy(np.array(validation_list[0]))
    tokens, scale = tokenizer.context_input_transform(series)

    for x_batch, y_batch, scale in train_loader:
        print("X batch shape :", x_batch.shape)
        print("Y batch shape :", y_batch.shape)
        print("Scale :", scale)
