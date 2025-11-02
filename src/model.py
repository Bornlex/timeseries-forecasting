import numpy as np
import torch
from torch import nn
from typing import Callable

from src.attention import MultiQueryAttention
from src.ffn import FFN
from src.config import ModelConfig
from src.normalization import LayerNorm
from src.positional import PositionalEmbedding


class ForecastingBlock(nn.Module):
    def __init__(self, embedding_size: int, n_head: int, hidden_size: int, dropout: float):
        super().__init__()

        self.__attention = MultiQueryAttention(embedding_size, n_head)
        self.__norm1 = LayerNorm(embedding_size)
        self.__fc = FFN(embedding_size, hidden_size, embedding_size, dropout)
        self.__norm2 = LayerNorm(embedding_size)

    def forward(self, x, kv_cache: dict | None = None):
        x = self.__norm1(x)
        dx1, kv_cache = self.__attention(x, kv_cache=kv_cache)
        x = self.__norm2(x + dx1)
        dx2 = self.__fc(x)
        x = x + dx2

        return x, kv_cache


class ForecastingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.device = 'cpu'

        self.__positional_embedding = PositionalEmbedding(config.n_embd)
        self.__embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.__blocks = nn.ModuleList([
            ForecastingBlock(config.n_embd, config.n_head, config.ffn_hidden_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.__fc = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, kv_cache: list | None = None, start_pos: int = 0):
        x = self.__embedding(x)
        positional_x = self.__positional_embedding(x, start_pos=start_pos)
        x = x + positional_x

        for i, block in enumerate(self.__blocks):
            current_kv_cache = kv_cache[i] if kv_cache else None
            x, current_kv_cache = block(x, kv_cache=current_kv_cache)

            if kv_cache is not None:
                kv_cache[i] = current_kv_cache

        x = self.__fc(x)

        return x, kv_cache

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)

    def save_weights(self, weights_path: str = None):
        torch.save(self.state_dict(), weights_path)
