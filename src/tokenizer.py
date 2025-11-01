from dataclasses import dataclass
import torch
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass
class TokenizerConfig:
    tokenizer_kwargs: Dict[str, Any]
    context_length: int
    prediction_length: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert self.pad_token_id < self.n_special_tokens and self.eos_token_id < self.n_special_tokens, (
            f"Special token id's must be smaller than {self.n_special_tokens=}"
        )


class MeanScaleUniformBins:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(torch.abs(context) * attention_mask, dim=-1) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )

        token_ids.clamp_(0, self.config.n_tokens - 1)

        token_ids[~attention_mask] = self.config.pad_token_id

        return token_ids, attention_mask, scale

    def _append_eos_token(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
        token_ids = torch.concat((token_ids, eos_tokens), dim=1)
        eos_mask = torch.full((batch_size, 1), fill_value=True)
        attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask

    def context_input_transform(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, attention_mask, scale = self._input_transform(context=context)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(token_ids=token_ids, attention_mask=attention_mask)

        return token_ids, attention_mask, scale

    def label_input_transform(self, label: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length = label.shape[-1]

        assert length == self.config.prediction_length
        token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

        if self.config.use_eos_token:
            token_ids, attention_mask = self._append_eos_token(token_ids=token_ids, attention_mask=attention_mask)

        return token_ids, attention_mask

    def output_transform(self, samples: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed
