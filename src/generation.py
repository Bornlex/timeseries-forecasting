import numpy as np
import torch
from typing import List

from src.tokenizer import scale_and_quantize


def tokens_to_values(
    token_ids: np.ndarray,
    low_limit: float,
    high_limit: float,
    num_bins: int,
    pad_token_id: int,
    scale: float
) -> np.ndarray:
    bin_width = (high_limit - low_limit) / float(num_bins)
    values = np.full(token_ids.shape, np.nan, dtype=np.float32)
    mask = (token_ids >= 0) & (token_ids < num_bins)

    if np.any(mask):
        centers = low_limit + (token_ids[mask].astype(np.float32) + 0.5) * bin_width
        values[mask] = centers * scale

    return values


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0):
    """
    Simple top-k/top-p filtering for logits (in-place filtering).
    """
    if top_k > 0:
        kth_val = torch.topk(logits, top_k).values.min()
        logits[logits < kth_val] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = cumulative > top_p
        # keep at least one
        cutoff[..., 0] = False
        sorted_logits[cutoff] = -float("Inf")

        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    return logits


def generate_from_series(
    series: List[float],
    model,
    context_length: int,
    low_limit: float,
    high_limit: float,
    num_bins: int,
    pad_token_id: int,
    max_tokens: int = 256,
    mode: str = "fixed_scale",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> List[float]:
    """
    Generate new values from an initial `series`.

    - If mode == "fixed_scale": compute scale once from the last `context_length` real values
      and keep it fixed for all steps (no retokenization).
    - If mode == "sliding_scale": after each generated value, recompute scale from the
      newest context (including generated values) and retokenize the context before the next step.

    Returns the original series with `max_tokens` appended decoded values (as floats).
    """
    model.eval()

    raw = list(series)
    if len(raw) < context_length:
        pad = [float("nan")] * (context_length - len(raw))
        raw = pad + raw

    working_values = np.array(raw, dtype=np.float32).tolist()

    def encode_context_from_values(values_segment: List[float]):
        arr = np.array(values_segment, dtype=np.float32)
        # scale_and_quantize expects a time_series array; pass the context (length == context_length)
        token_ids, used_scale = scale_and_quantize(
            arr,
            low_limit,
            high_limit,
            num_bins,
            pad_token_id,
            scale=None
        )

        return token_ids, used_scale

    if mode == "fixed_scale":
        context_vals = working_values[-context_length:]
        context_tokens, fixed_scale = encode_context_from_values(context_vals)
        current_token_ids = context_tokens.copy()
        current_scale = fixed_scale
    else:
        context_vals = working_values[-context_length:]
        current_token_ids, current_scale = encode_context_from_values(context_vals)

    with torch.no_grad():
        for step in range(max_tokens):
            indices_np = np.array(current_token_ids, dtype=np.int64)
            indices = torch.from_numpy(indices_np).view(1, -1).to(model.device)

            if hasattr(model.config, "block_size"):
                block_size = model.config.block_size
            else:
                block_size = indices.shape[1]

            if indices.shape[1] > block_size:
                indices_cond = indices[:, -block_size:]
            else:
                indices_cond = indices

            logits, _ = model(indices_cond)
            logits = logits[:, -1, :].squeeze(0)
            logits = logits / float(temperature)

            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            current_token_ids = np.concatenate([current_token_ids, np.array([next_token], dtype=np.int64)])

            pred_value = float(tokens_to_values(
                np.array([next_token], dtype=np.int64),
                low_limit,
                high_limit,
                num_bins,
                pad_token_id,
                current_scale
            )[0])

            working_values.append(pred_value)

            if mode == "sliding_scale":
                new_context = working_values[-context_length:]
                new_tokens, new_scale = encode_context_from_values(new_context)
                current_token_ids = new_tokens.copy()
                current_scale = new_scale
            else:
                if len(current_token_ids) > context_length:
                    current_token_ids = current_token_ids[-context_length:]

    return working_values
