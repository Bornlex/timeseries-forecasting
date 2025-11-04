import json
import numpy as np
import torch
from typing import List

from src.tokenizer import scale_and_quantize


def tokens_to_values(
    token_ids: np.ndarray,
    low_limit: float,
    high_limit: float,
    num_bins: int,
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
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        log_file: str = 'generation_log.json',
) -> List[float]:
    model.eval()

    token_log = []

    raw = list(series)
    if len(raw) < context_length:
        pad = [float("nan")] * (context_length - len(raw))
        raw = pad + raw

    working_values = np.array(raw, dtype=np.float32).tolist()

    def encode_context_from_values(values_segment: List[float]):
        arr = np.array(values_segment, dtype=np.float32)
        token_ids, used_scale = scale_and_quantize(
            arr,
            low_limit,
            high_limit,
            num_bins,
            pad_token_id,
            scale=None
        )

        return token_ids, used_scale

    context_vals = working_values[-context_length:]
    context_tokens, current_scale = encode_context_from_values(context_vals)
    current_token_ids = context_tokens.copy()

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

            logits[pad_token_id] = float('-inf')

            # token_max = int(num_bins * 0.9)
            # logits[token_max:] = float('-inf')

            logits_mean = logits.mean().item()
            logits_std = logits.std().item()
            logits_max = logits.max().item()
            logits_min = logits.min().item()

            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top5_probs, top5_tokens = torch.topk(probs, k=min(5, len(probs)))
            next_token = torch.multinomial(probs, num_samples=1).item()

            current_token_ids = np.concatenate([current_token_ids, np.array([next_token], dtype=np.int64)])

            pred_value = float(tokens_to_values(
                np.array([next_token], dtype=np.int64),
                low_limit,
                high_limit,
                num_bins,
                current_scale
            )[0])

            token_log.append({
                'step': step,
                'token': next_token,
                'value': pred_value,
                'scale': current_scale,
                'token_prob': probs[next_token].item(),
                'logits_mean': logits_mean,
                'logits_std': logits_std,
                'logits_max': logits_max,
                'logits_min': logits_min,
                'top5_tokens': top5_tokens.cpu().numpy().tolist(),
                'top5_probs': top5_probs.cpu().numpy().tolist(),
                'entropy': -(probs * torch.log(probs + 1e-10)).sum().item(),
            })

            working_values.append(pred_value)

            if len(current_token_ids) > context_length:
                current_token_ids = current_token_ids[-context_length:]

    if log_file:
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            else:
                return obj

        token_log_native = convert_to_native(token_log)

        with open(log_file, 'w') as f:
            json.dump({
                'generation_log': token_log_native,
                'summary': {
                    'total_steps': len(token_log),
                    'unique_tokens': len(set(entry['token'] for entry in token_log)),
                    'token_range': [min(entry['token'] for entry in token_log),
                                    max(entry['token'] for entry in token_log)],
                    'value_range': [min(entry['value'] for entry in token_log),
                                    max(entry['value'] for entry in token_log)],
                    'avg_entropy': np.mean([entry['entropy'] for entry in token_log]),
                    'temperature': temperature,
                }
            }, f, indent=2)

        print(f"Token distribution logged to {log_file}")
        print(f"Summary: {len(set(entry['token'] for entry in token_log))} unique tokens generated")
        print(f"Token range: [{min(entry['token'] for entry in token_log)}, {max(entry['token'] for entry in token_log)}]")
        print(f"Value range: [{min(entry['value'] for entry in token_log):.2f}, {max(entry['value'] for entry in token_log):.2f}]")
        print(f"Avg entropy: {np.mean([entry['entropy'] for entry in token_log]):.4f}")

    return working_values
