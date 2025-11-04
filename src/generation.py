from copy import deepcopy
import json
import numpy as np
import torch
from typing import List

from src.tokenizer import Tokenizer


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
        tokenizer: Tokenizer,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        log_file: str = 'generation_log.json',
) -> List[float]:
    model.eval()

    token_log = []

    raw = list(series)
    if len(raw) < tokenizer.config.context_length:
        pad = [float("nan")] * (tokenizer.config.context_length - len(raw))
        raw = pad + raw

    working_values = np.array(raw, dtype=np.float32)
    context_vals = torch.from_numpy(working_values[-tokenizer.config.context_length:]).long()
    context_tokens, current_scale = tokenizer.context_input_transform(context_vals)
    current_token_ids = deepcopy(context_tokens)

    working_values = working_values.tolist()

    with torch.no_grad():
        for step in range(max_tokens):
            indices = current_token_ids.view(1, -1).to(model.device)

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

            logits[tokenizer.config.pad_token_id] = float('-inf')

            #token_min = int(tokenizer.config.n_tokens * 0.1)
            #token_max = int(tokenizer.config.n_tokens * 0.9)
            #logits[:token_min] *= 0.2
            #logits[token_max:] *= 0.2

            logits_mean = logits.mean().item()
            logits_std = logits.std().item()
            logits_max = logits.max().item()
            logits_min = logits.min().item()

            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top5_probs, top5_tokens = torch.topk(probs, k=min(5, len(probs)))
            next_token = torch.multinomial(probs, num_samples=1)

            current_token_ids = torch.concat([current_token_ids, next_token.cpu()], dim=0)

            pred_value = float(tokenizer.output_transform(current_token_ids, current_scale).cpu().numpy()[0, -1])

            token_log.append({
                'step': step,
                'token': next_token.item(),
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

            if len(current_token_ids) > tokenizer.config.context_length:
                current_token_ids = current_token_ids[-tokenizer.config.context_length:]

        print(f"Summary: {len(set(entry['token'] for entry in token_log))} unique tokens generated")
        print(f"Token range: [{min(entry['token'] for entry in token_log)}, {max(entry['token'] for entry in token_log)}]")
        print(f"Value range: [{min(entry['value'] for entry in token_log):.2f}, {max(entry['value'] for entry in token_log):.2f}]")
        print(f"Avg entropy: {np.mean([entry['entropy'] for entry in token_log]):.4f}")

    return working_values
