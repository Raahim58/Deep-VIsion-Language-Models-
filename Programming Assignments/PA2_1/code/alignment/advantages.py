from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    batch_size, seq_len = rewards.shape
    for batch_idx in range(batch_size):
        valid_positions = torch.nonzero(mask[batch_idx] > 0, as_tuple=False).flatten()
        if len(valid_positions) == 0:
            continue
        gae = 0.0
        next_value = 0.0
        for pos in reversed(valid_positions.tolist()):
            value = values[batch_idx, pos]
            delta = rewards[batch_idx, pos] + gamma * next_value - value
            gae = delta + gamma * lam * gae
            advantages[batch_idx, pos] = gae
            returns[batch_idx, pos] = gae + value
            next_value = value
    return advantages, returns


def standardize_masked(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    valid = values[mask > 0]
    if valid.numel() == 0:
        return torch.zeros_like(values)
    mean = valid.mean()
    std = valid.std(unbiased=False)
    if torch.isnan(std) or std < eps:
        return torch.zeros_like(values)
    standardized = (values - mean) / (std + eps)
    return standardized * mask
