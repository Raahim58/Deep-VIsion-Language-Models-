"""
Advantage estimation utilities (GAE and normalisation).
"""
from __future__ import annotations

import torch


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (GAE) for a batch of token sequences.

    Args:
        rewards:       (B, T) per-token rewards (task reward placed at last token,
                       KL shaping at every response token)
        values:        (B, T) V_old(s_t) estimates from the critic
        response_mask: (B, T) 1 for response tokens, 0 for prompt/padding
        gamma:         discount factor (1.0 for LM tasks — no discounting)
        lam:           GAE lambda

    Returns:
        advantages: (B, T)
        returns:    (B, T) = V_old + advantages (GAE targets for critic)
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    # Iterate backwards through time
    for t in reversed(range(T)):
        mask_t = response_mask[:, t].float()
        if t == T - 1:
            next_value = torch.zeros(B, device=rewards.device)
        else:
            next_value = values[:, t + 1] * response_mask[:, t + 1].float()

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae * (response_mask[:, t + 1].float() if t < T - 1 else torch.zeros(B, device=rewards.device))
        advantages[:, t] = last_gae * mask_t

    returns = advantages + values
    return advantages, returns


def normalise_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalise advantages across all valid (masked) positions batch-wide.
    """
    valid = advantages[mask.bool()]
    if valid.numel() == 0:
        return advantages
    mean = valid.mean()
    std  = valid.std().clamp(min=eps)
    return (advantages - mean) / std
