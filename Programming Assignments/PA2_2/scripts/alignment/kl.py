"""
KL divergence utilities for RLHF training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_penalty(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Per-token KL penalty: beta * (log pi_old - log pi_ref).
    Used as the per-token reward shaping signal in PPO.

    Args:
        logp_policy:   (B, T) — log-probs of sampled tokens under old policy
        logp_ref:      (B, T) — log-probs of sampled tokens under πref
        response_mask: (B, T) — 1 for response tokens, 0 for prompt/padding

    Returns:
        kl: (B, T) per-token KL (already masked; prompt positions = 0)
    """
    kl = (logp_policy - logp_ref) * response_mask.float()
    return kl


@torch.no_grad()
def kl_from_ref(
    policy_model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Monte-Carlo estimate of KL(pi_policy || pi_ref) for monitoring.
    Uses sampled-token approximation: KL_t ≈ log pi_policy(y_t) - log pi_ref(y_t).

    Returns scalar mean KL across all response tokens in the batch.
    """
    from model.logprobs import token_logprobs

    logp_pol = token_logprobs(policy_model, input_ids, attention_mask)  # (B, T-1)
    logp_ref_vals = token_logprobs(ref_model, input_ids, attention_mask)  # (B, T-1)

    # response_mask is (B, T); shift to (B, T-1) to align with logp
    mask = response_mask[:, 1:].float()

    kl_tokens = (logp_pol - logp_ref_vals) * mask
    total_tokens = mask.sum().clamp(min=1)
    return kl_tokens.sum() / total_tokens


def token_level_kl_exact(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Exact per-position KL: sum_v pi(v) * (log pi(v) - log ref(v)).

    Args:
        log_probs_policy: (B, T, V) full log-softmax from policy
        log_probs_ref:    (B, T, V) full log-softmax from ref

    Returns:
        kl: (B, T) per-token KL values
    """
    probs_policy = log_probs_policy.exp()
    kl = (probs_policy * (log_probs_policy - log_probs_ref)).sum(dim=-1)
    return kl
