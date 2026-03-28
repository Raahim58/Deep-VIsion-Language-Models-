from __future__ import annotations

import torch

from model.logprobs import masked_mean


def sampled_token_kl(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    return policy_logprobs - ref_logprobs


def mean_masked_kl(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(sampled_token_kl(policy_logprobs, ref_logprobs), mask).mean()
