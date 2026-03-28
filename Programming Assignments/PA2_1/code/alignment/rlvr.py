from __future__ import annotations

import torch

from utils.metrics import gsm8k_exact_match


def gsm8k_verifiable_rewards(prompts: list[str], completions: list[str], gold_solutions: list[str]) -> torch.Tensor:
    rewards = [1.0 if gsm8k_exact_match(completion, gold) else 0.0 for completion, gold in zip(completions, gold_solutions)]
    return torch.tensor(rewards, dtype=torch.float32)
