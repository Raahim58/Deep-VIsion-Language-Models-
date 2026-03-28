from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_reward_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def score_sequences(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits.squeeze(-1)
