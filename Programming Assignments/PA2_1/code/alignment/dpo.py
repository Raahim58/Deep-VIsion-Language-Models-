from __future__ import annotations

import torch
import torch.nn.functional as F

from model.logprobs import sequence_response_logprobs


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def dpo_batch_metrics(policy, reference_model, batch: dict[str, torch.Tensor], beta: float) -> dict[str, torch.Tensor]:
    device = next(policy.parameters()).device
    batch = _to_device(batch, device)
    chosen_pi = sequence_response_logprobs(
        policy,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_response_mask"],
    )
    rejected_pi = sequence_response_logprobs(
        policy,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_response_mask"],
    )
    with torch.no_grad():
        chosen_ref = sequence_response_logprobs(
            reference_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        rejected_ref = sequence_response_logprobs(
            reference_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )

    margin = beta * ((chosen_pi - chosen_ref) - (rejected_pi - rejected_ref))
    loss = -F.logsigmoid(margin).mean()
    implicit_reward = margin.detach().mean()
    preference_accuracy = ((chosen_pi - rejected_pi) > 0).float().mean()
    return {
        "loss": loss,
        "implicit_reward_margin": implicit_reward,
        "preference_accuracy": preference_accuracy.detach(),
    }
