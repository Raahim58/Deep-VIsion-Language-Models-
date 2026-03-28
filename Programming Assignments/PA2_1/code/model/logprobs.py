from __future__ import annotations

import torch


def gather_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def model_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    return gather_token_logprobs(logits, labels)


def response_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_logprobs = model_token_logprobs(model, input_ids, attention_mask)
    valid_mask = response_mask[:, 1:].float()
    return token_logprobs, valid_mask


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / denom


def sequence_response_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    token_logprobs, valid_mask = response_token_logprobs(model, input_ids, attention_mask, response_mask)
    return masked_sum(token_logprobs, valid_mask)
