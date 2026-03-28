"""
Log-probability computation utilities.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities for the *generated* tokens.

    Args:
        model:          causal LM (policy or reference)
        input_ids:      (B, T) — full (prompt + response) token ids
        attention_mask: (B, T)
        response_start: (B,) int tensor — index where response starts per item.
                        If None, returns log-probs for ALL positions.

    Returns:
        logp: (B, T-1) — log p(token_t | token_{<t}), aligned so logp[:, i]
              corresponds to the probability of input_ids[:, i+1].
              Positions belonging to the prompt (and padding) should be masked
              out by the caller.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out.logits  # (B, T, V)
    # Shift: logits[:, i] predicts token at position i+1
    logits_shifted = logits[:, :-1, :]  # (B, T-1, V)
    targets = input_ids[:, 1:]           # (B, T-1)

    log_probs_all = F.log_softmax(logits_shifted, dim=-1)  # (B, T-1, V)
    # Gather the log-prob of the actual token
    logp = log_probs_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return logp


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the summed log-probability of the response portion only.

    response tokens = input_ids[:, prompt_len:]
    Padding tokens are excluded from the sum.

    Returns:
        sum_logp: (B,) scalar per sequence.
    """
    logp = token_logprobs(model, input_ids, attention_mask)  # (B, T-1)
    B, T_minus1 = logp.shape

    sum_logp = torch.zeros(B, device=logp.device)
    for i in range(B):
        plen = prompt_lens[i].item()
        # After left-padding, find where the real tokens start
        real_start = (attention_mask[i] == 1).nonzero(as_tuple=False)[0].item()
        # Response starts at real_start + plen (in original token count)
        # logp[:, i] = log p(token i+1), so response logprobs start at plen-1
        # We want positions plen..T-1 in the original sequence.
        # In the shifted tensor that is indices plen-1..T-2.
        resp_start_shifted = real_start + plen - 1
        resp_end_shifted = T_minus1  # up to last position

        if resp_start_shifted >= resp_end_shifted:
            continue

        # Mask out padding
        resp_mask = attention_mask[i, 1:]  # (T-1,) shifted
        resp_mask[:resp_start_shifted] = 0  # zero out prompt portion

        sum_logp[i] = (logp[i] * resp_mask.float()).sum()

    return sum_logp


def full_vocab_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Return full log-prob distribution over vocabulary at each position.
    Shape: (B, T-1, V). Used for exact KL computation in GRPO.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    return F.log_softmax(logits, dim=-1)
