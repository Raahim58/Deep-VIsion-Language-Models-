"""
Direct Preference Optimisation (DPO) loss and training step.

Reference: Rafailov et al., "Direct Preference Optimization:
Your Language Model is Secretly a Reward Model", NeurIPS 2023.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from model.logprobs import sequence_logprobs


def dpo_loss(
    policy_model,
    ref_model,
    input_ids_pos: torch.Tensor,
    attention_mask_pos: torch.Tensor,
    input_ids_neg: torch.Tensor,
    attention_mask_neg: torch.Tensor,
    prompt_lens: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the DPO loss for a batch of preference pairs.

    The reference model must be frozen. We wrap its forward pass with
    no_grad inside sequence_logprobs.

    Returns:
        loss:  scalar DPO loss
        info:  dict with diagnostics (margin z, preference accuracy, ...)
    """
    # Log-probs under the trainable policy
    logp_pos_policy = sequence_logprobs(policy_model, input_ids_pos, attention_mask_pos, prompt_lens)
    logp_neg_policy = sequence_logprobs(policy_model, input_ids_neg, attention_mask_neg, prompt_lens)

    # Log-probs under the frozen reference — no gradients
    with torch.no_grad():
        logp_pos_ref = sequence_logprobs(ref_model, input_ids_pos, attention_mask_pos, prompt_lens)
        logp_neg_ref = sequence_logprobs(ref_model, input_ids_neg, attention_mask_neg, prompt_lens)

    # z = beta * ((log pi(y+) - log pi(y-)) - (log ref(y+) - log ref(y-)))
    delta_policy = logp_pos_policy - logp_neg_policy
    delta_ref    = logp_pos_ref    - logp_neg_ref
    z = beta * (delta_policy - delta_ref)

    loss = -F.logsigmoid(z).mean()

    with torch.no_grad():
        pref_acc = (z > 0).float().mean().item()
        margin   = z.mean().item()

    info = {
        "dpo_loss":    loss.item(),
        "margin_z":    margin,
        "pref_acc":    pref_acc,
        "logp_pos":    logp_pos_policy.mean().item(),
        "logp_neg":    logp_neg_policy.mean().item(),
    }
    return loss, info


def dpo_step(
    policy_model,
    ref_model,
    batch: dict,
    optimizer,
    grad_scaler=None,
    beta: float = 0.1,
    max_grad_norm: float = 1.0,
) -> dict:
    """
    One gradient-accumulation-aware DPO step.
    Call optimizer.step() externally when grad_accum steps are done.
    """
    device = next(policy_model.parameters()).device
    ids_pos  = batch["input_ids_pos"].to(device)
    mask_pos = batch["attention_mask_pos"].to(device)
    ids_neg  = batch["input_ids_neg"].to(device)
    mask_neg = batch["attention_mask_neg"].to(device)
    plens    = batch["prompt_lens"].to(device)

    loss, info = dpo_loss(
        policy_model, ref_model,
        ids_pos, mask_pos, ids_neg, mask_neg, plens,
        beta=beta,
    )

    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()

    return info
