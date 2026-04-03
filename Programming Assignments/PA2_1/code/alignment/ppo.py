from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from alignment.advantages import compute_gae, standardize_masked
from alignment.kl import mean_masked_kl
from data.collators import tokenize_prompt_response
from model.generation import generate_completions
from model.logprobs import response_token_logprobs
from model.reward_model import score_sequences

MAX_GRAD_NORM = 1.0


def _left_pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    return torch.tensor(
        [[pad_value] * (max_len - len(seq)) + seq for seq in seqs],
        dtype=torch.long,
    )


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


def build_rollout_batch(
    tokenizer,
    prompts: list[str],
    completions: list[str],
    max_length: int,
) -> dict[str, torch.Tensor]:
    input_ids_list, attn_list, resp_list = [], [], []
    for prompt, completion in zip(prompts, completions):
        ids, resp_mask = tokenize_prompt_response(tokenizer, prompt, completion, max_length)
        input_ids_list.append(ids)
        attn_list.append([1] * len(ids))
        resp_list.append(resp_mask)
    return {
        "input_ids":      _left_pad(input_ids_list, tokenizer.pad_token_id),
        "attention_mask": _left_pad(attn_list, 0),
        "response_mask":  _left_pad(resp_list, 0),
    }


@torch.no_grad()
def score_with_reward_model(
    reward_model, reward_tokenizer, texts: list[str], max_length: int
) -> torch.Tensor:
    device = next(reward_model.parameters()).device
    batch = reward_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    return score_sequences(reward_model, batch["input_ids"], batch["attention_mask"]).float().detach().cpu()


@torch.no_grad()
def collect_ppo_rollout(
    policy,
    reference_model,
    value_model,
    reward_model,
    policy_tokenizer,
    reward_tokenizer,
    prompts: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    was_training_policy = policy.training
    was_training_value  = value_model.training
    policy.eval()
    value_model.eval()

    completions = generate_completions(
        policy,
        policy_tokenizer,
        prompts,
        max_new_tokens=config["ppo"]["max_new_tokens"],
        do_sample=config["generation"]["do_sample"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )

    rollout_batch = build_rollout_batch(policy_tokenizer, prompts, completions, config["max_seq_len"])
    device = next(policy.parameters()).device
    rollout_batch = {k: v.to(device) for k, v in rollout_batch.items()}

    old_logprobs, valid_mask = response_token_logprobs(
        policy,
        rollout_batch["input_ids"],
        rollout_batch["attention_mask"],
        rollout_batch["response_mask"],
    )
    ref_logprobs, _ = response_token_logprobs(
        reference_model,
        rollout_batch["input_ids"],
        rollout_batch["attention_mask"],
        rollout_batch["response_mask"],
    )

    # value estimates — shape (B, T-1), masked to response tokens only
    values = value_model(rollout_batch["input_ids"], rollout_batch["attention_mask"])[:, :-1]

    texts = [p + c for p, c in zip(prompts, completions)]
    sequence_rewards = score_with_reward_model(
        reward_model, reward_tokenizer, texts, config["max_seq_len"]
    ).to(device)

    # KL-shaped per-token rewards + terminal reward at last response token
    token_rewards = -config["ppo"]["beta"] * (old_logprobs - ref_logprobs) * valid_mask
    for i in range(token_rewards.size(0)):
        positions = torch.nonzero(valid_mask[i] > 0, as_tuple=False).flatten()
        if len(positions) > 0:
            token_rewards[i, positions[-1]] += sequence_rewards[i]

    advantages, returns = compute_gae(
        rewards=token_rewards,
        values=values,
        mask=valid_mask,
        gamma=config["ppo"]["gamma"],
        lam=config["ppo"]["gae_lambda"],
    )
    advantages = standardize_masked(advantages, valid_mask)

    if was_training_policy:
        policy.train()
    if was_training_value:
        value_model.train()

    return {
        **rollout_batch,
        "old_logprobs":    old_logprobs.detach(),
        "ref_logprobs":    ref_logprobs.detach(),
        "valid_mask":      valid_mask.detach(),
        "advantages":      advantages.detach(),
        "returns":         returns.detach(),
        "sequence_rewards": sequence_rewards.detach(),
        "completions":     completions,
    }


def ppo_update_step(
    policy,
    value_model,
    policy_optimizer,
    value_optimizer,
    rollout: dict[str, Any],
    epsilon: float,
    value_coef: float,
    entropy_coef: float = 0.0,
) -> dict[str, float]:
    policy.train()
    value_model.train()

    input_ids      = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    response_mask  = rollout["response_mask"]
    valid_mask     = rollout["valid_mask"]
    old_logprobs   = rollout["old_logprobs"]
    advantages     = rollout["advantages"]
    returns        = rollout["returns"]
    ref_logprobs   = rollout["ref_logprobs"]

    # ── policy loss ───────────────────────────────────────────────────────────
    token_logprobs, _ = response_token_logprobs(policy, input_ids, attention_mask, response_mask)
    ratio         = torch.exp(token_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_loss = -(
        torch.minimum(ratio * advantages, clipped_ratio * advantages) * valid_mask
    ).sum() / valid_mask.sum().clamp_min(1.0)

    entropy = torch.tensor(0.0, device=input_ids.device)
    if entropy_coef > 0:
        logits   = policy(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        probs    = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy  = -(probs * log_probs).sum(dim=-1)
        entropy  = (entropy * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    policy_optimizer.zero_grad(set_to_none=True)
    (policy_loss - entropy_coef * entropy).backward()
    torch.nn.utils.clip_grad_norm_(
        list(filter(lambda p: p.requires_grad, policy.parameters())),
        MAX_GRAD_NORM,
    )
    policy_grad_norm = _grad_norm(filter(lambda p: p.requires_grad, policy.parameters()))
    policy_optimizer.step()

    # ── value loss ────────────────────────────────────────────────────────────
    values = value_model(input_ids, attention_mask)[:, :-1]
    value_loss = (
        (values - returns) ** 2 * valid_mask
    ).sum() / valid_mask.sum().clamp_min(1.0)

    value_optimizer.zero_grad(set_to_none=True)
    (value_coef * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(
        list(filter(lambda p: p.requires_grad, value_model.parameters())),
        MAX_GRAD_NORM,
    )
    value_grad_norm = _grad_norm(filter(lambda p: p.requires_grad, value_model.parameters()))
    value_optimizer.step()

    approx_kl  = mean_masked_kl(token_logprobs.detach(), ref_logprobs, valid_mask).item()
    ratio_mean = ((ratio * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)).item()

    return {
        "policy_loss":      float(policy_loss.item()),
        "value_loss":       float(value_loss.item()),
        "entropy":          float(entropy.item()),
        "mean_kl":          float(approx_kl),
        "ratio_mean":       float(ratio_mean),
        "mean_reward":      float(rollout["sequence_rewards"].mean().item()),
        "policy_grad_norm": float(policy_grad_norm),
        "value_grad_norm":  float(value_grad_norm),
    }
