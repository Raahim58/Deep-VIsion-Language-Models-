from __future__ import annotations

from typing import Any, Callable

import torch

from alignment.kl import mean_masked_kl
from data.collators import tokenize_prompt_response
from model.generation import generate_completions
from model.logprobs import response_token_logprobs
from model.reward_model import score_sequences


def _left_pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    return torch.tensor([[pad_value] * (max_len - len(seq)) + seq for seq in seqs], dtype=torch.long)


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.norm(2).item() ** 2
    return total ** 0.5


def _build_batch(tokenizer, prompts: list[str], completions: list[str], max_length: int) -> dict[str, torch.Tensor]:
    input_ids = []
    attention_masks = []
    response_masks = []
    for prompt, completion in zip(prompts, completions):
        ids, response_mask = tokenize_prompt_response(tokenizer, prompt, completion, max_length)
        input_ids.append(ids)
        attention_masks.append([1] * len(ids))
        response_masks.append(response_mask)
    return {
        "input_ids": _left_pad(input_ids, tokenizer.pad_token_id),
        "attention_mask": _left_pad(attention_masks, 0),
        "response_mask": _left_pad(response_masks, 0),
    }


@torch.no_grad()
def _score_with_rm(reward_model, reward_tokenizer, texts: list[str], max_length: int) -> torch.Tensor:
    device = next(reward_model.parameters()).device
    batch = reward_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    return score_sequences(reward_model, batch["input_ids"], batch["attention_mask"]).float().detach().cpu()


@torch.no_grad()
def collect_group_rollout(
    policy,
    reference_model,
    tokenizer,
    prompts: list[str],
    config: dict[str, Any],
    reward_fn: Callable[[list[str], list[str]], torch.Tensor],
    max_new_tokens: int,
) -> dict[str, Any]:
    was_training = policy.training
    policy.eval()
    expanded_prompts = [prompt for prompt in prompts for _ in range(config["grpo"]["k_rollouts"])]
    completions = generate_completions(
        policy,
        tokenizer,
        expanded_prompts,
        max_new_tokens=max_new_tokens,
        do_sample=config["generation"]["do_sample"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )
    rollout_batch = _build_batch(tokenizer, expanded_prompts, completions, config["max_seq_len"])
    device = next(policy.parameters()).device
    rollout_batch = {key: value.to(device) for key, value in rollout_batch.items()}

    old_logprobs, valid_mask = response_token_logprobs(
        policy, rollout_batch["input_ids"], rollout_batch["attention_mask"], rollout_batch["response_mask"]
    )
    ref_logprobs, _ = response_token_logprobs(
        reference_model, rollout_batch["input_ids"], rollout_batch["attention_mask"], rollout_batch["response_mask"]
    )

    rewards = reward_fn(expanded_prompts, completions).to(device)
    if rewards.numel() != len(expanded_prompts):
        raise ValueError(
            f"GRPO reward function returned {rewards.numel()} rewards for {len(expanded_prompts)} completions."
        )
    rewards_grouped = rewards.view(len(prompts), config["grpo"]["k_rollouts"])
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, unbiased=False, keepdim=True)
    seq_advantages = (rewards_grouped - group_mean).view(-1)
    token_advantages = valid_mask * seq_advantages.unsqueeze(1)
    valid_advantages = token_advantages[valid_mask > 0]
    if valid_advantages.numel() > 0:
        adv_mean = valid_advantages.mean()
        adv_std = valid_advantages.std(unbiased=False).clamp_min(1.0e-8)
        token_advantages = ((token_advantages - adv_mean) / adv_std) * valid_mask
    degenerate_frac = float((group_std.squeeze(1) < 1.0e-4).float().mean().item())

    if was_training:
        policy.train()
    return {
        **rollout_batch,
        "old_logprobs": old_logprobs.detach(),
        "ref_logprobs": ref_logprobs.detach(),
        "valid_mask": valid_mask.detach(),
        "advantages": token_advantages.detach(),
        "sequence_rewards": rewards.detach(),
        "expanded_prompts": expanded_prompts,
        "completions": completions,
        "degenerate_frac": degenerate_frac,
    }


def grpo_update_step(
    policy,
    policy_optimizer,
    rollout: dict[str, Any],
    epsilon: float,
    beta: float,
) -> dict[str, float]:
    was_training = policy.training
    policy.train()
    input_ids = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    response_mask = rollout["response_mask"]
    valid_mask = rollout["valid_mask"]
    old_logprobs = rollout["old_logprobs"]
    ref_logprobs = rollout["ref_logprobs"]
    advantages = rollout["advantages"]

    new_logprobs, _ = response_token_logprobs(policy, input_ids, attention_mask, response_mask)
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    policy_loss = -(torch.minimum(unclipped, clipped) * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    kl_loss = mean_masked_kl(new_logprobs, ref_logprobs, valid_mask)

    policy_optimizer.zero_grad(set_to_none=True)
    total_loss = policy_loss + beta * kl_loss
    total_loss.backward()
    grad_norm = _grad_norm(policy.parameters())
    policy_optimizer.step()
    if was_training:
        policy.train()

    ratio_mean = ((ratio * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)).item()
    return {
        "policy_loss": float(policy_loss.item()),
        "mean_kl": float(kl_loss.item()),
        "ratio_mean": float(ratio_mean),
        "mean_reward": float(rollout["sequence_rewards"].mean().item()),
        "policy_grad_norm": float(grad_norm),
        "degenerate": float(rollout.get("degenerate_frac", 0.0)),
    }


def build_rm_reward_fn(reward_model, reward_tokenizer, max_length: int):
    def score(prompts: list[str], completions: list[str]) -> torch.Tensor:
        texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
        return _score_with_rm(reward_model, reward_tokenizer, texts, max_length)

    return score
