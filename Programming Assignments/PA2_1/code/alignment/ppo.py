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


def _left_pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    return torch.tensor([[pad_value] * (max_len - len(seq)) + seq for seq in seqs], dtype=torch.long)


def build_rollout_batch(tokenizer, prompts: list[str], completions: list[str], max_length: int) -> dict[str, torch.Tensor]:
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
def score_with_reward_model(reward_model, reward_tokenizer, texts: list[str], max_length: int) -> torch.Tensor:
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
    was_training = policy.training
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
    rollout_batch = {key: value.to(device) for key, value in rollout_batch.items()}

    old_logprobs, valid_mask = response_token_logprobs(
        policy, rollout_batch["input_ids"], rollout_batch["attention_mask"], rollout_batch["response_mask"]
    )
    ref_logprobs, _ = response_token_logprobs(
        reference_model, rollout_batch["input_ids"], rollout_batch["attention_mask"], rollout_batch["response_mask"]
    )
    values = value_model(rollout_batch["input_ids"], rollout_batch["attention_mask"])[:, :-1]

    texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
    sequence_rewards = score_with_reward_model(reward_model, reward_tokenizer, texts, config["max_seq_len"]).to(device)

    token_rewards = -config["ppo"]["beta"] * (old_logprobs - ref_logprobs) * valid_mask
    for row_idx in range(token_rewards.size(0)):
        positions = torch.nonzero(valid_mask[row_idx] > 0, as_tuple=False).flatten()
        if len(positions) > 0:
            token_rewards[row_idx, positions[-1]] += sequence_rewards[row_idx]

    advantages, returns = compute_gae(
        rewards=token_rewards,
        values=values,
        mask=valid_mask,
        gamma=config["ppo"]["gamma"],
        lam=config["ppo"]["gae_lambda"],
    )
    advantages = standardize_masked(advantages, valid_mask)

    if was_training:
        policy.train()
    value_model.train()
    return {
        **rollout_batch,
        "old_logprobs": old_logprobs.detach(),
        "ref_logprobs": ref_logprobs.detach(),
        "valid_mask": valid_mask.detach(),
        "advantages": advantages.detach(),
        "returns": returns.detach(),
        "sequence_rewards": sequence_rewards.detach(),
        "completions": completions,
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
    was_training = policy.training
    policy.eval()
    input_ids = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    response_mask = rollout["response_mask"]
    valid_mask = rollout["valid_mask"]
    old_logprobs = rollout["old_logprobs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]
    ref_logprobs = rollout["ref_logprobs"]

    token_logprobs, _ = response_token_logprobs(policy, input_ids, attention_mask, response_mask)
    ratio = torch.exp(token_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    policy_loss = -(torch.minimum(unclipped, clipped) * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    entropy = torch.tensor(0.0, device=input_ids.device)
    if entropy_coef > 0:
        logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    policy_optimizer.zero_grad(set_to_none=True)
    total_policy_loss = policy_loss - entropy_coef * entropy
    total_policy_loss.backward()
    policy_optimizer.step()
    if was_training:
        policy.train()

    values = value_model(input_ids, attention_mask)[:, :-1]
    value_loss = ((values - returns) ** 2 * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    value_optimizer.zero_grad(set_to_none=True)
    (value_coef * value_loss).backward()
    value_optimizer.step()

    approx_kl = mean_masked_kl(token_logprobs.detach(), ref_logprobs, valid_mask).item()
    ratio_mean = ((ratio * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)).item()
    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "mean_kl": float(approx_kl),
        "ratio_mean": float(ratio_mean),
        "mean_reward": float(rollout["sequence_rewards"].mean().item()),
    }
