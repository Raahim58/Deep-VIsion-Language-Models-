# # """
# Group Relative Policy Optimisation (GRPO).

# Reference: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical
# Reasoning in Open Language Models", arXiv 2024.
# """
# from __future__ import annotations

# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass, field
# from typing import List

# from model.logprobs import token_logprobs
# from model.generation import generate_k_responses
# from model.reward_model import score_texts
# from alignment.advantages import normalise_advantages


# @dataclass
# class GRPORollout:
#     """Data from one GRPO rollout step."""
#     prompts:       List[str]
#     responses:     List[List[str]]       # [B, K] strings
#     input_ids:     torch.Tensor          # (B*K, T)
#     attention_mask:torch.Tensor          # (B*K, T)
#     response_mask: torch.Tensor          # (B*K, T)
#     logp_old:      torch.Tensor          # (B*K, T-1)
#     advantages:    torch.Tensor          # (B*K, T)
#     rewards:       List[List[float]]     # [B, K] scalars
#     degenerate_frac: float               # fraction of prompts where all K rewards equal


# @torch.no_grad()
# def grpo_rollout(
#     policy_model,
#     ref_model,
#     reward_fn,          # callable(texts: List[str]) -> List[float]
#     policy_tokenizer,
#     prompts: List[str],
#     K: int = 4,
#     max_new_tokens: int = 128,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     device: str = "cuda",
# ) -> GRPORollout:
#     """
#     For each of B prompts, sample K completions, score them, compute
#     group-relative advantages, and cache old log-probs.
#     """
#     B = len(prompts)
#     all_responses   = []   # [B][K]
#     all_full_texts  = []   # B*K strings (flattened)
#     flat_prompts    = []   # B*K prompt strings

#     for prompt in prompts:
#         resps = generate_k_responses(
#             policy_model, policy_tokenizer, prompt, K,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature, top_p=top_p, device=device,
#         )
#         all_responses.append(resps)
#         for r in resps:
#             all_full_texts.append(prompt + " " + r)
#             flat_prompts.append(prompt)

#     # Score all B*K completions
#     flat_rewards = reward_fn(all_full_texts)   # length B*K

#     # Reshape to [B, K]
#     rewards_2d = [flat_rewards[i*K:(i+1)*K] for i in range(B)]

#     # Group-relative advantages
#     group_advantages = []   # B*K scalars
#     degenerate = 0
#     for k_rewards in rewards_2d:
#         mu = sum(k_rewards) / K
#         if all(abs(r - mu) < 1e-6 for r in k_rewards):
#             degenerate += 1
#         for r in k_rewards:
#             group_advantages.append(r - mu)

#     degenerate_frac = degenerate / B

#     # Tokenise all B*K sequences
#     enc = policy_tokenizer(
#         all_full_texts, return_tensors="pt", padding=True,
#         truncation=True, max_length=1024,
#     )
#     input_ids      = enc["input_ids"].to(device)       # (B*K, T)
#     attention_mask = enc["attention_mask"].to(device)  # (B*K, T)
#     BK, T = input_ids.shape

#     # Build response mask
#     prompt_ids_list = [
#         policy_tokenizer.encode(p, add_special_tokens=False) for p in flat_prompts
#     ]
#     response_mask = _build_response_mask(input_ids, attention_mask, prompt_ids_list, T)

#     # Broadcast scalar advantage to all response tokens of that completion
#     adv_tensor = torch.zeros(BK, T, device=device)
#     for i, adv_val in enumerate(group_advantages):
#         adv_tensor[i] = adv_val * response_mask[i].float()

#     # Normalise advantages batch-wide over all valid response tokens
#     adv_tensor = normalise_advantages(adv_tensor, response_mask)

#     # Cache log-probs under current (old) policy
#     logp_old = token_logprobs(policy_model, input_ids, attention_mask)   # (B*K, T-1)

#     return GRPORollout(
#         prompts=prompts,
#         responses=all_responses,
#         input_ids=input_ids.cpu(),
#         attention_mask=attention_mask.cpu(),
#         response_mask=response_mask.cpu(),
#         logp_old=logp_old.cpu(),
#         advantages=adv_tensor.cpu(),
#         rewards=rewards_2d,
#         degenerate_frac=degenerate_frac,
#     )


# def grpo_update(
#     policy_model,
#     ref_model,
#     rollout: GRPORollout,
#     optimizer,
#     epsilon: float = 0.2,
#     beta: float = 0.1,
#     max_grad_norm: float = 1.0,
#     use_approx_kl: bool = True,
#     device: str = "cuda",
# ) -> dict:
#     """
#     One GRPO gradient step using the collected rollout.
#     """
#     input_ids      = rollout.input_ids.to(device)
#     attention_mask = rollout.attention_mask.to(device)
#     response_mask  = rollout.response_mask.to(device)
#     logp_old       = rollout.logp_old.to(device)
#     advantages     = rollout.advantages.to(device)

#     resp_mask_shift = response_mask[:, 1:].float()  # (B*K, T-1)
#     adv_shift       = advantages[:, 1:]              # (B*K, T-1)

#     # Current policy log-probs
#     logp_new = token_logprobs(policy_model, input_ids, attention_mask)  # (B*K, T-1)

#     ratio = (logp_new - logp_old).exp()

#     surr1 = ratio * adv_shift
#     surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * adv_shift
#     policy_loss = -torch.min(surr1, surr2)

#     # Normalise by response length per sequence (1/T_k weighting)
#     lengths = resp_mask_shift.sum(dim=1).clamp(min=1)  # (B*K,)
#     policy_loss = (policy_loss * resp_mask_shift).sum(dim=1) / lengths
#     policy_loss = policy_loss.mean()

#     # KL term (approximate or exact)
#     if use_approx_kl:
#         with torch.no_grad():
#             logp_ref = token_logprobs(ref_model, input_ids, attention_mask)
#         kl_tok = (logp_new - logp_ref) * resp_mask_shift
#         kl_loss = kl_tok.sum() / resp_mask_shift.sum().clamp(min=1)
#     else:
#         kl_loss = torch.tensor(0.0, device=device)

#     total_loss = policy_loss + beta * kl_loss

#     optimizer.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
#     optimizer.step()

#     return {
#         "grpo_loss":   total_loss.item(),
#         "policy_loss": policy_loss.item(),
#         "kl_loss":     kl_loss.item(),
#         "degenerate":  rollout.degenerate_frac,
#         "mean_reward": sum(r for row in rollout.rewards for r in row) / (len(rollout.rewards) * len(rollout.rewards[0])),
#     }


# def _build_response_mask(input_ids, attention_mask, prompt_ids_list, T):
#     B = input_ids.shape[0]
#     mask = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
#     for i, prompt_ids in enumerate(prompt_ids_list):
#         real_start = (attention_mask[i] == 1).nonzero(as_tuple=False)[0].item()
#         resp_start = min(real_start + len(prompt_ids), T)
#         mask[i, resp_start:] = attention_mask[i, resp_start:]
#     return mask

"""
Group Relative Policy Optimisation (GRPO).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

from model.generation import generate_k_responses
from alignment.advantages import normalise_advantages


@dataclass
class GRPORollout:
    prompts:         List[str]
    responses:       List[List[str]]
    input_ids:       torch.Tensor
    attention_mask:  torch.Tensor
    response_mask:   torch.Tensor
    logp_old:        torch.Tensor
    advantages:      torch.Tensor
    rewards:         List[List[float]]
    degenerate_frac: float


@torch.no_grad()
def grpo_rollout(
    policy_model,
    ref_model,
    reward_fn,
    policy_tokenizer,
    prompts: List[str],
    K: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> GRPORollout:

    B = len(prompts)
    all_responses  = []
    all_full_texts = []
    flat_prompts   = []

    for prompt in prompts:
        resps = generate_k_responses(
            policy_model, policy_tokenizer, prompt, K,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, device=device,
        )
        all_responses.append(resps)
        for r in resps:
            all_full_texts.append(prompt + " " + r)
            flat_prompts.append(prompt)

    flat_rewards = reward_fn(all_full_texts)
    rewards_2d   = [flat_rewards[i*K:(i+1)*K] for i in range(B)]

    group_advantages = []
    degenerate = 0
    for k_rewards in rewards_2d:
        mu = sum(k_rewards) / K
        if all(abs(r - mu) < 1e-6 for r in k_rewards):
            degenerate += 1
        for r in k_rewards:
            group_advantages.append(r - mu)

    enc = policy_tokenizer(
        all_full_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    BK, T = input_ids.shape

    prompt_ids_list = [
        policy_tokenizer.encode(p, add_special_tokens=False) for p in flat_prompts
    ]
    response_mask = _build_response_mask(input_ids, attention_mask, prompt_ids_list, T)

    adv_tensor = torch.zeros(BK, T, device=device)
    for i, adv_val in enumerate(group_advantages):
        adv_tensor[i] = adv_val * response_mask[i].float()
    adv_tensor = normalise_advantages(adv_tensor, response_mask)

    # Cache old log-probs — eval mode so dropout is off
    policy_model.eval()
    targets  = input_ids[:, 1:]
    out_old  = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    logp_old = F.log_softmax(out_old.logits[:, :-1, :], dim=-1).gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1).detach()

    return GRPORollout(
        prompts=prompts,
        responses=all_responses,
        input_ids=input_ids.cpu(),
        attention_mask=attention_mask.cpu(),
        response_mask=response_mask.cpu(),
        logp_old=logp_old.cpu(),
        advantages=adv_tensor.cpu(),
        rewards=rewards_2d,
        degenerate_frac=degenerate / B,
    )


def grpo_update(
    policy_model,
    ref_model,
    rollout: GRPORollout,
    optimizer,
    epsilon: float = 0.2,
    beta: float = 0.1,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> dict:

    input_ids      = rollout.input_ids.to(device)
    attention_mask = rollout.attention_mask.to(device)
    response_mask  = rollout.response_mask.to(device)
    logp_old       = rollout.logp_old.to(device)
    advantages     = rollout.advantages.to(device)
    targets        = input_ids[:, 1:]

    resp_mask_shift = response_mask[:, 1:].float()
    adv_shift       = advantages[:, 1:]

    policy_model.train()
    optimizer.zero_grad()

    out      = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    logp_new = F.log_softmax(out.logits[:, :-1, :], dim=-1).gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)

    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * adv_shift
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * adv_shift
    policy_loss = -torch.min(surr1, surr2)

    lengths     = resp_mask_shift.sum(dim=1).clamp(min=1)
    policy_loss = (policy_loss * resp_mask_shift).sum(dim=1) / lengths
    policy_loss = policy_loss.mean()

    with torch.no_grad():
        out_ref  = ref_model(input_ids=input_ids, attention_mask=attention_mask)
        logp_ref = F.log_softmax(out_ref.logits[:, :-1, :], dim=-1).gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)
    kl_tok  = (logp_new - logp_ref) * resp_mask_shift
    kl_loss = kl_tok.sum() / resp_mask_shift.sum().clamp(min=1)

    total_loss = policy_loss + beta * kl_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
    optimizer.step()

    mean_reward = sum(r for row in rollout.rewards for r in row) / (
        len(rollout.rewards) * len(rollout.rewards[0])
    )

    return {
        "grpo_loss":   total_loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_loss":     kl_loss.item(),
        "degenerate":  rollout.degenerate_frac,
        "mean_reward": mean_reward,
    }


def _build_response_mask(input_ids, attention_mask, prompt_ids_list, T):
    B = input_ids.shape[0]
    mask = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
    for i, prompt_ids in enumerate(prompt_ids_list):
        real_start = (attention_mask[i] == 1).nonzero(as_tuple=False)[0].item()
        resp_start = min(real_start + len(prompt_ids), T)
        mask[i, resp_start:] = attention_mask[i, resp_start:]
    return mask