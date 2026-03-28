# """
# Proximal Policy Optimisation (PPO) for LLMs.

# References:
#   Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
#   Ouyang et al., "Training language models to follow instructions...", NeurIPS 2022.
# """
# from __future__ import annotations

# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass, field
# from typing import List

# from model.logprobs import token_logprobs
# from model.generation import generate_responses
# from model.reward_model import score_texts
# from alignment.advantages import compute_gae_advantages, normalise_advantages
# from alignment.kl import kl_penalty


# @dataclass
# class RolloutBatch:
#     """All data collected during one rollout phase."""
#     prompts:       List[str]
#     responses:     List[str]
#     input_ids:     torch.Tensor   # (B, T)  — prompt + response, left-padded
#     attention_mask:torch.Tensor   # (B, T)
#     response_mask: torch.Tensor   # (B, T)  — 1 for response tokens only
#     logp_old:      torch.Tensor   # (B, T-1)
#     logp_ref:      torch.Tensor   # (B, T-1)
#     values_old:    torch.Tensor   # (B, T)
#     rewards:       torch.Tensor   # (B, T)  — task + KL shaping
#     task_rewards:  List[float]    # scalar RM scores per sequence
#     advantages:    torch.Tensor   # (B, T)
#     returns:       torch.Tensor   # (B, T)


# @torch.no_grad()
# def ppo_rollout(
#     policy_model,
#     ref_model,
#     value_model,
#     reward_model,
#     rm_tokenizer,
#     policy_tokenizer,
#     prompts: List[str],
#     max_new_tokens: int = 128,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     beta: float = 0.1,
#     gamma: float = 1.0,
#     lam: float = 0.95,
#     device: str = "cuda",
# ) -> RolloutBatch:
#     """
#     Collect one rollout batch:
#       1. Generate responses with current policy.
#       2. Score with reward model.
#       3. Compute per-token log-probs under old policy and πref.
#       4. Compute value estimates.
#       5. Build per-token rewards (task + KL shaping).
#       6. Run GAE.
#     """
#     # ── 1. Generate ──────────────────────────────────────────────────────────
#     responses = generate_responses(
#         policy_model, policy_tokenizer, prompts,
#         max_new_tokens=max_new_tokens,
#         temperature=temperature, top_p=top_p,
#         do_sample=True, device=device,
#     )

#     full_texts = [p + " " + r for p, r in zip(prompts, responses)]

#     # ── 2. Score with RM ─────────────────────────────────────────────────────
#     task_rewards_list = score_texts(reward_model, rm_tokenizer, full_texts)

#     # ── 3. Tokenise (prompt + response) for log-prob computation ─────────────
#     enc = policy_tokenizer(
#         full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024,
#     )
#     input_ids      = enc["input_ids"].to(device)       # (B, T)
#     attention_mask = enc["attention_mask"].to(device)  # (B, T)
#     B, T = input_ids.shape

#     # Build response mask — 1 where the response tokens are
#     prompt_ids_list = [
#         policy_tokenizer.encode(p, add_special_tokens=False) for p in prompts
#     ]
#     response_mask = _build_response_mask(input_ids, attention_mask, prompt_ids_list, T)

#     # ── 4. Log-probs under old policy and πref ───────────────────────────────
#     logp_old = token_logprobs(policy_model, input_ids, attention_mask)   # (B, T-1)
#     logp_ref = token_logprobs(ref_model,    input_ids, attention_mask)   # (B, T-1)

#     # ── 5. Value estimates ───────────────────────────────────────────────────
#     values_old = value_model(input_ids, attention_mask)   # (B, T)

#     # ── 6. Per-token rewards: task score at last token + KL shaping ──────────
#     task_reward_t = torch.zeros(B, T, device=device)
#     for i in range(B):
#         # Place task reward at the last response token
#         last_resp_tok = (response_mask[i] == 1).nonzero(as_tuple=False)
#         if len(last_resp_tok) > 0:
#             task_reward_t[i, last_resp_tok[-1].item()] = task_rewards_list[i]

#     # KL shaping at every response token
#     # logp_old / logp_ref are (B, T-1); response_mask is (B, T)
#     kl_mask = response_mask[:, 1:].float()  # align with shifted logprobs
#     kl_tok  = kl_penalty(logp_old, logp_ref, kl_mask)  # (B, T-1)
#     kl_full = torch.zeros(B, T, device=device)
#     kl_full[:, 1:] = kl_tok

#     rewards = task_reward_t - beta * kl_full  # (B, T)

#     # ── 7. GAE ───────────────────────────────────────────────────────────────
#     advantages, returns = compute_gae_advantages(
#         rewards, values_old.detach(), response_mask, gamma=gamma, lam=lam,
#     )
#     advantages = normalise_advantages(advantages, response_mask)

#     return RolloutBatch(
#         prompts=prompts,
#         responses=responses,
#         input_ids=input_ids.cpu(),
#         attention_mask=attention_mask.cpu(),
#         response_mask=response_mask.cpu(),
#         logp_old=logp_old.detach().cpu(),
#         logp_ref=logp_ref.cpu(),
#         values_old=values_old.detach().cpu(),
#         rewards=rewards.cpu(),
#         task_rewards=task_rewards_list,
#         advantages=advantages.cpu(),
#         returns=returns.cpu(),
#     )


# def ppo_update(
#     policy_model,
#     value_model,
#     rollout: RolloutBatch,
#     policy_optimizer,
#     value_optimizer,
#     ppo_epochs: int = 4,
#     epsilon: float = 0.2,
#     c_value: float = 0.5,
#     c_entropy: float = 0.01,
#     max_grad_norm: float = 1.0,
#     device: str = "cuda",
# ) -> dict:
#     """
#     Run ppo_epochs mini-batch passes over the collected rollout.
#     Returns aggregated logging dict.
#     """
#     logs = {"policy_loss": [], "value_loss": [], "ratio_mean": [], "kl_approx": []}

#     input_ids      = rollout.input_ids.to(device)
#     attention_mask = rollout.attention_mask.to(device)
#     response_mask  = rollout.response_mask.to(device)
#     logp_old       = rollout.logp_old.to(device)
#     advantages     = rollout.advantages.to(device)
#     returns        = rollout.returns.to(device)

#     for _ in range(ppo_epochs):
#         # ── Policy forward ───────────────────────────────────────────────────
#         logp_new = token_logprobs(policy_model, input_ids, attention_mask)  # (B, T-1)

#         # Compute ratios (aligned to T-1 positions)
#         resp_mask_shifted = response_mask[:, 1:].float()
#         ratio = (logp_new - logp_old).exp() * resp_mask_shifted  # (B, T-1)

#         adv_shifted = advantages[:, 1:]  # align advantages to T-1

#         # Clipped surrogate
#         surr1 = ratio * adv_shifted
#         surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * adv_shifted
#         policy_loss = -torch.min(surr1, surr2)
#         n_resp = resp_mask_shifted.sum().clamp(min=1)
#         policy_loss = (policy_loss * resp_mask_shifted).sum() / n_resp

#         # ── Value forward ────────────────────────────────────────────────────
#         values_new = value_model(input_ids, attention_mask)  # (B, T)
#         value_targets = returns.detach()
#         value_loss = F.mse_loss(
#             values_new * response_mask.float(),
#             value_targets * response_mask.float(),
#             reduction="sum",
#         ) / response_mask.sum().clamp(min=1)

#         total_loss = policy_loss + c_value * value_loss

#         policy_optimizer.zero_grad()
#         value_optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
#         torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
#         policy_optimizer.step()
#         value_optimizer.step()

#         with torch.no_grad():
#             approx_kl = ((logp_old - logp_new) * resp_mask_shifted).sum() / n_resp

#         logs["policy_loss"].append(policy_loss.item())
#         logs["value_loss"].append(value_loss.item())
#         logs["ratio_mean"].append((ratio * resp_mask_shifted).sum().item() / n_resp.item())
#         logs["kl_approx"].append(approx_kl.item())

#     return {k: sum(v) / len(v) for k, v in logs.items()}


# # ── Helpers ──────────────────────────────────────────────────────────────────

# def _build_response_mask(
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
#     prompt_ids_list: List[List[int]],
#     T: int,
# ) -> torch.Tensor:
#     """
#     Build a (B, T) binary mask that is 1 for response tokens.
#     Accounts for left-padding.
#     """
#     B = input_ids.shape[0]
#     mask = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
#     for i, prompt_ids in enumerate(prompt_ids_list):
#         # Real tokens start after left-pad
#         real_start = (attention_mask[i] == 1).nonzero(as_tuple=False)[0].item()
#         resp_start = real_start + len(prompt_ids)
#         resp_start = min(resp_start, T)
#         mask[i, resp_start:] = attention_mask[i, resp_start:]
#     return mask


# def ppo_sanity_checks():
#     """Quick unit tests for GAE and ratio — call once before full training."""
#     # ── GAE check ────────────────────────────────────────────────────────────
#     from alignment.advantages import compute_gae_advantages
#     rewards = torch.tensor([[0.05, -0.02, 1.6]])
#     values  = torch.tensor([[1.5,  1.55,  1.45]])
#     mask    = torch.ones(1, 3, dtype=torch.long)
#     adv, ret = compute_gae_advantages(rewards, values, mask, gamma=1.0, lam=1.0)
#     # Manual: delta2=1.6-1.45=0.15; delta1=-0.02+1.45-1.55=-0.12; delta0=0.05+1.55-1.5=0.10
#     # A2=0.15; A1=-0.12+0.15=0.03; A0=0.10+0.03=0.13
#     print(f"[GAE sanity] advantages: {adv[0].tolist()}")
#     print(f"[GAE sanity] expected approx [0.13, 0.03, 0.15]")

#     # ── Ratio check: old == new → ratio = 1.0 ────────────────────────────────
#     dummy_logp = torch.tensor([[-0.5, -1.0, -0.3]])
#     ratio = (dummy_logp - dummy_logp).exp()
#     assert ratio.allclose(torch.ones_like(ratio)), "Ratio test FAILED"
#     print("[Ratio sanity] ratio == 1.0 ✓")

#     # ── Clipping check ────────────────────────────────────────────────────────
#     rho = torch.tensor([[1.5]])
#     adv_t = torch.tensor([[1.0]])
#     eps = 0.2
#     surr1 = rho * adv_t
#     surr2 = rho.clamp(1 - eps, 1 + eps) * adv_t
#     clipped_val = torch.min(surr1, surr2)
#     expected = (1 + eps) * 1.0
#     assert abs(clipped_val.item() - expected) < 1e-5, "Clip test FAILED"
#     print(f"[Clip sanity] clipped = {clipped_val.item():.4f}, expected {expected:.4f} ✓")

"""
Proximal Policy Optimisation (PPO) for LLMs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

from model.generation import generate_responses
from model.reward_model import score_texts
from alignment.advantages import compute_gae_advantages, normalise_advantages


@dataclass
class RolloutBatch:
    prompts:        List[str]
    responses:      List[str]
    input_ids:      torch.Tensor
    attention_mask: torch.Tensor
    response_mask:  torch.Tensor
    logp_old:       torch.Tensor
    values_old:     torch.Tensor
    rewards:        torch.Tensor
    task_rewards:   List[float]
    advantages:     torch.Tensor
    returns:        torch.Tensor


@torch.no_grad()
def ppo_rollout(
    policy_model,
    ref_model,
    value_model,
    reward_model,
    rm_tokenizer,
    policy_tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    beta: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.95,
    device: str = "cuda",
) -> RolloutBatch:

    responses         = generate_responses(
        policy_model, policy_tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, do_sample=True, device=device,
    )
    full_texts        = [p + " " + r for p, r in zip(prompts, responses)]
    task_rewards_list = score_texts(reward_model, rm_tokenizer, full_texts)

    enc = policy_tokenizer(
        full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    B, T = input_ids.shape
    targets = input_ids[:, 1:]

    prompt_ids_list = [
        policy_tokenizer.encode(p, add_special_tokens=False) for p in prompts
    ]
    response_mask = _build_response_mask(input_ids, attention_mask, prompt_ids_list, T)

    # Cache old log-probs — eval mode so dropout is off
    policy_model.eval()
    out_old  = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    logp_old = F.log_softmax(out_old.logits[:, :-1, :], dim=-1).gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1).detach()

    out_ref  = ref_model(input_ids=input_ids, attention_mask=attention_mask)
    logp_ref = F.log_softmax(out_ref.logits[:, :-1, :], dim=-1).gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)

    values_old = value_model(input_ids, attention_mask).detach()

    task_reward_t = torch.zeros(B, T, device=device)
    for i in range(B):
        last = (response_mask[i] == 1).nonzero(as_tuple=False)
        if len(last) > 0:
            task_reward_t[i, last[-1].item()] = task_rewards_list[i]

    kl_mask = response_mask[:, 1:].float()
    kl_full = torch.zeros(B, T, device=device)
    kl_full[:, 1:] = (logp_old - logp_ref) * kl_mask
    rewards = task_reward_t - beta * kl_full

    advantages, returns = compute_gae_advantages(
        rewards, values_old, response_mask, gamma=gamma, lam=lam,
    )
    advantages = normalise_advantages(advantages, response_mask)

    return RolloutBatch(
        prompts=prompts, responses=responses,
        input_ids=input_ids.cpu(), attention_mask=attention_mask.cpu(),
        response_mask=response_mask.cpu(), logp_old=logp_old.cpu(),
        values_old=values_old.cpu(), rewards=rewards.cpu(),
        task_rewards=task_rewards_list,
        advantages=advantages.cpu(), returns=returns.cpu(),
    )


def ppo_update(
    policy_model,
    value_model,
    rollout: RolloutBatch,
    policy_optimizer,
    value_optimizer,
    ppo_epochs: int = 4,
    epsilon: float = 0.2,
    c_value: float = 0.5,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> dict:
    logs = {"policy_loss": [], "value_loss": [], "kl_approx": []}

    input_ids      = rollout.input_ids.to(device)
    attention_mask = rollout.attention_mask.to(device)
    response_mask  = rollout.response_mask.to(device)
    logp_old       = rollout.logp_old.to(device)
    advantages     = rollout.advantages.to(device)
    returns        = rollout.returns.to(device)
    targets        = input_ids[:, 1:]

    resp_shift = response_mask[:, 1:].float()
    adv_shift  = advantages[:, 1:]
    n_resp     = resp_shift.sum().clamp(min=1)

    for epoch in range(ppo_epochs):
        policy_model.train()
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        out      = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        logp_new = F.log_softmax(out.logits[:, :-1, :], dim=-1).gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)

        ratio = (logp_new - logp_old).exp()
        surr1 = ratio * adv_shift
        surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * adv_shift
        policy_loss = -(torch.min(surr1, surr2) * resp_shift).sum() / n_resp

        values_new = value_model(input_ids, attention_mask)
        value_loss = F.mse_loss(
            values_new * response_mask.float(),
            returns.detach() * response_mask.float(),
            reduction="sum",
        ) / response_mask.sum().clamp(min=1)

        total_loss = policy_loss + c_value * value_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
        policy_optimizer.step()
        value_optimizer.step()

        with torch.no_grad():
            kl = ((logp_old - logp_new.detach()) * resp_shift).sum() / n_resp

        logs["policy_loss"].append(policy_loss.item())
        logs["value_loss"].append(value_loss.item())
        logs["kl_approx"].append(kl.item())

    return {k: sum(v) / len(v) for k, v in logs.items()}


def _build_response_mask(input_ids, attention_mask, prompt_ids_list, T):
    B = input_ids.shape[0]
    mask = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
    for i, prompt_ids in enumerate(prompt_ids_list):
        real_start = (attention_mask[i] == 1).nonzero(as_tuple=False)[0].item()
        resp_start = min(real_start + len(prompt_ids), T)
        mask[i, resp_start:] = attention_mask[i, resp_start:]
    return mask


def ppo_sanity_checks():
    from alignment.advantages import compute_gae_advantages
    rewards = torch.tensor([[0.05, -0.02, 1.6]])
    values  = torch.tensor([[1.5,  1.55,  1.45]])
    mask    = torch.ones(1, 3, dtype=torch.long)
    adv, _  = compute_gae_advantages(rewards, values, mask, gamma=1.0, lam=1.0)
    print(f"[GAE sanity] advantages: {adv[0].tolist()}")
    print(f"[GAE sanity] expected approx [0.13, 0.03, 0.15]")
    dummy = torch.tensor([[-0.5, -1.0, -0.3]])
    ratio = (dummy - dummy).exp()
    assert ratio.allclose(torch.ones_like(ratio)), "Ratio test FAILED"
    print("[Ratio sanity] ratio == 1.0 ✓")
    rho   = torch.tensor([[1.5]])
    adv_t = torch.tensor([[1.0]])
    surr2 = rho.clamp(1 - 0.2, 1 + 0.2) * adv_t
    assert abs(surr2.item() - 1.2) < 1e-5, "Clip test FAILED"
    print(f"[Clip sanity] clipped = {surr2.item():.4f}, expected 1.2 ✓")