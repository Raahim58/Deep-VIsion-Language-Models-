"""
RLVR — Reinforcement Learning with Verifiable Rewards.
Uses GRPO update logic with a binary verifiable reward instead of a learned RM.
"""
from __future__ import annotations

import torch
from typing import List, Optional

from data.parsing import extract_gsm8k_answer, answers_match
from alignment.grpo import grpo_rollout, grpo_update


def rlvr_reward(
    completions: List[str],
    gold_answers: List[float],
) -> List[float]:
    """
    Binary verifiable reward for GSM8K.
    Returns 1.0 if the extracted answer matches the gold, 0.0 otherwise.
    Completions and gold_answers must be in the same order.
    """
    rewards = []
    for text, gold in zip(completions, gold_answers):
        pred = extract_gsm8k_answer(text)
        rewards.append(1.0 if answers_match(pred, gold) else 0.0)
    return rewards


def make_rlvr_reward_fn(gold_answers_flat: List[float]):
    """
    Returns a reward_fn(texts) -> List[float] closure suitable for grpo_rollout.
    gold_answers_flat must be pre-ordered to match the flat B*K texts list.
    """
    def _fn(texts: List[str]) -> List[float]:
        assert len(texts) == len(gold_answers_flat), (
            f"Expected {len(gold_answers_flat)} texts, got {len(texts)}"
        )
        return rlvr_reward(texts, gold_answers_flat)
    return _fn


def rlvr_rollout(
    policy_model,
    ref_model,
    policy_tokenizer,
    prompts: List[str],
    gold_answers: List[float],
    K: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """
    GRPO rollout with verifiable reward.
    gold_answers[i] is the correct answer for prompts[i].
    """
    B = len(prompts)
    # Expand gold answers for B*K completions
    gold_flat = []
    for g in gold_answers:
        gold_flat.extend([g] * K)

    reward_fn = make_rlvr_reward_fn(gold_flat)

    return grpo_rollout(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        policy_tokenizer=policy_tokenizer,
        prompts=prompts,
        K=K,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )


@torch.no_grad()
def eval_gsm8k_pass_at_1(
    policy_model,
    policy_tokenizer,
    items: List[dict],
    max_new_tokens: int = 256,
    device: str = "cuda",
    max_samples: int = 200,
) -> float:
    """
    Evaluate pass@1 on GSM8K test set using greedy decoding.
    Returns fraction of correct answers.
    """
    from model.generation import generate_responses
    items = items[:max_samples]
    prompts = [it["prompt"] for it in items]
    golds   = [it["gold_answer"] for it in items]

    # Greedy (no sampling)
    responses = generate_responses(
        policy_model, policy_tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False, device=device,
    )

    correct = 0
    for resp, gold in zip(responses, golds):
        pred = extract_gsm8k_answer(resp)
        if answers_match(pred, gold):
            correct += 1
    return correct / len(items)
