from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from alignment.grpo import build_rm_reward_fn
from alignment.kl import mean_masked_kl
from alignment.ppo import build_rollout_batch
from data.gsm8k import load_gsm8k_dataset
from data.hh_rlhf import load_hh_dataset, make_prompt_dataset
from model.generation import generate_completions
from model.loading import (
    load_policy_model,
    load_policy_tokenizer,
    load_reference_model,
    load_reward_model,
    load_reward_tokenizer,
)
from model.logprobs import response_token_logprobs
from utils.config import load_config
from utils.io import ensure_dir, save_json
from utils.metrics import gsm8k_exact_match
from utils.text import strip_special_tokens


def evaluate_candidate_vs_reference(config: dict, candidate_checkpoint: str, reference_checkpoint: str) -> dict:
    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_eval_split"], config["evaluation"]["prompts"])
    )
    prompts = [row["prompt"] for row in prompts_dataset]

    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    candidate = load_policy_model(config, checkpoint=candidate_checkpoint, trainable=False)
    reference = load_reference_model(config, checkpoint=reference_checkpoint)
    reward_model = load_reward_model(config, checkpoint=config["models"].get("rm_checkpoint"), trainable=False)
    reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])

    candidate_outputs = generate_completions(
        candidate,
        policy_tokenizer,
        prompts,
        max_new_tokens=config["evaluation"]["max_new_tokens"],
        do_sample=False,
    )
    reference_outputs = generate_completions(
        reference,
        policy_tokenizer,
        prompts,
        max_new_tokens=config["evaluation"]["max_new_tokens"],
        do_sample=False,
    )

    candidate_rewards = reward_fn(prompts, candidate_outputs)
    reference_rewards = reward_fn(prompts, reference_outputs)
    win_rate = float((candidate_rewards > reference_rewards).float().mean().item())

    rollout = build_rollout_batch(policy_tokenizer, prompts, candidate_outputs, config["max_seq_len"])
    device = next(candidate.parameters()).device
    rollout = {key: value.to(device) for key, value in rollout.items()}
    candidate_logprobs, valid_mask = response_token_logprobs(
        candidate, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"]
    )
    reference_logprobs, _ = response_token_logprobs(
        reference, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"]
    )
    mean_kl = float(mean_masked_kl(candidate_logprobs, reference_logprobs, valid_mask).item())

    sample_rows = []
    for prompt, cand, ref in zip(prompts[: config["evaluation"]["sample_table_size"]], candidate_outputs, reference_outputs):
        sample_rows.append(
            {
                "prompt": strip_special_tokens(prompt),
                "candidate": strip_special_tokens(cand),
                "reference": strip_special_tokens(ref),
            }
        )

    return {
        "reward_model_win_rate_vs_sft": win_rate,
        "mean_sampled_token_kl": mean_kl,
        "sample_rows": sample_rows,
    }


def evaluate_gsm8k_pass_at_1(config: dict, candidate_checkpoint: str) -> dict:
    dataset = load_gsm8k_dataset(config, "test", config["data"]["gsm_eval_samples"])
    prompts = [row["prompt"] for row in dataset]
    gold = [row["answer"] for row in dataset]

    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    candidate = load_policy_model(config, checkpoint=candidate_checkpoint, trainable=False)
    outputs = generate_completions(
        candidate,
        policy_tokenizer,
        prompts,
        max_new_tokens=config["rlvr"]["max_new_tokens"],
        do_sample=False,
    )
    pass_at_1 = sum(gsm8k_exact_match(pred, target) for pred, target in zip(outputs, gold)) / max(1, len(gold))
    return {"gsm8k_pass_at_1": float(pass_at_1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-gsm8k", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = evaluate_candidate_vs_reference(config, args.policy_checkpoint, args.reference_checkpoint)
    if args.include_gsm8k:
        result.update(evaluate_gsm8k_pass_at_1(config, args.policy_checkpoint))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.policy_checkpoint) / "eval"
    ensure_dir(out_dir)
    save_json(out_dir / "eval_results.json", {k: v for k, v in result.items() if k != "sample_rows"})
    pd.DataFrame(result["sample_rows"]).to_csv(out_dir / "sample_generations.csv", index=False)


if __name__ == "__main__":
    main()
