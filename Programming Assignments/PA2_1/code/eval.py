from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from alignment.dpo import dpo_batch_metrics
from alignment.grpo import build_rm_reward_fn
from alignment.kl import mean_masked_kl
from alignment.ppo import build_rollout_batch
from data.collators import DPOCollator
from data.gsm8k import load_gsm8k_dataset
from data.hh_rlhf import load_hh_dataset, make_dpo_dataset, make_prompt_dataset
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


def _load_eval_context(config: dict[str, Any], reference_checkpoint: str):
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    reference = load_reference_model(config, checkpoint=reference_checkpoint)
    reward_model = load_reward_model(config, checkpoint=config["models"].get("rm_checkpoint"), trainable=False)
    reward_model.config.pad_token_id = int(reward_tokenizer.pad_token_id)
    reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])
    return policy_tokenizer, reward_tokenizer, reference, reward_model, reward_fn


def _load_run_summary(checkpoint: str) -> dict[str, Any]:
    summary_path = Path(checkpoint) / 'summary.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding='utf-8'))
    return {}


def evaluate_pairwise_preference_accuracy(config: dict[str, Any], policy_checkpoint: str) -> dict[str, float]:
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    policy = load_policy_model(config, checkpoint=policy_checkpoint, trainable=False)
    reference_checkpoint = config["models"].get("sft_checkpoint") or config["models"].get("policy_checkpoint") or policy_checkpoint
    reference = load_reference_model(config, checkpoint=reference_checkpoint)
    dataset = make_dpo_dataset(load_hh_dataset(config, config["data"]["hh_eval_split"], config["evaluation"]["prompts"]))
    loader = DataLoader(
        dataset,
        batch_size=config["dpo"]["batch_size"],
        shuffle=False,
        collate_fn=DPOCollator(policy_tokenizer, config["max_seq_len"]),
        num_workers=config["num_workers"],
    )
    values = []
    for batch in loader:
        metrics = dpo_batch_metrics(policy, reference, batch, config["dpo"]["beta"])
        values.append(float(metrics["preference_accuracy"].item()))
    return {"heldout_preference_accuracy": sum(values) / max(1, len(values))}


def evaluate_candidate_vs_reference(config: dict, candidate_checkpoint: str, reference_checkpoint: str) -> dict:
    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_eval_split"], config["evaluation"]["prompts"])
    )
    prompts = [row["prompt"] for row in prompts_dataset]

    policy_tokenizer, reward_tokenizer, reference, reward_model, reward_fn = _load_eval_context(config, reference_checkpoint)
    candidate = load_policy_model(config, checkpoint=candidate_checkpoint, trainable=False)

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
    for prompt, cand, ref, cand_score, ref_score in zip(
        prompts[: config["evaluation"]["sample_table_size"]],
        candidate_outputs,
        reference_outputs,
        candidate_rewards.tolist(),
        reference_rewards.tolist(),
    ):
        sample_rows.append(
            {
                "prompt": strip_special_tokens(prompt),
                "candidate": strip_special_tokens(cand),
                "candidate_rm_score": cand_score,
                "reference": strip_special_tokens(ref),
                "reference_rm_score": ref_score,
            }
        )

    result = {
        "reward_model_win_rate_vs_sft": win_rate,
        "mean_sampled_token_kl": mean_kl,
        "mean_candidate_rm_score": float(candidate_rewards.mean().item()),
        "mean_reference_rm_score": float(reference_rewards.mean().item()),
        "sample_rows": sample_rows,
    }
    result.update(evaluate_pairwise_preference_accuracy(config, candidate_checkpoint))
    return result


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


def evaluate_method_comparison(
    config: dict[str, Any],
    method_checkpoints: dict[str, str],
    reference_checkpoint: str,
    include_gsm8k: bool = False,
) -> dict[str, Any]:
    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_eval_split"], config["evaluation"]["prompts"])
    )
    prompts = [row["prompt"] for row in prompts_dataset]
    policy_tokenizer, reward_tokenizer, reference, reward_model, reward_fn = _load_eval_context(config, reference_checkpoint)

    outputs_by_method: dict[str, list[str]] = {}
    scores_by_method: dict[str, list[float]] = {}
    metrics_rows = []
    resource_rows = []

    reference_outputs = generate_completions(
        reference,
        policy_tokenizer,
        prompts,
        max_new_tokens=config["evaluation"]["max_new_tokens"],
        do_sample=False,
    )
    reference_scores = reward_fn(prompts, reference_outputs)
    outputs_by_method['sft'] = reference_outputs
    scores_by_method['sft'] = reference_scores.tolist()
    resource_rows.append(
        {
            "method": "sft",
            "checkpoint": reference_checkpoint,
            **{
                "peak_vram_gb": _load_run_summary(reference_checkpoint).get("peak_vram_gb"),
                "mean_step_time_sec": _load_run_summary(reference_checkpoint).get("mean_step_time_sec"),
                "total_training_time_sec": _load_run_summary(reference_checkpoint).get("total_training_time_sec"),
            },
        }
    )
    reference_rollout = build_rollout_batch(policy_tokenizer, prompts, reference_outputs, config["max_seq_len"])
    reference_device = next(reference.parameters()).device
    reference_rollout = {key: value.to(reference_device) for key, value in reference_rollout.items()}

    for name, checkpoint in method_checkpoints.items():
        model = load_policy_model(config, checkpoint=checkpoint, trainable=False)
        outputs = generate_completions(
            model,
            policy_tokenizer,
            prompts,
            max_new_tokens=config["evaluation"]["max_new_tokens"],
            do_sample=False,
        )
        scores = reward_fn(prompts, outputs)
        rollout = build_rollout_batch(policy_tokenizer, prompts, outputs, config["max_seq_len"])
        device = next(model.parameters()).device
        rollout = {key: value.to(device) for key, value in rollout.items()}
        policy_logprobs, valid_mask = response_token_logprobs(model, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"])
        ref_logprobs, _ = response_token_logprobs(reference, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"])
        outputs_by_method[name] = outputs
        scores_by_method[name] = scores.tolist()
        row = {
            "method": name,
            "checkpoint": checkpoint,
            "rm_win_rate_vs_sft": float((scores > reference_scores).float().mean().item()),
            "mean_rm_score": float(scores.mean().item()),
            "mean_kl_from_sft": float(mean_masked_kl(policy_logprobs, ref_logprobs, valid_mask).item()),
        }
        row.update(evaluate_pairwise_preference_accuracy(config, checkpoint))
        if include_gsm8k and name.lower() == 'rlvr':
            row.update(evaluate_gsm8k_pass_at_1(config, checkpoint))
        metrics_rows.append(row)

        summary = _load_run_summary(checkpoint)
        resource_rows.append(
            {
                "method": name,
                "checkpoint": checkpoint,
                "peak_vram_gb": summary.get("peak_vram_gb"),
                "mean_step_time_sec": summary.get("mean_step_time_sec"),
                "total_training_time_sec": summary.get("total_training_time_sec"),
            }
        )

    sample_rows = []
    sample_count = config["evaluation"]["sample_table_size"]
    for idx in range(min(sample_count, len(prompts))):
        row = {"prompt": strip_special_tokens(prompts[idx])}
        for name in ["sft", *method_checkpoints.keys()]:
            row[f"{name}_response"] = strip_special_tokens(outputs_by_method[name][idx])
            row[f"{name}_rm_score"] = scores_by_method[name][idx]
        sample_rows.append(row)

    return {
        "metrics": metrics_rows,
        "sample_rows": sample_rows,
        "resource_rows": resource_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=checkpoint for comparison mode")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-gsm8k", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.reference_checkpoint) / "eval"
    ensure_dir(out_dir)

    if args.candidate:
        candidates = dict(item.split("=", 1) for item in args.candidate)
        result = evaluate_method_comparison(config, candidates, args.reference_checkpoint, include_gsm8k=args.include_gsm8k)
        save_json(out_dir / "comparison_metrics.json", {"metrics": result["metrics"], "resource_rows": result["resource_rows"]})
        pd.DataFrame(result["metrics"]).to_csv(out_dir / "comparison_metrics.csv", index=False)
        pd.DataFrame(result["resource_rows"]).to_csv(out_dir / "resource_table.csv", index=False)
        pd.DataFrame(result["sample_rows"]).to_csv(out_dir / "comparison_samples.csv", index=False)
        return

    if not args.policy_checkpoint:
        raise ValueError("Provide --policy-checkpoint for single-model evaluation or --candidate for comparison mode.")

    result = evaluate_candidate_vs_reference(config, args.policy_checkpoint, args.reference_checkpoint)
    if args.include_gsm8k:
        result.update(evaluate_gsm8k_pass_at_1(config, args.policy_checkpoint))

    save_json(out_dir / "eval_results.json", {k: v for k, v in result.items() if k != "sample_rows"})
    pd.DataFrame(result["sample_rows"]).to_csv(out_dir / "sample_generations.csv", index=False)


if __name__ == "__main__":
    main()
