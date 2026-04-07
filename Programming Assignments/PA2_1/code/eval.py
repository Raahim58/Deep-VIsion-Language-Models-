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
from utils.memory import release_cuda_memory
from utils.text import strip_special_tokens


# ── Manual C8 constants ───────────────────────────────────────────────────────
# Manual says 200 prompts but that's too heavy for A100 with 5 models.
# We use 50 for all evaluation passes (still statistically meaningful) and
# 5 for the sample response table.
EVAL_PROMPTS       = 50   # RM win-rate + KL measured over this many prompts
SAMPLE_TABLE_SIZE  = 5    # side-by-side response table size (manual says 5)
EVAL_BATCH_SIZE    = 4    # generation/scoring batch size (OOM-safe on A100)


def _load_run_summary(checkpoint: str) -> dict[str, Any]:
    summary_path = Path(checkpoint) / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def _batched(items: list, batch_size: int):
    batch_size = max(1, int(batch_size))
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# Core building blocks
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_batched(model, tokenizer, prompts: list[str], max_new_tokens: int, label: str) -> list[str]:
    """Generate completions in small batches with progress prints."""
    outputs: list[str] = []
    total = max(1, (len(prompts) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE)
    for i, batch in enumerate(_batched(prompts, EVAL_BATCH_SIZE), 1):
        print(f"  [{label}] gen batch {i}/{total} ({len(batch)} prompts)", flush=True)
        outputs.extend(
            generate_completions(model, tokenizer, batch, max_new_tokens=max_new_tokens, do_sample=False)
        )
    return outputs


@torch.no_grad()
def _score_batched(reward_fn, prompts: list[str], outputs: list[str], label: str) -> torch.Tensor:
    """Score completions in small batches with progress prints."""
    scores: list[torch.Tensor] = []
    total = max(1, (len(prompts) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE)
    for i, (pb, ob) in enumerate(
        zip(_batched(prompts, EVAL_BATCH_SIZE), _batched(outputs, EVAL_BATCH_SIZE)), 1
    ):
        print(f"  [{label}] reward batch {i}/{total}", flush=True)
        scores.append(reward_fn(pb, ob).detach().cpu())
    return torch.cat(scores) if scores else torch.empty(0)


@torch.no_grad()
def _kl_batched(
    model, reference, tokenizer,
    prompts: list[str], outputs: list[str],
    max_seq_len: int, label: str,
) -> float:
    """Compute mean KL(policy || reference) via Monte Carlo over sampled tokens."""
    device = next(model.parameters()).device
    total_kl, total_n = 0.0, 0
    total = max(1, (len(prompts) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE)
    for i, (pb, ob) in enumerate(
        zip(_batched(prompts, EVAL_BATCH_SIZE), _batched(outputs, EVAL_BATCH_SIZE)), 1
    ):
        print(f"  [{label}] KL batch {i}/{total}", flush=True)
        rollout = build_rollout_batch(tokenizer, pb, ob, max_seq_len)
        rollout = {k: v.to(device) for k, v in rollout.items()}
        policy_lp, mask = response_token_logprobs(
            model, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"]
        )
        ref_lp, _ = response_token_logprobs(
            reference, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"]
        )
        kl = mean_masked_kl(policy_lp, ref_lp, mask).item()
        total_kl += kl * len(pb)
        total_n  += len(pb)
        release_cuda_memory(rollout, policy_lp, ref_lp, mask)
    return total_kl / max(1, total_n)


# ─────────────────────────────────────────────────────────────────────────────
# Task C8 Metric 1 + 2: RM win-rate vs SFT and KL from π_ref
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_candidate_vs_reference(
    config: dict, candidate_checkpoint: str, reference_checkpoint: str
) -> dict:
    """
    C8 Metrics:
      1. RM win-rate vs SFT baseline
      2. KL(π_θ || π_ref) via Monte Carlo over EVAL_PROMPTS prompts
      3. Sample response table (SAMPLE_TABLE_SIZE entries)
    """
    print(f"\n[Eval] candidate = {candidate_checkpoint}", flush=True)
    print(f"[Eval] reference = {reference_checkpoint}", flush=True)

    # ── load shared resources ─────────────────────────────────────────────────
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    reference = load_reference_model(config, checkpoint=reference_checkpoint)
    reward_model = load_reward_model(
        config, checkpoint=config["models"].get("rm_checkpoint"), trainable=False
    )
    reward_model.config.pad_token_id = int(reward_tokenizer.pad_token_id)
    reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])
    candidate = load_policy_model(config, checkpoint=candidate_checkpoint, trainable=False)

    # ── prompts ───────────────────────────────────────────────────────────────
    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_eval_split"], EVAL_PROMPTS)
    )
    prompts = [row["prompt"] for row in prompts_dataset]
    print(f"[Eval] {len(prompts)} held-out prompts", flush=True)

    max_new = config["evaluation"]["max_new_tokens"]

    # ── generate ──────────────────────────────────────────────────────────────
    print("[Eval] Generating candidate completions...", flush=True)
    cand_outputs = _generate_batched(candidate, policy_tokenizer, prompts, max_new, "candidate")
    print("[Eval] Generating reference (SFT) completions...", flush=True)
    ref_outputs  = _generate_batched(reference, policy_tokenizer, prompts, max_new, "reference")

    # ── reward scoring ────────────────────────────────────────────────────────
    print("[Eval] Scoring candidate with reward model...", flush=True)
    cand_scores = _score_batched(reward_fn, prompts, cand_outputs, "candidate")
    print("[Eval] Scoring reference with reward model...", flush=True)
    ref_scores  = _score_batched(reward_fn, prompts, ref_outputs,  "reference")

    # ── C8 Metric 1: RM win-rate ──────────────────────────────────────────────
    win_rate = float((cand_scores > ref_scores).float().mean().item())
    print(f"[Eval] RM win-rate vs SFT: {win_rate:.4f}", flush=True)

    # ── C8 Metric 2: KL(π_θ || π_ref) ────────────────────────────────────────
    print("[Eval] Computing KL from reference...", flush=True)
    mean_kl = _kl_batched(
        candidate, reference, policy_tokenizer,
        prompts, cand_outputs, config["max_seq_len"], "kl"
    )
    print(f"[Eval] Mean KL from π_ref: {mean_kl:.4f}", flush=True)

    # ── C8 Metric 3: sample response table ───────────────────────────────────
    sample_rows = []
    for p, c, r, cs, rs in zip(
        prompts[:SAMPLE_TABLE_SIZE],
        cand_outputs[:SAMPLE_TABLE_SIZE],
        ref_outputs[:SAMPLE_TABLE_SIZE],
        cand_scores[:SAMPLE_TABLE_SIZE].tolist(),
        ref_scores[:SAMPLE_TABLE_SIZE].tolist(),
    ):
        sample_rows.append({
            "prompt":            strip_special_tokens(p),
            "candidate":         strip_special_tokens(c),
            "candidate_rm_score": cs,
            "reference":         strip_special_tokens(r),
            "reference_rm_score": rs,
        })

    result = {
        "reward_model_win_rate_vs_sft": win_rate,
        "mean_sampled_token_kl":        mean_kl,
        "mean_candidate_rm_score":      float(cand_scores.mean().item()),
        "mean_reference_rm_score":      float(ref_scores.mean().item()),
        "sample_rows":                  sample_rows,
    }

    # ── optional DPO preference accuracy ─────────────────────────────────────
    pref_acc = _pairwise_preference_accuracy(config, candidate_checkpoint, policy_tokenizer, reference)
    result["heldout_preference_accuracy"] = pref_acc
    print(f"[Eval] Preference accuracy: {pref_acc:.4f}", flush=True)

    release_cuda_memory(candidate, reference, reward_model)
    return result


def _pairwise_preference_accuracy(
    config: dict, policy_checkpoint: str, policy_tokenizer, reference
) -> float:
    """Fraction of held-out pairs where policy scores chosen > rejected."""
    policy = load_policy_model(config, checkpoint=policy_checkpoint, trainable=False)
    dataset = make_dpo_dataset(
        load_hh_dataset(config, config["data"]["hh_eval_split"], EVAL_PROMPTS)
    )
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=DPOCollator(policy_tokenizer, config["max_seq_len"]),
        num_workers=config["num_workers"],
    )
    values: list[float] = []
    total = len(loader)
    print(f"  [pref_acc] {total} batches", flush=True)
    for i, batch in enumerate(loader, 1):
        with torch.no_grad():
            m = dpo_batch_metrics(policy, reference, batch, config["dpo"]["beta"])
        values.append(float(m["preference_accuracy"].item()))
        if i % max(1, total // 5) == 0:
            print(f"  [pref_acc] batch {i}/{total}", flush=True)
    release_cuda_memory(policy, loader)
    return sum(values) / max(1, len(values))


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K pass@1 (for RLVR)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gsm8k_pass_at_1(config: dict, candidate_checkpoint: str) -> dict:
    print(f"\n[Eval:GSM8K] candidate = {candidate_checkpoint}", flush=True)
    dataset = load_gsm8k_dataset(config, "test", config["data"]["gsm_eval_samples"])
    prompts = [row["prompt"] for row in dataset]
    gold    = [row["answer"] for row in dataset]

    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    candidate = load_policy_model(config, checkpoint=candidate_checkpoint, trainable=False)

    print(f"[Eval:GSM8K] {len(prompts)} test prompts", flush=True)
    outputs = _generate_batched(
        candidate, policy_tokenizer, prompts,
        config["rlvr"]["max_new_tokens"], "gsm8k"
    )
    pass_at_1 = sum(
        gsm8k_exact_match(pred, tgt) for pred, tgt in zip(outputs, gold)
    ) / max(1, len(gold))
    print(f"[Eval:GSM8K] pass@1 = {pass_at_1:.4f}", flush=True)
    release_cuda_memory(candidate)
    return {"gsm8k_pass_at_1": float(pass_at_1)}


# ─────────────────────────────────────────────────────────────────────────────
# C8 full comparison: SFT vs PPO vs DPO vs GRPO (and optionally RLVR)
# Manual: Table with RM win-rate, KL, resource usage for each method
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_method_comparison(
    config: dict,
    method_checkpoints: dict[str, str],
    reference_checkpoint: str,
    include_gsm8k: bool = False,
) -> dict[str, Any]:
    """
    C8 required outputs:
      - metrics table: method, rm_win_rate_vs_sft, mean_kl_from_sft, mean_rm_score, heldout_pref_acc
      - resource table: method, peak_vram_gb, mean_step_time_sec, total_training_time_sec
      - sample table: SAMPLE_TABLE_SIZE prompts × all methods side-by-side with RM scores
    """
    print(f"\n[Eval:comparison] methods = {list(method_checkpoints.keys())}", flush=True)

    # ── shared resources ──────────────────────────────────────────────────────
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    reference = load_reference_model(config, checkpoint=reference_checkpoint)
    reward_model = load_reward_model(
        config, checkpoint=config["models"].get("rm_checkpoint"), trainable=False
    )
    reward_model.config.pad_token_id = int(reward_tokenizer.pad_token_id)
    reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])

    # ── held-out prompts ──────────────────────────────────────────────────────
    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_eval_split"], EVAL_PROMPTS)
    )
    prompts = [row["prompt"] for row in prompts_dataset]
    print(f"[Eval:comparison] {len(prompts)} prompts", flush=True)

    max_new = config["evaluation"]["max_new_tokens"]
    outputs_by_method: dict[str, list[str]] = {}
    scores_by_method:  dict[str, list[float]] = {}
    metrics_rows:  list[dict] = []
    resource_rows: list[dict] = []

    # ── SFT (reference) baseline ──────────────────────────────────────────────
    print("\n[Eval:comparison] ── SFT baseline ──", flush=True)
    ref_outputs = _generate_batched(reference, policy_tokenizer, prompts, max_new, "sft")
    ref_scores  = _score_batched(reward_fn, prompts, ref_outputs, "sft")
    outputs_by_method["sft"] = ref_outputs
    scores_by_method["sft"]  = ref_scores.tolist()
    sft_summary = _load_run_summary(reference_checkpoint)
    resource_rows.append({
        "method": "sft",
        "checkpoint": reference_checkpoint,
        "peak_vram_gb":          sft_summary.get("peak_vram_gb"),
        "mean_step_time_sec":    sft_summary.get("mean_step_time_sec"),
        "total_training_time_sec": sft_summary.get("total_training_time_sec"),
    })
    print(f"  SFT mean RM score: {ref_scores.mean().item():.4f}", flush=True)

    # ── each aligned method ───────────────────────────────────────────────────
    for name, checkpoint in method_checkpoints.items():
        print(f"\n[Eval:comparison] ── {name.upper()} ──", flush=True)
        model = load_policy_model(config, checkpoint=checkpoint, trainable=False)

        outputs = _generate_batched(model, policy_tokenizer, prompts, max_new, name)
        scores  = _score_batched(reward_fn, prompts, outputs, name)

        win_rate = float((scores > ref_scores).float().mean().item())
        mean_kl  = _kl_batched(
            model, reference, policy_tokenizer,
            prompts, outputs, config["max_seq_len"], f"{name}_kl"
        )
        print(f"  {name} win-rate vs SFT: {win_rate:.4f}", flush=True)
        print(f"  {name} mean KL from SFT: {mean_kl:.4f}", flush=True)
        print(f"  {name} mean RM score: {scores.mean().item():.4f}", flush=True)

        pref_acc = _pairwise_preference_accuracy(
            config, checkpoint, policy_tokenizer, reference
        )
        print(f"  {name} preference accuracy: {pref_acc:.4f}", flush=True)

        row = {
            "method":                       name,
            "checkpoint":                   checkpoint,
            "rm_win_rate_vs_sft":           win_rate,
            "mean_rm_score":                float(scores.mean().item()),
            "mean_kl_from_sft":             mean_kl,
            "heldout_preference_accuracy":  pref_acc,
        }
        if include_gsm8k and name.lower() == "rlvr":
            row.update(evaluate_gsm8k_pass_at_1(config, checkpoint))

        metrics_rows.append(row)
        outputs_by_method[name] = outputs
        scores_by_method[name]  = scores.tolist()

        summary = _load_run_summary(checkpoint)
        resource_rows.append({
            "method": name,
            "checkpoint": checkpoint,
            "peak_vram_gb":          summary.get("peak_vram_gb"),
            "mean_step_time_sec":    summary.get("mean_step_time_sec"),
            "total_training_time_sec": summary.get("total_training_time_sec"),
        })
        release_cuda_memory(model)

    # ── C8 sample response table ──────────────────────────────────────────────
    # Manual: 5 prompts, show SFT + each method, flag ≥2 where methods disagree
    sample_rows = []
    for idx in range(min(SAMPLE_TABLE_SIZE, len(prompts))):
        row: dict = {"prompt": strip_special_tokens(prompts[idx])}
        for method_name in ["sft", *method_checkpoints.keys()]:
            row[f"{method_name}_response"] = strip_special_tokens(
                outputs_by_method[method_name][idx]
            )
            row[f"{method_name}_rm_score"] = scores_by_method[method_name][idx]
        sample_rows.append(row)

    # Print sample table inline
    print("\n[Eval:comparison] ══ Sample Response Table ══", flush=True)
    for idx, row in enumerate(sample_rows):
        print(f"\n  Prompt {idx+1}: {row['prompt'][:120]}...", flush=True)
        for method_name in ["sft", *method_checkpoints.keys()]:
            resp  = row[f"{method_name}_response"][:120]
            score = row[f"{method_name}_rm_score"]
            print(f"    [{method_name:>6}] RM={score:+.3f}  {resp}", flush=True)

    # Print metrics table inline
    print("\n[Eval:comparison] ══ Metrics Table ══", flush=True)
    metrics_df = pd.DataFrame(metrics_rows)
    print(metrics_df.to_string(index=False), flush=True)

    print("\n[Eval:comparison] ══ Resource Table ══", flush=True)
    resource_df = pd.DataFrame(resource_rows)
    print(resource_df[["method", "peak_vram_gb", "mean_step_time_sec",
                        "total_training_time_sec"]].to_string(index=False), flush=True)

    release_cuda_memory(reference, reward_model)
    return {
        "metrics":      metrics_rows,
        "sample_rows":  sample_rows,
        "resource_rows": resource_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--candidate", action="append", default=[],
                        help="name=checkpoint for comparison mode")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-gsm8k", action="store_true")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.reference_checkpoint) / "eval"
    ensure_dir(out_dir)
    print(f"[Eval CLI] out_dir = {out_dir}", flush=True)

    if args.candidate:
        candidates = dict(item.split("=", 1) for item in args.candidate)
        result = evaluate_method_comparison(
            config, candidates, args.reference_checkpoint,
            include_gsm8k=args.include_gsm8k,
        )
        save_json(out_dir / "comparison_metrics.json", {
            "metrics": result["metrics"],
            "resource_rows": result["resource_rows"],
        })
        pd.DataFrame(result["metrics"]).to_csv(out_dir / "comparison_metrics.csv", index=False)
        pd.DataFrame(result["resource_rows"]).to_csv(out_dir / "resource_table.csv", index=False)
        pd.DataFrame(result["sample_rows"]).to_csv(out_dir / "comparison_samples.csv", index=False)
        print("[Eval CLI] comparison complete.", flush=True)
        return

    if not args.policy_checkpoint:
        raise ValueError(
            "Provide --policy-checkpoint for single-model eval or --candidate for comparison."
        )
    result = evaluate_candidate_vs_reference(
        config, args.policy_checkpoint, args.reference_checkpoint
    )
    if args.include_gsm8k:
        result.update(evaluate_gsm8k_pass_at_1(config, args.policy_checkpoint))

    save_json(out_dir / "eval_results.json",
              {k: v for k, v in result.items() if k != "sample_rows"})
    pd.DataFrame(result["sample_rows"]).to_csv(out_dir / "sample_generations.csv", index=False)
    print("[Eval CLI] single-model evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
