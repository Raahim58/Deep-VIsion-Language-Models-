from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from eval import evaluate_candidate_vs_reference
from train_rl import run_dpo, run_grpo, run_ppo
from utils.config import deep_merge_dicts, load_config
from utils.io import ensure_dir, make_run_dir, save_json
from utils.plotting import plot_metric_curves


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_metrics_frame(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _run_variant(config: dict, ablation: str) -> dict:
    """Dispatch to the right training function based on the ablation type."""
    if ablation in {"kl_sweep", "clip_sweep"}:
        method = config.get("ablation", {}).get("method", "ppo")
        if method == "ppo":
            return run_ppo(config)
        if method == "grpo":
            return run_grpo(config)
        raise ValueError("kl_sweep/clip_sweep require method=ppo or method=grpo")
    if ablation == "k_sweep":
        return run_grpo(config)
    if ablation == "dpo_beta_sweep":
        return run_dpo(config)
    raise ValueError(f"Unsupported ablation: {ablation}")




def _save_variant_artifacts(run_result: dict, metrics: dict) -> None:
    run_dir = Path(run_result["run_dir"])
    eval_dir = ensure_dir(run_dir / "eval")
    scalar_metrics = {k: v for k, v in metrics.items() if k != "sample_rows"}
    save_json(eval_dir / "eval_results.json", scalar_metrics)
    if "sample_rows" in metrics:
        pd.DataFrame(metrics["sample_rows"]).to_csv(eval_dir / "sample_generations.csv", index=False)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        plot_metric_curves(metrics_path, run_dir / "metrics.png")

def _find_default_yaml() -> Path | None:
    """Walk up from cwd to find configs/default.yaml."""
    search = Path(os.getcwd()).resolve()
    for _ in range(5):
        candidate = search / "configs" / "default.yaml"
        if candidate.exists():
            return candidate
        search = search.parent
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single focused ablation sweep."
    )
    parser.add_argument("--config", action="append", required=True,
                        help="YAML config path(s); repeated. The runtime config written by the "
                             "notebook is sufficient — default.yaml is auto-merged as a base.")
    parser.add_argument(
        "--ablation",
        choices=["kl_sweep", "clip_sweep", "k_sweep", "dpo_beta_sweep"],
        required=True,
    )
    parser.add_argument("--method", choices=["ppo", "grpo"], default=None,
                        help="Required for kl_sweep / clip_sweep.")
    args = parser.parse_args()

    # ── build config: always start from default.yaml so no key is missing ────
    default_yaml = _find_default_yaml()
    base_paths: list[str] = []
    if default_yaml is not None:
        base_paths.append(str(default_yaml))
    base_paths.extend(args.config)   # user-supplied configs override defaults

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in base_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    base_config = load_config(unique_paths)
    base_config.setdefault("ablation", {})
    if args.method is not None:
        base_config["ablation"]["method"] = args.method

    reference_checkpoint = (
        base_config["models"].get("sft_checkpoint")
        or base_config["models"].get("policy_checkpoint")
    )
    if not reference_checkpoint:
        print(
            "ERROR: sft_checkpoint is not set in the config. "
            "Run SFT first and make sure the checkpoint path is written to the runtime config.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_dir = make_run_dir(base_config["output_dir"], f"ablation_{args.ablation}")
    results: list[dict] = []

    # ── ablation sweeps ───────────────────────────────────────────────────────

    if args.ablation == "kl_sweep":
        method = base_config["ablation"].get("method", "ppo")
        for beta in [0.0, 0.05, 0.1, 0.5]:
            config = deepcopy(base_config)
            config[method]["beta"] = beta
            result  = _run_variant(config, args.ablation)
            metrics = evaluate_candidate_vs_reference(config, result["policy_checkpoint"], reference_checkpoint)
            _save_variant_artifacts(result, metrics)
            results.append({
                "method":              method,
                "beta":                beta,
                "run_dir":             result["run_dir"],
                "mean_rm_score":       metrics["mean_candidate_rm_score"],
                "mean_kl":             metrics["mean_sampled_token_kl"],
                "rm_win_rate_vs_sft":  metrics["reward_model_win_rate_vs_sft"],
            })

    elif args.ablation == "clip_sweep":
        method = base_config["ablation"].get("method", "ppo")
        for eps in [0.05, 0.2, 0.5, 1e9]:  # 1e9 ≈ no clipping
            config = deepcopy(base_config)
            config[method]["epsilon"] = eps
            result  = _run_variant(config, args.ablation)
            metrics = evaluate_candidate_vs_reference(config, result["policy_checkpoint"], reference_checkpoint)
            _save_variant_artifacts(result, metrics)
            frame   = _load_metrics_frame(result["run_dir"])
            results.append({
                "method":         method,
                "epsilon":        "inf" if eps > 100 else eps,
                "run_dir":        result["run_dir"],
                "mean_rm_score":  metrics["mean_candidate_rm_score"],
                "mean_kl":        metrics["mean_sampled_token_kl"],
                "reward_variance": float(frame["mean_reward"].var()) if "mean_reward" in frame.columns else None,
            })

    elif args.ablation == "k_sweep":
        base_calls = (
            base_config["grpo"]["prompts_per_step"]
            * base_config["grpo"]["k_rollouts"]
        )
        for k in [1, 2, 4, 8]:
            config = deepcopy(base_config)
            config["grpo"]["k_rollouts"]      = k
            config["grpo"]["prompts_per_step"] = max(1, base_calls // k)
            result  = _run_variant(config, args.ablation)
            metrics = evaluate_candidate_vs_reference(config, result["policy_checkpoint"], reference_checkpoint)
            _save_variant_artifacts(result, metrics)
            frame   = _load_metrics_frame(result["run_dir"])
            results.append({
                "k_rollouts":        k,
                "prompts_per_step":  config["grpo"]["prompts_per_step"],
                "run_dir":           result["run_dir"],
                "mean_rm_score":     metrics["mean_candidate_rm_score"],
                "mean_kl":           metrics["mean_sampled_token_kl"],
                "degenerate_fraction": float(frame["degenerate"].mean()) if "degenerate" in frame.columns else None,
            })

    elif args.ablation == "dpo_beta_sweep":
        for beta in [0.01, 0.1, 0.5, 1.0]:
            config = deepcopy(base_config)
            config["dpo"]["beta"] = beta
            result  = _run_variant(config, args.ablation)
            metrics = evaluate_candidate_vs_reference(config, result["policy_checkpoint"], reference_checkpoint)
            _save_variant_artifacts(result, metrics)
            results.append({
                "beta":                      beta,
                "run_dir":                   result["run_dir"],
                "mean_rm_score":             metrics["mean_candidate_rm_score"],
                "mean_kl":                   metrics["mean_sampled_token_kl"],
                "heldout_preference_accuracy": metrics.get("heldout_preference_accuracy", None),
                "rm_win_rate_vs_sft":        metrics["reward_model_win_rate_vs_sft"],
            })

    # ── save results ──────────────────────────────────────────────────────────
    save_json(run_dir / "ablation_results.json", {
        "ablation": args.ablation,
        "results":  results,
    })
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "ablation_results.csv", index=False)
    print(f"\nAblation '{args.ablation}' complete. Results saved to {run_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
