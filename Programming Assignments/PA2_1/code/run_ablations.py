from __future__ import annotations

"""
Task C7: Focused Ablations
Run ONE of:
  1. kl_sweep      – β ∈ {0, 0.05, 0.1, 0.5}   on PPO or GRPO (200 steps each)
  2. clip_sweep    – ε ∈ {0.05, 0.2, 0.5, ∞}    on PPO or GRPO (200 steps each)
  3. k_sweep       – K ∈ {1, 2, 4, 8}            on GRPO, total RM calls constant
  4. dpo_beta_sweep– β ∈ {0.01, 0.1, 0.5, 1.0}  on DPO

For each variant:
  a) Train the method
  b) Evaluate: RM score on greedy outputs + KL from π_ref  (30 prompts, fast)
  c) Print results inline immediately after each variant
  d) Save ablation_results.csv with all variants
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from eval import evaluate_candidate_vs_reference
from train_rl import run_dpo, run_grpo, run_ppo
from utils.config import load_config
from utils.io import ensure_dir, make_run_dir, save_json
from utils.plotting import plot_metric_curves

# Fewer eval prompts inside each variant so it's fast enough on A100
ABLATION_EVAL_PROMPTS = 30

# Steps per variant (manual: 200)
ABLATION_STEPS = 200


def _find_default_yaml() -> Path | None:
    search = Path(os.getcwd()).resolve()
    for _ in range(6):
        c = search / "configs" / "default.yaml"
        if c.exists():
            return c
        search = search.parent
    return None


def _load_metrics_frame(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir) / "metrics.jsonl"
    return pd.read_json(p, lines=True) if p.exists() else pd.DataFrame()


def _run_variant(config: dict, ablation: str) -> dict:
    if ablation in {"kl_sweep", "clip_sweep"}:
        method = config.get("_ablation_method", "ppo")
        return run_ppo(config) if method == "ppo" else run_grpo(config)
    if ablation == "k_sweep":
        return run_grpo(config)
    if ablation == "dpo_beta_sweep":
        return run_dpo(config)
    raise ValueError(f"Unknown ablation: {ablation}")


def _eval_variant(config: dict, result: dict, reference_checkpoint: str, label: str) -> dict:
    """Evaluate one variant; prints inline, saves per-variant files."""
    print(f"\n  ── eval: {label} ──", flush=True)
    eval_cfg = deepcopy(config)
    eval_cfg["evaluation"]["prompts"] = ABLATION_EVAL_PROMPTS
    try:
        metrics = evaluate_candidate_vs_reference(
            eval_cfg, result["policy_checkpoint"], reference_checkpoint
        )
    except Exception as exc:
        print(f"  [WARN] eval failed for {label}: {exc}", flush=True)
        return {}

    print(f"  [{label}] RM win-rate vs SFT : {metrics.get('reward_model_win_rate_vs_sft'):.4f}", flush=True)
    print(f"  [{label}] Mean KL from π_ref : {metrics.get('mean_sampled_token_kl'):.4f}", flush=True)
    print(f"  [{label}] Mean RM score      : {metrics.get('mean_candidate_rm_score'):.4f}", flush=True)
    if metrics.get("heldout_preference_accuracy") is not None:
        print(f"  [{label}] Pref accuracy      : {metrics['heldout_preference_accuracy']:.4f}", flush=True)

    run_dir = Path(result["run_dir"])
    mpath = run_dir / "metrics.jsonl"
    if mpath.exists():
        try:
            plot_metric_curves(mpath, run_dir / "metrics.png")
            print(f"  [{label}] plot → {run_dir}/metrics.png", flush=True)
        except Exception:
            pass

    edr = ensure_dir(run_dir / "eval")
    save_json(edr / "eval_results.json", {k: v for k, v in metrics.items() if k != "sample_rows"})
    if "sample_rows" in metrics:
        pd.DataFrame(metrics["sample_rows"]).to_csv(edr / "sample_generations.csv", index=False)
    return metrics


def _summarise(ablation: str, rows: list[dict]) -> None:
    print(f"\n{'═'*60}", flush=True)
    print(f"  {ablation} — final summary", flush=True)
    print(f"{'═'*60}", flush=True)
    df = pd.DataFrame(rows)
    display_cols = [c for c in df.columns if c != "run_dir"]
    print(df[display_cols].to_string(index=False), flush=True)
    print(flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def _kl_sweep(base: dict, method: str, ref_ckpt: str) -> list[dict]:
    """C7.1 – β sweep, report RM score + KL."""
    betas = [0.0, 0.05, 0.1, 0.5]
    print(f"\n[kl_sweep] method={method}  β={betas}", flush=True)
    rows = []
    for beta in betas:
        cfg = deepcopy(base)
        cfg[method]["beta"]   = beta
        cfg[method]["steps"]  = ABLATION_STEPS
        cfg["_ablation_method"] = method
        label = f"{method}_β={beta}"
        print(f"\n[kl_sweep] ▶ training {label}", flush=True)
        result  = _run_variant(cfg, "kl_sweep")
        metrics = _eval_variant(cfg, result, ref_ckpt, label)
        frame   = _load_metrics_frame(result["run_dir"])
        rows.append({
            "method":             method,
            "beta":               beta,
            "run_dir":            result["run_dir"],
            "mean_rm_score":      metrics.get("mean_candidate_rm_score"),
            "mean_kl":            metrics.get("mean_sampled_token_kl"),
            "rm_win_rate_vs_sft": metrics.get("reward_model_win_rate_vs_sft"),
            "final_mean_reward":  float(frame["mean_reward"].iloc[-1])
                                  if "mean_reward" in frame.columns and len(frame) else None,
        })
        _summarise("kl_sweep (so far)", rows)
    return rows


def _clip_sweep(base: dict, method: str, ref_ckpt: str) -> list[dict]:
    """C7.2 – ε sweep, report grad-norm variance + RM score."""
    epsilons = [0.05, 0.2, 0.5, 1e9]
    print(f"\n[clip_sweep] method={method}  ε={epsilons}", flush=True)
    rows = []
    for eps in epsilons:
        cfg = deepcopy(base)
        cfg[method]["epsilon"] = eps
        cfg[method]["steps"]   = ABLATION_STEPS
        cfg["_ablation_method"] = method
        eps_label = "inf" if eps > 100 else eps
        label = f"{method}_ε={eps_label}"
        print(f"\n[clip_sweep] ▶ training {label}", flush=True)
        result  = _run_variant(cfg, "clip_sweep")
        metrics = _eval_variant(cfg, result, ref_ckpt, label)
        frame   = _load_metrics_frame(result["run_dir"])
        grad_var   = float(frame["policy_grad_norm"].var()) if "policy_grad_norm" in frame.columns and len(frame) else None
        reward_var = float(frame["mean_reward"].var())      if "mean_reward"       in frame.columns and len(frame) else None
        if grad_var   is not None: print(f"  [{label}] Grad-norm variance: {grad_var:.6f}", flush=True)
        if reward_var is not None: print(f"  [{label}] Reward variance   : {reward_var:.6f}", flush=True)
        rows.append({
            "method":             method,
            "epsilon":            eps_label,
            "run_dir":            result["run_dir"],
            "mean_rm_score":      metrics.get("mean_candidate_rm_score"),
            "mean_kl":            metrics.get("mean_sampled_token_kl"),
            "grad_norm_variance": grad_var,
            "reward_variance":    reward_var,
        })
        _summarise("clip_sweep (so far)", rows)
    return rows


def _k_sweep(base: dict, ref_ckpt: str) -> list[dict]:
    """C7.3 – K sweep on GRPO, total RM calls/step fixed."""
    k_values   = [1, 2, 4, 8]
    base_calls = base["grpo"]["prompts_per_step"] * base["grpo"]["k_rollouts"]
    print(f"\n[k_sweep] K={k_values}  base_calls/step={base_calls}", flush=True)
    rows = []
    for k in k_values:
        cfg = deepcopy(base)
        cfg["grpo"]["k_rollouts"]       = k
        cfg["grpo"]["prompts_per_step"] = max(1, base_calls // k)
        cfg["grpo"]["steps"]            = ABLATION_STEPS
        label = f"GRPO_K={k}"
        print(f"\n[k_sweep] ▶ training {label}  (prompts/step={cfg['grpo']['prompts_per_step']})", flush=True)
        result  = _run_variant(cfg, "k_sweep")
        metrics = _eval_variant(cfg, result, ref_ckpt, label)
        frame   = _load_metrics_frame(result["run_dir"])
        degen = float(frame["degenerate"].mean()) if "degenerate" in frame.columns and len(frame) else None
        if degen is not None:
            print(f"  [{label}] Degenerate batch fraction: {degen:.4f}", flush=True)
        print(
            f"  [{label}] Theory: larger K → lower variance in group baseline μ_b "
            f"(each mean estimated from more rollouts → less noise in advantages).",
            flush=True,
        )
        rows.append({
            "k_rollouts":          k,
            "prompts_per_step":    cfg["grpo"]["prompts_per_step"],
            "run_dir":             result["run_dir"],
            "mean_rm_score":       metrics.get("mean_candidate_rm_score"),
            "mean_kl":             metrics.get("mean_sampled_token_kl"),
            "degenerate_fraction": degen,
        })
        _summarise("k_sweep (so far)", rows)
    return rows


def _dpo_beta_sweep(base: dict, ref_ckpt: str) -> list[dict]:
    """C7.4 – DPO β sweep, report pref acc + KL + RM score."""
    betas = [0.01, 0.1, 0.5, 1.0]
    print(f"\n[dpo_beta_sweep] β={betas}", flush=True)
    rows = []
    for beta in betas:
        cfg = deepcopy(base)
        cfg["dpo"]["beta"] = beta
        label = f"DPO_β={beta}"
        print(f"\n[dpo_beta_sweep] ▶ training {label}", flush=True)
        result  = _run_variant(cfg, "dpo_beta_sweep")
        metrics = _eval_variant(cfg, result, ref_ckpt, label)
        print(
            f"  [{label}] β controls KL regularisation: higher β → stays closer "
            f"to SFT (lower RM gaming risk); lower β → more RM optimisation.",
            flush=True,
        )
        rows.append({
            "beta":                        beta,
            "run_dir":                     result["run_dir"],
            "mean_rm_score":               metrics.get("mean_candidate_rm_score"),
            "mean_kl":                     metrics.get("mean_sampled_token_kl"),
            "heldout_preference_accuracy": metrics.get("heldout_preference_accuracy"),
            "rm_win_rate_vs_sft":          metrics.get("reward_model_win_rate_vs_sft"),
        })
        _summarise("dpo_beta_sweep (so far)", rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="C7: Focused Ablations")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--ablation",
                        choices=["kl_sweep", "clip_sweep", "k_sweep", "dpo_beta_sweep"],
                        required=True)
    parser.add_argument("--method", choices=["ppo", "grpo"], default=None,
                        help="Required for kl_sweep / clip_sweep.")
    args = parser.parse_args()

    # merge default.yaml first
    ordered: list[str] = []
    seen: set[str] = set()
    dy = _find_default_yaml()
    if dy:
        ordered.append(str(dy)); seen.add(str(dy))
    for p in args.config:
        if p not in seen:
            ordered.append(p); seen.add(p)

    base_config = load_config(ordered)

    ref_ckpt = (
        base_config["models"].get("sft_checkpoint")
        or base_config["models"].get("policy_checkpoint")
    )
    if not ref_ckpt:
        print("ERROR: sft_checkpoint not set. Run SFT first.", file=sys.stderr)
        sys.exit(1)
    if base_config["models"].get("rm_checkpoint") is None:
        print("ERROR: rm_checkpoint not set. Train RM first.", file=sys.stderr)
        sys.exit(1)
    if args.ablation in {"kl_sweep", "clip_sweep"} and args.method is None:
        print(f"ERROR: --method required for {args.ablation}.", file=sys.stderr)
        sys.exit(1)

    method = args.method or "grpo"
    abl_dir = make_run_dir(base_config["output_dir"], f"ablation_{args.ablation}")
    print(f"\n[Ablation] {args.ablation} → {abl_dir}", flush=True)
    print(f"[Ablation] reference: {ref_ckpt}", flush=True)

    if args.ablation == "kl_sweep":
        rows = _kl_sweep(base_config, method, ref_ckpt)
    elif args.ablation == "clip_sweep":
        rows = _clip_sweep(base_config, method, ref_ckpt)
    elif args.ablation == "k_sweep":
        rows = _k_sweep(base_config, ref_ckpt)
    else:
        rows = _dpo_beta_sweep(base_config, ref_ckpt)

    _summarise(args.ablation, rows)
    save_json(abl_dir / "ablation_results.json", {"ablation": args.ablation, "results": rows})
    df = pd.DataFrame(rows)
    df.to_csv(abl_dir / "ablation_results.csv", index=False)
    print(f"[Ablation] Saved → {abl_dir}", flush=True)


if __name__ == "__main__":
    main()
