"""
Evaluation script: win-rate, KL, sample table, GSM8K pass@1.

Usage:
    python eval.py --methods dpo ppo grpo --run_dir runs/eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config
from utils.seed import set_seed
from utils.logging_utils import get_logger
from utils.io import ensure_dir
from utils.metrics import win_rate, preference_accuracy
from utils.plotting import plot_method_comparison
from utils.text import format_sample_table, truncate_str
from model.loading import load_policy, get_tokenizer
from model.lora import apply_lora, freeze_model
from model.reward_model import RewardModel, score_texts
from model.generation import generate_responses
from data.hh_rlhf import load_hh_rlhf
from alignment.kl import kl_from_ref


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods",   nargs="+", default=["sft"],
                   help="Which methods to compare (each needs a runs/<method>/adapter dir)")
    p.add_argument("--sft_ckpt",  default="runs/sft/sft_adapter")
    p.add_argument("--rm_ckpt",   default=None)
    p.add_argument("--run_dir",   default="runs/eval")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main(args):
    logger = get_logger("eval")
    set_seed(args.seed)

    cfg    = load_config("configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out    = ensure_dir(args.run_dir)

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading test prompts …")
    test_data = load_hh_rlhf("test", max_samples=args.n_prompts)
    prompts   = [d["prompt"] for d in test_data][: args.n_prompts]

    # ── RM ───────────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    rm_tok = AutoTokenizer.from_pretrained(cfg["reward_model"])
    rm_tok.padding_side = "right"
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token

    rm = RewardModel.load_frozen(cfg["reward_model"], use_8bit=cfg.get("use_8bit_frozen", True))
    if args.rm_ckpt:
        from utils.io import load_checkpoint
        load_checkpoint(rm, args.rm_ckpt, strict=False)

    # ── SFT baseline ─────────────────────────────────────────────────────────
    logger.info("Loading SFT baseline …")
    tok = get_tokenizer(cfg["policy_model"], padding_side="left")
    sft_model = _load_model_with_adapter(cfg, args.sft_ckpt, device)
    sft_responses = _generate_all(sft_model, tok, prompts, device)
    sft_scores    = score_texts(rm, rm_tok, [p + " " + r for p, r in zip(prompts, sft_responses)])

    results: dict[str, dict] = {"sft": {"responses": sft_responses, "scores": sft_scores}}

    # ── Reference model for KL ────────────────────────────────────────────────
    ref_model = _load_model_with_adapter(cfg, args.sft_ckpt, device)
    freeze_model(ref_model)

    # ── Evaluate each method ──────────────────────────────────────────────────
    for method in args.methods:
        if method == "sft":
            continue
        adapter_path = f"runs/{method}/adapter"
        if not Path(adapter_path).exists():
            logger.warning(f"Adapter not found: {adapter_path} — skipping {method}")
            continue

        logger.info(f"Evaluating {method} …")
        model     = _load_model_with_adapter(cfg, adapter_path, device)
        responses = _generate_all(model, tok, prompts, device)
        full_texts = [p + " " + r for p, r in zip(prompts, responses)]
        scores    = score_texts(rm, rm_tok, full_texts)
        wr        = win_rate(scores, sft_scores)

        # KL from reference (approx)
        kl_vals = _compute_kl(model, ref_model, tok, prompts[:50], device)

        results[method] = {
            "responses": responses,
            "scores":    scores,
            "win_rate":  wr,
            "mean_kl":   kl_vals,
        }
        logger.info(f"  {method}: win_rate={wr:.4f}  mean_kl={kl_vals:.4f}")

        del model
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = {m: {"win_rate": v.get("win_rate", 0), "mean_kl": v.get("mean_kl", 0)}
               for m, v in results.items() if m != "sft"}
    if summary:
        plot_method_comparison(summary, metric="win_rate", save_path=out / "win_rate.png")

    # ── Sample response table (5 prompts) ────────────────────────────────────
    sample_indices = list(range(min(5, len(prompts))))
    table_rows = []
    for i in sample_indices:
        row = {"prompt": truncate_str(prompts[i], 120)}
        for method, mres in results.items():
            row[method] = truncate_str(mres["responses"][i], 100)
            row[f"{method}_score"] = f"{mres['scores'][i]:.3f}"
        table_rows.append(row)

    columns = ["prompt"] + [c for m in results for c in [m, f"{m}_score"]]
    table_md = format_sample_table(table_rows, columns)
    (out / "sample_table.md").write_text(table_md)
    logger.info(f"Sample table → {out / 'sample_table.md'}")

    # ── Resource summary ──────────────────────────────────────────────────────
    from utils.memory import memory_stats
    logger.info(f"Memory: {memory_stats()}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model_with_adapter(cfg, adapter_path, device):
    model = load_policy(cfg["policy_model"], dtype=cfg.get("dtype", "bfloat16"),
                        gradient_checkpointing=False, device_map=device)
    model = apply_lora(model, r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16),
                       dropout=0.0,
                       target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]))
    if adapter_path and Path(adapter_path).exists():
        model.load_adapter(adapter_path, adapter_name="default")
    return model


@torch.no_grad()
def _generate_all(model, tok, prompts, device, batch_size=8, max_new_tokens=128):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        resps = generate_responses(model, tok, batch, max_new_tokens=max_new_tokens,
                                   do_sample=False, device=device)
        responses.extend(resps)
    return responses


@torch.no_grad()
def _compute_kl(model, ref_model, tok, prompts, device) -> float:
    """Monte-Carlo KL estimate over a subset of prompts."""
    from model.logprobs import token_logprobs
    total_kl, total_tok = 0.0, 0
    for prompt in prompts:
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=512)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        lp_pol = token_logprobs(model,     ids, mask)
        lp_ref = token_logprobs(ref_model, ids, mask)
        kl_toks = (lp_pol - lp_ref).clamp(min=0)
        total_kl  += kl_toks.sum().item()
        total_tok += kl_toks.numel()
    return total_kl / max(total_tok, 1)


if __name__ == "__main__":
    main(parse_args())
