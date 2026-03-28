"""
Reward model training script.

Usage:
    python train_rm.py [--config configs/rm.yaml] [--max_samples 5000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from data.hh_rlhf import load_hh_rlhf, RMDataset
from data.collators import RMCollator
from model.reward_model import RewardModel
from utils.config import load_config, merge_configs
from utils.seed import set_seed
from utils.memory import print_memory
from utils.logging_utils import get_logger, MetricsLogger, log_metrics
from utils.io import ensure_dir, save_checkpoint
from utils.metrics import preference_accuracy
from utils.plotting import plot_reward_distribution


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/rm.yaml")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--run_dir",     default="runs/rm")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def rm_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, lambda_reg: float = 1e-3) -> torch.Tensor:
    """Margin ranking loss with L2 regularisation on reward magnitude."""
    ranking = -F.logsigmoid(r_pos - r_neg).mean()
    reg     = lambda_reg * (r_pos.pow(2) + r_neg.pow(2)).mean()
    return ranking + reg


def train(args):
    logger = get_logger("rm")
    set_seed(args.seed)

    default_cfg = load_config("configs/default.yaml")
    task_cfg    = load_config(args.config)
    cfg         = merge_configs(default_cfg, task_cfg)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir  = ensure_dir(args.run_dir)
    mlog     = MetricsLogger(run_dir / "metrics.jsonl")
    lambda_r = cfg.get("lambda_reg", 1e-3)

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading HH-RLHF …")
    train_data = load_hh_rlhf("train", max_samples=args.max_samples)
    test_data  = load_hh_rlhf("test",  max_samples=500)

    # RM uses its own tokenizer (Llama-3.2-1B)
    rm_tok = AutoTokenizer.from_pretrained(cfg["reward_model"])
    rm_tok.padding_side = "right"  # RM reads last real token → right padding
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token

    max_len  = cfg.get("max_seq_len", 1024)
    collator = RMCollator(rm_tok, max_len)

    batch_size   = cfg.get("batch_size", 8)
    train_loader = DataLoader(RMDataset(train_data), batch_size=batch_size, shuffle=True,  collate_fn=collator)
    test_loader  = DataLoader(RMDataset(test_data),  batch_size=batch_size, shuffle=False, collate_fn=collator)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading reward model …")
    rm = RewardModel(cfg["reward_model"], use_8bit=False)
    # Optionally freeze backbone and train only the classification head
    # For simplicity we keep the full model trainable here.
    optimizer = torch.optim.AdamW(
        [p for p in rm.parameters() if p.requires_grad],
        lr=cfg.get("lr", 1e-4), weight_decay=0.01,
    )
    epochs      = cfg.get("epochs", 1)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, cfg.get("warmup_steps", 30), total_steps
    )

    log_every = cfg.get("log_every", 10)
    global_step = 0

    for epoch in range(epochs):
        rm.train()
        running_loss = 0.0
        running_acc  = 0.0

        for batch in tqdm(train_loader, desc=f"RM epoch {epoch+1}"):
            ids_pos  = batch["input_ids_pos"].to(device)
            mask_pos = batch["attention_mask_pos"].to(device)
            ids_neg  = batch["input_ids_neg"].to(device)
            mask_neg = batch["attention_mask_neg"].to(device)

            r_pos = rm(ids_pos, mask_pos)
            r_neg = rm(ids_neg, mask_neg)
            loss  = rm_loss(r_pos, r_neg, lambda_r)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                acc = (r_pos > r_neg).float().mean().item()
            running_loss += loss.item()
            running_acc  += acc

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                avg_acc  = running_acc  / log_every
                info = {"loss": avg_loss, "pref_acc": avg_acc}
                log_metrics(logger, global_step, info, prefix="RM")
                mlog.log(global_step, info)
                running_loss = running_acc = 0.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    rm.eval()
    all_pos, all_neg = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="RM eval"):
            ids_pos  = batch["input_ids_pos"].to(device)
            mask_pos = batch["attention_mask_pos"].to(device)
            ids_neg  = batch["input_ids_neg"].to(device)
            mask_neg = batch["attention_mask_neg"].to(device)
            all_pos.extend(rm(ids_pos, mask_pos).cpu().tolist())
            all_neg.extend(rm(ids_neg, mask_neg).cpu().tolist())

    test_acc = preference_accuracy(all_pos, all_neg)
    logger.info(f"Test preference accuracy: {test_acc:.4f}")
    mlog.log(global_step, {"test_pref_acc": test_acc})

    target = cfg.get("target_accuracy", 0.60)
    if test_acc < target:
        logger.warning(f"RM accuracy {test_acc:.2%} below target {target:.0%}. "
                       "Consider more training or a larger dataset.")

    plot_reward_distribution(all_pos, all_neg, save_path=run_dir / "reward_dist.png")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_checkpoint(rm, run_dir / "rm.pt", extra={"test_pref_acc": test_acc})
    print_memory()
    mlog.close()
    logger.info(f"RM training complete. Saved to {run_dir}")
    return rm


if __name__ == "__main__":
    train(parse_args())
