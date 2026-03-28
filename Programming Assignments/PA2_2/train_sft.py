"""
SFT warm-up training script.

Usage:
    python train_sft.py [--config configs/sft.yaml] [--max_samples 2000]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from data.hh_rlhf import load_hh_rlhf, SFTDataset
from data.collators import SFTCollator
from model.loading import load_policy, get_tokenizer
from model.lora import apply_lora, freeze_model
from utils.config import load_config, merge_configs
from utils.seed import set_seed
from utils.memory import print_memory, clear_cache
from utils.logging_utils import get_logger, MetricsLogger, log_metrics
from utils.io import ensure_dir, save_peft_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/sft.yaml")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--run_dir",     default="runs/sft")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def train(args):
    logger = get_logger("sft")
    set_seed(args.seed)

    default_cfg = load_config("configs/default.yaml")
    task_cfg    = load_config(args.config)
    cfg         = merge_configs(default_cfg, task_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    run_dir = ensure_dir(args.run_dir)
    metrics_log = MetricsLogger(run_dir / "metrics.jsonl")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading HH-RLHF …")
    train_data = load_hh_rlhf("train", max_samples=args.max_samples)
    test_data  = load_hh_rlhf("test",  max_samples=500)

    tokenizer = get_tokenizer(cfg["policy_model"], padding_side="left")
    max_len   = cfg.get("max_seq_len", 1024)

    train_ds  = SFTDataset(train_data, tokenizer, max_len)
    test_ds   = SFTDataset(test_data,  tokenizer, max_len)
    collator  = SFTCollator(tokenizer, max_len)

    batch_size = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collator)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading policy …")
    model = load_policy(
        cfg["policy_model"],
        dtype=cfg.get("dtype", "bfloat16"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        device_map=device,
    )
    model = apply_lora(
        model,
        r=cfg.get("lora_r", 8),
        alpha=cfg.get("lora_alpha", 16),
        dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr = cfg.get("lr", 2e-4)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    grad_accum   = cfg.get("grad_accum", 4)
    epochs       = cfg.get("epochs", 1)
    total_steps  = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = cfg.get("warmup_steps", 50)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ──────────────────────────────────────────────────────────
    global_step = 0
    log_every   = cfg.get("log_every", 10)
    eval_every  = cfg.get("eval_every", 100)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"SFT epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            running_loss += out.loss.item()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = running_loss / log_every
                    ppl = math.exp(min(avg_loss, 20))
                    info = {"loss": avg_loss, "ppl": ppl, "lr": scheduler.get_last_lr()[0]}
                    log_metrics(logger, global_step, info, prefix="SFT")
                    metrics_log.log(global_step, info)
                    running_loss = 0.0

                if global_step % eval_every == 0:
                    val_loss = _eval(model, test_loader, device)
                    val_ppl  = math.exp(min(val_loss, 20))
                    logger.info(f"[step {global_step}] VAL loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                    metrics_log.log(global_step, {"val_loss": val_loss, "val_ppl": val_ppl})
                    model.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt_path = run_dir / "sft_adapter"
    save_peft_checkpoint(model, ckpt_path)
    print_memory()
    metrics_log.close()
    logger.info(f"SFT complete. Checkpoint at {ckpt_path}")
    return model, tokenizer


@torch.no_grad()
def _eval(model, loader, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        total_loss += out.loss.item()
        n += 1
        if n >= 20:  # quick eval
            break
    return total_loss / max(n, 1)


if __name__ == "__main__":
    train(parse_args())
