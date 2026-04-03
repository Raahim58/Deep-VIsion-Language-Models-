from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data.collators import SFTCollator
from data.hh_rlhf import load_hh_dataset, make_sft_dataset
from model.loading import load_policy_model, load_policy_tokenizer
from utils.config import load_config
from utils.io import make_run_dir, save_json
from utils.logging_utils import JsonlMetricLogger, emit_step_log, get_logger
from utils.memory import amp_context, get_torch_dtype, release_cuda_memory
from utils.seed import seed_everything


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.norm(2).item() ** 2
    return total ** 0.5


def train_sft(config: dict) -> dict:
    seed_everything(config["seed"])
    logger = get_logger("train_sft")
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_dir = make_run_dir(config["output_dir"], "sft")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    model = load_policy_model(config, checkpoint=None, trainable=True)

    train_dataset = make_sft_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_train_samples"])
    )
    collator = SFTCollator(tokenizer=tokenizer, max_length=config["max_seq_len"])
    loader = DataLoader(
        train_dataset,
        batch_size=config["sft"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["num_workers"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["sft"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )
    total_steps = max(1, len(loader) * config["sft"]["epochs"] // config["sft"]["grad_accum"])
    warmup_steps = max(1, int(total_steps * config["sft"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    dtype = get_torch_dtype(config["prefer_bf16"])
    scaler = GradScaler(enabled=torch.cuda.is_available() and dtype == torch.float16)
    log_every = max(1, int(config["sft"].get("log_every", 1)))

    global_step = 0
    latest_loss = None
    model.train()
    for epoch in range(config["sft"]["epochs"]):
        progress = tqdm(loader, desc=f"SFT epoch {epoch + 1}/{config['sft']['epochs']}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
            with amp_context(dtype):
                outputs = model(**batch)
                loss = outputs.loss / config["sft"]["grad_accum"]
            latest_loss = float(loss.item() * config["sft"]["grad_accum"])
            scaler.scale(loss).backward()

            if step % config["sft"]["grad_accum"] == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = _grad_norm(model.parameters())
                lr = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                metrics = {"step": global_step, "loss": latest_loss, "grad_norm": grad_norm, "lr": lr}
                metric_logger.log(metrics)
                if global_step % log_every == 0:
                    emit_step_log(logger, f"[SFT {global_step}] loss={latest_loss:.4f} grad_norm={grad_norm:.4f} lr={lr:.6g}")
                progress.set_postfix(loss=latest_loss, grad=grad_norm)
        if len(loader) % config["sft"]["grad_accum"] != 0 and latest_loss is not None:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = _grad_norm(model.parameters())
            lr = optimizer.param_groups[0]["lr"]
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            metrics = {"step": global_step, "loss": latest_loss, "grad_norm": grad_norm, "lr": lr}
            metric_logger.log(metrics)
            if global_step % log_every == 0:
                emit_step_log(logger, f"[SFT {global_step}] loss={latest_loss:.4f} grad_norm={grad_norm:.4f} lr={lr:.6g}")

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    total_training_time_sec = time.perf_counter() - run_start_time
    save_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "stage": "sft",
            "policy_checkpoint": str(run_dir),
            "total_training_time_sec": total_training_time_sec,
            "mean_step_time_sec": total_training_time_sec / max(1, global_step),
            "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        },
    )
    logger.info("Saved SFT adapter to %s", run_dir)
    result = {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}
    release_cuda_memory(model, optimizer, scheduler, scaler, loader, train_dataset, collator)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="YAML config path; can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_sft(config)


if __name__ == "__main__":
    main()
