from __future__ import annotations

import argparse
import time

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data.collators import PairwiseRewardCollator
from data.hh_rlhf import load_hh_dataset, make_dpo_dataset
from model.loading import load_reward_model, load_reward_tokenizer
from model.reward_model import pairwise_reward_loss, score_sequences
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


def train_reward_model(config: dict) -> dict:
    seed_everything(config["seed"])
    logger = get_logger("train_rm")
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_dir = make_run_dir(config["output_dir"], "rm")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    model = load_reward_model(config, trainable=True)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    train_dataset = make_dpo_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_train_samples"])
    )
    collator = PairwiseRewardCollator(tokenizer=tokenizer, max_length=config["max_seq_len"])
    loader = DataLoader(
        train_dataset,
        batch_size=config["rm"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["num_workers"],
    )

    optimizer = AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=config["rm"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )
    total_steps = max(1, len(loader) * config["rm"]["epochs"] // config["rm"]["grad_accum"])
    warmup_steps = max(1, int(total_steps * config["rm"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    dtype = get_torch_dtype(config["prefer_bf16"])
    scaler = GradScaler(enabled=torch.cuda.is_available() and dtype == torch.float16)
    log_every = max(1, int(config["rm"].get("log_every", 1)))

    global_step = 0
    latest_loss = None
    latest_acc = None
    model.train()
    for epoch in range(config["rm"]["epochs"]):
        progress = tqdm(loader, desc=f"RM epoch {epoch + 1}/{config['rm']['epochs']}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
            with amp_context(dtype):
                chosen_scores = score_sequences(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected_scores = score_sequences(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
                loss = pairwise_reward_loss(chosen_scores, rejected_scores) / config["rm"]["grad_accum"]
            latest_loss = float(loss.item() * config["rm"]["grad_accum"])
            latest_acc = float((chosen_scores > rejected_scores).float().mean().item())
            scaler.scale(loss).backward()

            if step % config["rm"]["grad_accum"] == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = _grad_norm(model.parameters())
                lr = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                metrics = {
                    "step": global_step,
                    "loss": latest_loss,
                    "pairwise_acc": latest_acc,
                    "grad_norm": grad_norm,
                    "lr": lr,
                }
                metric_logger.log(metrics)
                if global_step % log_every == 0:
                    emit_step_log(logger, f"[RM {global_step}] loss={latest_loss:.4f} pref_acc={latest_acc:.4f} grad_norm={grad_norm:.4f} lr={lr:.6g}")
                progress.set_postfix(loss=latest_loss, acc=latest_acc, grad=grad_norm)
        if len(loader) % config["rm"]["grad_accum"] != 0 and latest_loss is not None and latest_acc is not None:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = _grad_norm(model.parameters())
            lr = optimizer.param_groups[0]["lr"]
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            metrics = {
                "step": global_step,
                "loss": latest_loss,
                "pairwise_acc": latest_acc,
                "grad_norm": grad_norm,
                "lr": lr,
            }
            metric_logger.log(metrics)
            if global_step % log_every == 0:
                emit_step_log(logger, f"[RM {global_step}] loss={latest_loss:.4f} pref_acc={latest_acc:.4f} grad_norm={grad_norm:.4f} lr={lr:.6g}")

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    total_training_time_sec = time.perf_counter() - run_start_time
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "rm", "rm_checkpoint": str(run_dir), "total_training_time_sec": total_training_time_sec, "mean_step_time_sec": total_training_time_sec / max(1, global_step), "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0})
    logger.info("Saved RM adapter to %s", run_dir)
    result = {"run_dir": str(run_dir), "rm_checkpoint": str(run_dir)}
    release_cuda_memory(model, optimizer, scheduler, scaler, loader, train_dataset, collator)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_reward_model(config)


if __name__ == "__main__":
    main()
