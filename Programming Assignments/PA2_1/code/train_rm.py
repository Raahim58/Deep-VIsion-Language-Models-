from __future__ import annotations

import argparse

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
from utils.logging_utils import JsonlMetricLogger, get_logger
from utils.memory import amp_context, get_torch_dtype
from utils.seed import seed_everything


def train_reward_model(config: dict) -> dict:
    seed_everything(config["seed"])
    logger = get_logger("train_rm")
    run_dir = make_run_dir(config["output_dir"], "rm")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    model = load_reward_model(config, trainable=True)
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

    global_step = 0
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
            scaler.scale(loss).backward()

            if step % config["rm"]["grad_accum"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                metrics = {
                    "step": global_step,
                    "loss": float(loss.item() * config["rm"]["grad_accum"]),
                    "pairwise_acc": float((chosen_scores > rejected_scores).float().mean().item()),
                }
                metric_logger.log(metrics)
                progress.set_postfix(loss=metrics["loss"], acc=metrics["pairwise_acc"])
        if len(loader) % config["rm"]["grad_accum"] != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "rm", "rm_checkpoint": str(run_dir)})
    logger.info("Saved RM adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "rm_checkpoint": str(run_dir)}


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
