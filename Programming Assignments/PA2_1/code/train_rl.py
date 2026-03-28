from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from alignment.dpo import dpo_batch_metrics
from alignment.grpo import build_rm_reward_fn, collect_group_rollout, grpo_update_step
from alignment.ppo import collect_ppo_rollout, ppo_update_step
from alignment.rlvr import gsm8k_verifiable_rewards
from data.collators import DPOCollator
from data.gsm8k import load_gsm8k_dataset
from data.hh_rlhf import load_hh_dataset, make_dpo_dataset, make_prompt_dataset
from model.loading import (
    load_policy_model,
    load_policy_tokenizer,
    load_reference_model,
    load_reward_model,
    load_reward_tokenizer,
    load_value_model,
)
from utils.config import load_config
from utils.io import make_run_dir, save_json
from utils.logging_utils import JsonlMetricLogger, get_logger
from utils.seed import seed_everything


def _sample_rows(dataset, count: int):
    indices = random.sample(range(len(dataset)), k=min(count, len(dataset)))
    return [dataset[int(index)] for index in indices]


def _resolve_sft_checkpoint(config: dict) -> str | None:
    return config["models"].get("sft_checkpoint") or config["models"].get("policy_checkpoint")


def run_dpo(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "dpo")
    logger = get_logger("train_dpo")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    start_checkpoint = _resolve_sft_checkpoint(config)
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)

    dataset = make_dpo_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_train_samples"])
    )
    loader = DataLoader(
        dataset,
        batch_size=config["dpo"]["batch_size"],
        shuffle=True,
        collate_fn=DPOCollator(policy_tokenizer, config["max_seq_len"]),
        num_workers=config["num_workers"],
    )
    optimizer = AdamW(
        filter(lambda param: param.requires_grad, policy.parameters()),
        lr=config["dpo"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )

    policy.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["dpo"]["epochs"]):
        progress = tqdm(loader, desc=f"DPO epoch {epoch + 1}/{config['dpo']['epochs']}")
        for step, batch in enumerate(progress, start=1):
            metrics = dpo_batch_metrics(policy, reference, batch, config["dpo"]["beta"])
            loss = metrics["loss"] / config["dpo"]["grad_accum"]
            loss.backward()
            if step % config["dpo"]["grad_accum"] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                row = {
                    "step": global_step,
                    "loss": float(metrics["loss"].item()),
                    "implicit_reward_margin": float(metrics["implicit_reward_margin"].item()),
                    "preference_accuracy": float(metrics["preference_accuracy"].item()),
                }
                metric_logger.log(row)
                progress.set_postfix(loss=row["loss"], acc=row["preference_accuracy"])
        if len(loader) % config["dpo"]["grad_accum"] != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "dpo", "policy_checkpoint": str(run_dir)})
    logger.info("Saved DPO adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def run_ppo(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "ppo")
    logger = get_logger("train_ppo")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_prompt_pool_samples"])
    )
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    start_checkpoint = _resolve_sft_checkpoint(config)
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)
    reward_model = load_reward_model(config, checkpoint=config["models"].get("rm_checkpoint"), trainable=False)
    value_model = load_value_model(config)

    policy_optimizer = AdamW(
        filter(lambda param: param.requires_grad, policy.parameters()),
        lr=config["ppo"]["lr_policy"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )
    value_optimizer = AdamW(
        filter(lambda param: param.requires_grad, value_model.parameters()),
        lr=config["ppo"]["lr_value"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )

    for step in tqdm(range(1, config["ppo"]["steps"] + 1), desc="PPO"):
        prompts = [row["prompt"] for row in _sample_rows(prompts_dataset, config["ppo"]["prompts_per_step"])]
        rollout = collect_ppo_rollout(
            policy,
            reference,
            value_model,
            reward_model,
            policy_tokenizer,
            reward_tokenizer,
            prompts,
            config,
        )
        metrics = ppo_update_step(
            policy=policy,
            value_model=value_model,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            rollout=rollout,
            epsilon=config["ppo"]["epsilon"],
            value_coef=config["ppo"]["value_coef"],
            entropy_coef=config["ppo"]["entropy_coef"],
        )
        metrics["step"] = step
        metric_logger.log(metrics)
        if step % config["save_every"] == 0:
            checkpoint_dir = run_dir / f"checkpoint_step_{step}"
            policy.save_pretrained(checkpoint_dir)
            policy_tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "ppo", "policy_checkpoint": str(run_dir)})
    logger.info("Saved PPO adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def run_grpo(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "grpo")
    logger = get_logger("train_grpo")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_prompt_pool_samples"])
    )
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    start_checkpoint = _resolve_sft_checkpoint(config)
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)
    reward_model = load_reward_model(config, checkpoint=config["models"].get("rm_checkpoint"), trainable=False)
    reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])

    optimizer = AdamW(
        filter(lambda param: param.requires_grad, policy.parameters()),
        lr=config["grpo"]["lr_policy"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )

    for step in tqdm(range(1, config["grpo"]["steps"] + 1), desc="GRPO"):
        prompts = [row["prompt"] for row in _sample_rows(prompts_dataset, config["grpo"]["prompts_per_step"])]
        rollout = collect_group_rollout(
            policy=policy,
            reference_model=reference,
            tokenizer=policy_tokenizer,
            prompts=prompts,
            config=config,
            reward_fn=reward_fn,
            max_new_tokens=config["grpo"]["max_new_tokens"],
        )
        metrics = grpo_update_step(
            policy=policy,
            policy_optimizer=optimizer,
            rollout=rollout,
            epsilon=config["grpo"]["epsilon"],
            beta=config["grpo"]["beta"],
        )
        metrics["step"] = step
        metric_logger.log(metrics)
        if step % config["save_every"] == 0:
            checkpoint_dir = run_dir / f"checkpoint_step_{step}"
            policy.save_pretrained(checkpoint_dir)
            policy_tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "grpo", "policy_checkpoint": str(run_dir)})
    logger.info("Saved GRPO adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def run_rlvr(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "rlvr")
    logger = get_logger("train_rlvr")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    gsm_train = load_gsm8k_dataset(config, "train", config["data"]["gsm_train_samples"])
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    start_checkpoint = _resolve_sft_checkpoint(config)
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)

    optimizer = AdamW(
        filter(lambda param: param.requires_grad, policy.parameters()),
        lr=config["rlvr"]["lr_policy"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )

    grpo_like_config = {**config, "grpo": {**config["grpo"], "k_rollouts": config["rlvr"]["k_rollouts"]}}

    for step in tqdm(range(1, config["rlvr"]["steps"] + 1), desc="RLVR"):
        rows = _sample_rows(gsm_train, config["rlvr"]["prompts_per_step"])
        prompts = [row["prompt"] for row in rows]
        gold_answers = [row["answer"] for row in rows for _ in range(config["rlvr"]["k_rollouts"])]

        def reward_fn(expanded_prompts, completions):
            return gsm8k_verifiable_rewards(expanded_prompts, completions, gold_answers)

        rollout = collect_group_rollout(
            policy=policy,
            reference_model=reference,
            tokenizer=policy_tokenizer,
            prompts=prompts,
            config=grpo_like_config,
            reward_fn=reward_fn,
            max_new_tokens=config["rlvr"]["max_new_tokens"],
        )
        metrics = grpo_update_step(
            policy=policy,
            policy_optimizer=optimizer,
            rollout=rollout,
            epsilon=config["rlvr"]["epsilon"],
            beta=config["rlvr"]["beta"],
        )
        metrics["step"] = step
        metric_logger.log(metrics)
        if step % config["save_every"] == 0:
            checkpoint_dir = run_dir / f"checkpoint_step_{step}"
            policy.save_pretrained(checkpoint_dir)
            policy_tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "rlvr", "policy_checkpoint": str(run_dir)})
    logger.info("Saved RLVR adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dpo", "ppo", "grpo", "rlvr"], required=True)
    parser.add_argument("--config", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.method == "dpo":
        run_dpo(config)
    elif args.method == "ppo":
        run_ppo(config)
    elif args.method == "grpo":
        run_grpo(config)
    elif args.method == "rlvr":
        run_rlvr(config)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
