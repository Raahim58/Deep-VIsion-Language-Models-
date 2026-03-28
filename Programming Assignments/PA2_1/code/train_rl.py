from __future__ import annotations

import argparse
import time
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from alignment.dpo import dpo_batch_metrics
from alignment.kl import mean_masked_kl
from alignment.grpo import build_rm_reward_fn, collect_group_rollout, grpo_update_step
from alignment.ppo import build_rollout_batch, collect_ppo_rollout, ppo_update_step
from alignment.rlvr import gsm8k_verifiable_rewards
from data.collators import DPOCollator
from data.gsm8k import load_gsm8k_dataset
from data.hh_rlhf import load_hh_dataset, make_dpo_dataset, make_prompt_dataset
from model.generation import generate_completions
from model.logprobs import response_token_logprobs
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
from utils.logging_utils import JsonlMetricLogger, emit_step_log, get_logger
from utils.metrics import gsm8k_exact_match
from utils.seed import seed_everything


def _sample_rows(dataset, count: int):
    indices = random.sample(range(len(dataset)), k=min(count, len(dataset)))
    return [dataset[int(index)] for index in indices]


def _resolve_sft_checkpoint(config: dict) -> str | None:
    return config["models"].get("sft_checkpoint") or config["models"].get("policy_checkpoint")


def _require_checkpoint(path: str | None, label: str) -> str:
    if not path:
        raise ValueError(f"{label} is required but was not set. Train/load the prerequisite checkpoint first.")
    return path


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.norm(2).item() ** 2
    return total ** 0.5


def _log_every(config: dict, section: str) -> int:
    return max(1, int(config.get(section, {}).get("log_every", 1)))


def _mean_response_length(tokenizer, completions: list[str]) -> float:
    if not completions:
        return 0.0
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in completions]
    return float(sum(lengths) / len(lengths))


def _evaluate_rlvr_pass_at_1(policy, tokenizer, dataset, max_new_tokens: int) -> float:
    prompts = [row["prompt"] for row in dataset]
    gold = [row["answer"] for row in dataset]
    outputs = generate_completions(policy, tokenizer, prompts, max_new_tokens=max_new_tokens, do_sample=False)
    correct = sum(gsm8k_exact_match(pred, target) for pred, target in zip(outputs, gold))
    return correct / max(1, len(gold))


def _peak_vram_gb() -> float:
    return float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0


def _evaluate_dpo_holdout(policy, reference, reward_fn, tokenizer, heldout_prompts: list[str], heldout_loader, config: dict) -> dict[str, float]:
    outputs = generate_completions(
        policy,
        tokenizer,
        heldout_prompts,
        max_new_tokens=config["evaluation"]["max_new_tokens"],
        do_sample=False,
    )
    scores = reward_fn(heldout_prompts, outputs)
    rollout = build_rollout_batch(tokenizer, heldout_prompts, outputs, config["max_seq_len"])
    device = next(policy.parameters()).device
    rollout = {key: value.to(device) for key, value in rollout.items()}
    policy_logprobs, valid_mask = response_token_logprobs(
        policy, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"]
    )
    ref_logprobs, _ = response_token_logprobs(
        reference, rollout["input_ids"], rollout["attention_mask"], rollout["response_mask"]
    )
    pref_acc_values = []
    for batch in heldout_loader:
        metrics = dpo_batch_metrics(policy, reference, batch, config["dpo"]["beta"])
        pref_acc_values.append(float(metrics["preference_accuracy"].item()))
    return {
        "heldout_rm_score": float(scores.mean().item()),
        "heldout_mean_kl": float(mean_masked_kl(policy_logprobs, ref_logprobs, valid_mask).item()),
        "heldout_preference_accuracy": sum(pref_acc_values) / max(1, len(pref_acc_values)),
    }




def run_dpo(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "dpo")
    logger = get_logger("train_dpo")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    start_checkpoint = _require_checkpoint(_resolve_sft_checkpoint(config), "SFT policy checkpoint for DPO")
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

    rm_checkpoint = config["models"].get("rm_checkpoint")
    heldout_prompts = [
        row["prompt"]
        for row in make_prompt_dataset(
            load_hh_dataset(config, config["data"]["hh_eval_split"], config["evaluation"]["prompts"])
        )
    ]
    heldout_loader = None
    reward_fn = None
    if rm_checkpoint:
        reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
        reward_model = load_reward_model(config, checkpoint=rm_checkpoint, trainable=False)
        reward_model.config.pad_token_id = int(reward_tokenizer.pad_token_id)
        reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])
        heldout_loader = DataLoader(
            make_dpo_dataset(load_hh_dataset(config, config["data"]["hh_eval_split"], config["evaluation"]["prompts"])),
            batch_size=config["dpo"]["batch_size"],
            shuffle=False,
            collate_fn=DPOCollator(policy_tokenizer, config["max_seq_len"]),
            num_workers=config["num_workers"],
        )

    log_every = _log_every(config, "dpo")
    policy.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    latest_metrics = None
    for epoch in range(config["dpo"]["epochs"]):
        progress = tqdm(loader, desc=f"DPO epoch {epoch + 1}/{config['dpo']['epochs']}")
        for step, batch in enumerate(progress, start=1):
            metrics = dpo_batch_metrics(policy, reference, batch, config["dpo"]["beta"])
            loss = metrics["loss"] / config["dpo"]["grad_accum"]
            loss.backward()
            latest_metrics = metrics
            if step % config["dpo"]["grad_accum"] == 0:
                grad_norm = _grad_norm(policy.parameters())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                row = {
                    "step": global_step,
                    "loss": float(metrics["loss"].item()),
                    "implicit_reward_margin": float(metrics["implicit_reward_margin"].item()),
                    "preference_accuracy": float(metrics["preference_accuracy"].item()),
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if reward_fn is not None and global_step % int(config.get("eval_every", 25)) == 0:
                    row.update(
                        _evaluate_dpo_holdout(
                            policy,
                            reference,
                            reward_fn,
                            policy_tokenizer,
                            heldout_prompts,
                            heldout_loader,
                            config,
                        )
                    )
                metric_logger.log(row)
                if global_step % log_every == 0:
                    suffix = ''
                    if 'heldout_rm_score' in row:
                        suffix = (
                            f" heldout_rm={row['heldout_rm_score']:.4f}"
                            f" heldout_kl={row['heldout_mean_kl']:.4f}"
                            f" heldout_pref={row['heldout_preference_accuracy']:.4f}"
                        )
                    emit_step_log(
                        logger,
                        f'[DPO {global_step}] loss={row["loss"]:.4f} z={row["implicit_reward_margin"]:.4f} '
                        f'pref_acc={row["preference_accuracy"]:.4f} grad_norm={row["grad_norm"]:.4f} lr={row["lr"]:.6g}{suffix}',
                    )
                progress.set_postfix(loss=row["loss"], acc=row["preference_accuracy"], grad=row["grad_norm"])
        if len(loader) % config["dpo"]["grad_accum"] != 0 and latest_metrics is not None:
            grad_norm = _grad_norm(policy.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            row = {
                "step": global_step,
                "loss": float(latest_metrics["loss"].item()),
                "implicit_reward_margin": float(latest_metrics["implicit_reward_margin"].item()),
                "preference_accuracy": float(latest_metrics["preference_accuracy"].item()),
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if reward_fn is not None and global_step % int(config.get("eval_every", 25)) == 0:
                row.update(
                    _evaluate_dpo_holdout(
                        policy,
                        reference,
                        reward_fn,
                        policy_tokenizer,
                        heldout_prompts,
                        heldout_loader,
                        config,
                    )
                )
            metric_logger.log(row)
            if global_step % log_every == 0:
                suffix = ''
                if 'heldout_rm_score' in row:
                    suffix = (
                        f" heldout_rm={row['heldout_rm_score']:.4f}"
                        f" heldout_kl={row['heldout_mean_kl']:.4f}"
                        f" heldout_pref={row['heldout_preference_accuracy']:.4f}"
                    )
                emit_step_log(
                    logger,
                    f'[DPO {global_step}] loss={row["loss"]:.4f} z={row["implicit_reward_margin"]:.4f} '
                    f'pref_acc={row["preference_accuracy"]:.4f} grad_norm={row["grad_norm"]:.4f} lr={row["lr"]:.6g}{suffix}',
                )

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    total_training_time_sec = time.perf_counter() - run_start_time
    save_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "stage": "dpo",
            "policy_checkpoint": str(run_dir),
            "total_training_time_sec": total_training_time_sec,
            "mean_step_time_sec": total_training_time_sec / max(1, global_step),
            "peak_vram_gb": _peak_vram_gb(),
        },
    )
    logger.info("Saved DPO adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def run_ppo(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "ppo")
    logger = get_logger("train_ppo")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_prompt_pool_samples"])
    )
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    start_checkpoint = _require_checkpoint(_resolve_sft_checkpoint(config), "SFT policy checkpoint for PPO")
    rm_checkpoint = _require_checkpoint(config["models"].get("rm_checkpoint"), "Reward-model checkpoint for PPO")
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)
    reward_model = load_reward_model(config, checkpoint=rm_checkpoint, trainable=False)
    reward_model.config.pad_token_id = int(reward_tokenizer.pad_token_id)
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
    log_every = _log_every(config, "ppo")

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
        for update_epoch in range(config["ppo"].get("update_epochs", 1)):
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
            metrics["update_epoch"] = update_epoch + 1
            metric_logger.log(metrics)
            if step % log_every == 0:
                emit_step_log(
                    logger,
                    f'[PPO {step}.{update_epoch + 1}] reward={metrics["mean_reward"]:.4f} kl={metrics["mean_kl"]:.4f} '
                    f'policy={metrics["policy_loss"]:.4f} value={metrics["value_loss"]:.4f} '
                    f'pol_grad={metrics["policy_grad_norm"]:.4f} val_grad={metrics["value_grad_norm"]:.4f} '
                    f'ratio={metrics["ratio_mean"]:.4f}',
                )
        if step % config["save_every"] == 0:
            checkpoint_dir = run_dir / f"checkpoint_step_{step}"
            policy.save_pretrained(checkpoint_dir)
            policy_tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    total_training_time_sec = time.perf_counter() - run_start_time
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "ppo", "policy_checkpoint": str(run_dir), "total_training_time_sec": total_training_time_sec, "mean_step_time_sec": total_training_time_sec / max(1, config["ppo"]["steps"]), "peak_vram_gb": _peak_vram_gb()})
    logger.info("Saved PPO adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def run_grpo(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "grpo")
    logger = get_logger("train_grpo")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts_dataset = make_prompt_dataset(
        load_hh_dataset(config, config["data"]["hh_train_split"], config["data"]["hh_prompt_pool_samples"])
    )
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    reward_tokenizer = load_reward_tokenizer(config["models"]["reward_name"])
    start_checkpoint = _require_checkpoint(_resolve_sft_checkpoint(config), "SFT policy checkpoint for GRPO")
    rm_checkpoint = _require_checkpoint(config["models"].get("rm_checkpoint"), "Reward-model checkpoint for GRPO")
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)
    reward_model = load_reward_model(config, checkpoint=rm_checkpoint, trainable=False)
    reward_model.config.pad_token_id = int(reward_tokenizer.pad_token_id)
    reward_fn = build_rm_reward_fn(reward_model, reward_tokenizer, config["max_seq_len"])

    optimizer = AdamW(
        filter(lambda param: param.requires_grad, policy.parameters()),
        lr=config["grpo"]["lr_policy"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )
    log_every = _log_every(config, "grpo")

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
        for update_epoch in range(config["grpo"].get("update_epochs", 1)):
            metrics = grpo_update_step(
                policy=policy,
                policy_optimizer=optimizer,
                rollout=rollout,
                epsilon=config["grpo"]["epsilon"],
                beta=config["grpo"]["beta"],
            )
            metrics["step"] = step
            metrics["update_epoch"] = update_epoch + 1
            metric_logger.log(metrics)
            if step % log_every == 0:
                emit_step_log(
                    logger,
                    f'[GRPO {step}.{update_epoch + 1}] reward={metrics["mean_reward"]:.4f} kl={metrics["mean_kl"]:.4f} '
                    f'policy={metrics["policy_loss"]:.4f} grad={metrics["policy_grad_norm"]:.4f} '
                    f'ratio={metrics["ratio_mean"]:.4f} degen={metrics["degenerate"]:.4f}',
                )
        if step % config["save_every"] == 0:
            checkpoint_dir = run_dir / f"checkpoint_step_{step}"
            policy.save_pretrained(checkpoint_dir)
            policy_tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    total_training_time_sec = time.perf_counter() - run_start_time
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "grpo", "policy_checkpoint": str(run_dir), "total_training_time_sec": total_training_time_sec, "mean_step_time_sec": total_training_time_sec / max(1, config["grpo"]["steps"]), "peak_vram_gb": _peak_vram_gb()})
    logger.info("Saved GRPO adapter to %s", run_dir)
    return {"run_dir": str(run_dir), "policy_checkpoint": str(run_dir)}


def run_rlvr(config: dict) -> dict:
    seed_everything(config["seed"])
    run_dir = make_run_dir(config["output_dir"], "rlvr")
    logger = get_logger("train_rlvr")
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    run_start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gsm_train = load_gsm8k_dataset(config, "train", config["data"]["gsm_train_samples"])
    gsm_eval = load_gsm8k_dataset(config, "test", config["data"]["gsm_eval_samples"])
    policy_tokenizer = load_policy_tokenizer(config["models"]["policy_name"])
    start_checkpoint = _require_checkpoint(_resolve_sft_checkpoint(config), "SFT policy checkpoint for RLVR")
    policy = load_policy_model(config, checkpoint=start_checkpoint, trainable=True)
    reference = load_reference_model(config, checkpoint=start_checkpoint)

    optimizer = AdamW(
        filter(lambda param: param.requires_grad, policy.parameters()),
        lr=config["rlvr"]["lr_policy"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_eps"],
    )
    log_every = _log_every(config, "rlvr")

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
        mean_response_length = _mean_response_length(policy_tokenizer, rollout.get("completions", []))
        for update_epoch in range(config["rlvr"].get("update_epochs", 1)):
            metrics = grpo_update_step(
                policy=policy,
                policy_optimizer=optimizer,
                rollout=rollout,
                epsilon=config["rlvr"]["epsilon"],
                beta=config["rlvr"]["beta"],
            )
            metrics["step"] = step
            metrics["update_epoch"] = update_epoch + 1
            metrics["mean_response_length"] = mean_response_length
            if step % 25 == 0 and update_epoch == config["rlvr"].get("update_epochs", 1) - 1:
                metrics["pass_at_1"] = _evaluate_rlvr_pass_at_1(
                    policy,
                    policy_tokenizer,
                    gsm_eval,
                    max_new_tokens=config["rlvr"]["max_new_tokens"],
                )
            metric_logger.log(metrics)
            if step % log_every == 0:
                suffix = ''
                if "pass_at_1" in metrics:
                    suffix = f' pass@1={metrics["pass_at_1"]:.4f}'
                emit_step_log(
                    logger,
                    f'[RLVR {step}.{update_epoch + 1}] reward={metrics["mean_reward"]:.4f} kl={metrics["mean_kl"]:.4f} '
                    f'policy={metrics["policy_loss"]:.4f} grad={metrics["policy_grad_norm"]:.4f} '
                    f'ratio={metrics["ratio_mean"]:.4f} degen={metrics["degenerate"]:.4f} '
                    f'mean_len={metrics["mean_response_length"]:.2f}{suffix}',
                )
        if step % config["save_every"] == 0:
            checkpoint_dir = run_dir / f"checkpoint_step_{step}"
            policy.save_pretrained(checkpoint_dir)
            policy_tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(run_dir)
    policy_tokenizer.save_pretrained(run_dir)
    total_training_time_sec = time.perf_counter() - run_start_time
    save_json(run_dir / "summary.json", {"run_dir": str(run_dir), "stage": "rlvr", "policy_checkpoint": str(run_dir), "total_training_time_sec": total_training_time_sec, "mean_step_time_sec": total_training_time_sec / max(1, config["rlvr"]["steps"]), "peak_vram_gb": _peak_vram_gb()})
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
