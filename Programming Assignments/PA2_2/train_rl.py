"""
RL training script: DPO | PPO | GRPO | RLVR.

Usage:
    python train_rl.py --method dpo   [--config configs/dpo.yaml]
    python train_rl.py --method ppo   [--config configs/ppo.yaml]
    python train_rl.py --method grpo  [--config configs/grpo.yaml]
    python train_rl.py --method rlvr  [--config configs/rlvr.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config, merge_configs
from utils.seed import set_seed
from utils.memory import print_memory, clear_cache
from utils.logging_utils import get_logger, MetricsLogger, log_metrics
from utils.io import ensure_dir, save_peft_checkpoint
from model.loading import load_policy, load_reward_backbone, get_tokenizer
from model.lora import apply_lora, freeze_model
from model.reward_model import RewardModel, score_texts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      required=True, choices=["dpo", "ppo", "grpo", "rlvr"])
    p.add_argument("--config",      default=None)
    p.add_argument("--sft_ckpt",    default=None, help="Path to SFT PEFT adapter dir")
    p.add_argument("--rm_ckpt",     default=None, help="Path to saved RM checkpoint (.pt)")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--run_dir",     default=None)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main(args):
    method = args.method
    if args.config is None:
        args.config = f"configs/{method}.yaml"
    if args.run_dir is None:
        args.run_dir = f"runs/{method}"

    logger  = get_logger(method)
    set_seed(args.seed)
    default_cfg = load_config("configs/default.yaml")
    task_cfg    = load_config(args.config)
    cfg         = merge_configs(default_cfg, task_cfg)
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir     = ensure_dir(args.run_dir)
    mlog        = MetricsLogger(run_dir / "metrics.jsonl")

    # ── Policy and tokenizer ──────────────────────────────────────────────────
    logger.info("Loading policy …")
    policy = load_policy(
        cfg["policy_model"], dtype=cfg.get("dtype", "bfloat16"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        device_map=device,
    )
    policy = apply_lora(
        policy,
        r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16),
        dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    )
    if args.sft_ckpt:
        from peft import PeftModel
        logger.info(f"Loading SFT adapter from {args.sft_ckpt}")
        # If policy is already a PeftModel, load adapter weights
        policy.load_adapter(args.sft_ckpt, adapter_name="default")

    policy_tok = get_tokenizer(cfg["policy_model"], padding_side="left")

    # ── Reference model (frozen copy) ────────────────────────────────────────
    logger.info("Loading reference model …")
    ref_model = load_policy(
        cfg["policy_model"], dtype=cfg.get("dtype", "bfloat16"),
        gradient_checkpointing=False, device_map=device,
    )
    if args.sft_ckpt:
        from peft import PeftModel
        ref_model = apply_lora(ref_model, r=cfg.get("lora_r", 8),
                               alpha=cfg.get("lora_alpha", 16),
                               dropout=0.0,
                               target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]))
        ref_model.load_adapter(args.sft_ckpt, adapter_name="default")
    freeze_model(ref_model)

    # ── Route to method ───────────────────────────────────────────────────────
    if method == "dpo":
        _run_dpo(cfg, policy, ref_model, policy_tok, device, run_dir, mlog, logger, args)
    elif method == "ppo":
        _run_ppo(cfg, policy, ref_model, policy_tok, device, run_dir, mlog, logger, args)
    elif method == "grpo":
        _run_grpo(cfg, policy, ref_model, policy_tok, device, run_dir, mlog, logger, args)
    elif method == "rlvr":
        _run_rlvr(cfg, policy, ref_model, policy_tok, device, run_dir, mlog, logger, args)

    save_peft_checkpoint(policy, run_dir / "adapter")
    print_memory()
    mlog.close()
    logger.info(f"{method.upper()} training complete.")
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# DPO
# ─────────────────────────────────────────────────────────────────────────────

def _run_dpo(cfg, policy, ref_model, tok, device, run_dir, mlog, logger, args):
    from data.hh_rlhf import load_hh_rlhf, DPODataset
    from data.collators import DPOCollator
    from alignment.dpo import dpo_step

    train_data = load_hh_rlhf("train", max_samples=args.max_samples)
    collator   = DPOCollator(tok, cfg.get("max_seq_len", 1024))
    loader     = DataLoader(DPODataset(train_data),
                            batch_size=cfg.get("batch_size", 8),
                            shuffle=True, collate_fn=collator)

    optimizer  = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.get("lr", 5e-5), weight_decay=0.01,
    )
    grad_accum  = cfg.get("grad_accum", 4)
    epochs      = cfg.get("epochs", 1)
    total_steps = (len(loader) // grad_accum) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, cfg.get("warmup_steps", 30), total_steps
    )
    beta       = cfg.get("beta", 0.1)
    log_every  = cfg.get("log_every", 10)
    eval_every = cfg.get("eval_every", 25)

    global_step  = 0
    running_info = {}

    for epoch in range(epochs):
        policy.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"DPO epoch {epoch+1}")):
            info = dpo_step(policy, ref_model, batch, optimizer, beta=beta)
            for k, v in info.items():
                running_info[k] = running_info.get(k, 0) + v

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg = {k: v / log_every for k, v in running_info.items()}
                    log_metrics(logger, global_step, avg, prefix="DPO")
                    mlog.log(global_step, avg)
                    running_info = {}

                if global_step % eval_every == 0:
                    logger.info(f"[step {global_step}] checkpoint logged")


# ─────────────────────────────────────────────────────────────────────────────
# PPO
# ─────────────────────────────────────────────────────────────────────────────

def _run_ppo(cfg, policy, ref_model, tok, device, run_dir, mlog, logger, args):
    from data.hh_rlhf import load_hh_rlhf, PromptDataset
    from model.value_model import ValueModel
    from model.reward_model import RewardModel
    from alignment.ppo import ppo_rollout, ppo_update, ppo_sanity_checks

    ppo_sanity_checks()

    train_data   = load_hh_rlhf("train", max_samples=args.max_samples)
    prompt_ds    = PromptDataset(train_data)
    rm_tok       = AutoTokenizer.from_pretrained(cfg["reward_model"])
    rm_tok.padding_side = "right"
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token

    logger.info("Loading reward model …")
    rm = RewardModel.load_frozen(cfg["reward_model"], use_8bit=cfg.get("use_8bit_frozen", True))
    if args.rm_ckpt:
        from utils.io import load_checkpoint
        load_checkpoint(rm, args.rm_ckpt, strict=False)

    logger.info("Loading value model …")
    vm = ValueModel(cfg["value_model"], freeze_backbone=True,
                    use_8bit=False)

    pol_opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.get("lr_policy", 1e-5),
    )
    val_opt = torch.optim.AdamW(
        [p for p in vm.parameters() if p.requires_grad],
        lr=cfg.get("lr_value", 1e-4),
    )

    prompts_per_step = cfg.get("prompts_per_step", 8)
    update_steps     = cfg.get("update_steps", 200)
    log_every        = cfg.get("log_every", 5)
    eval_every       = cfg.get("eval_every", 25)
    sampler          = RandomSampler(prompt_ds, replacement=True,
                                     num_samples=update_steps * prompts_per_step)

    all_prompts = [prompt_ds[i] for i in range(len(prompt_ds))]
    import random

    for step in tqdm(range(update_steps), desc="PPO"):
        batch_prompts = random.sample(all_prompts, prompts_per_step)

        clear_cache()
        rollout = ppo_rollout(
            policy_model=policy, ref_model=ref_model,
            value_model=vm, reward_model=rm, rm_tokenizer=rm_tok,
            policy_tokenizer=tok, prompts=batch_prompts,
            max_new_tokens=cfg.get("max_new_tokens", 128),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            beta=cfg.get("beta", 0.1),
            gamma=cfg.get("gamma", 1.0),
            lam=cfg.get("lam", 0.95),
            device=device,
        )

        info = ppo_update(
            policy, vm, rollout, pol_opt, val_opt,
            ppo_epochs=cfg.get("ppo_epochs", 4),
            epsilon=cfg.get("epsilon", 0.2),
            c_value=cfg.get("c_value", 0.5),
            device=device,
        )
        info["mean_task_reward"] = sum(rollout.task_rewards) / len(rollout.task_rewards)

        if (step + 1) % log_every == 0:
            log_metrics(logger, step + 1, info, prefix="PPO")
            mlog.log(step + 1, info)

        if (step + 1) % eval_every == 0:
            logger.info(f"[step {step+1}] PPO eval checkpoint")


# ─────────────────────────────────────────────────────────────────────────────
# GRPO
# ─────────────────────────────────────────────────────────────────────────────

def _run_grpo(cfg, policy, ref_model, tok, device, run_dir, mlog, logger, args):
    from data.hh_rlhf import load_hh_rlhf, PromptDataset
    from model.reward_model import RewardModel, score_texts
    from alignment.grpo import grpo_rollout, grpo_update
    import random

    train_data = load_hh_rlhf("train", max_samples=args.max_samples)
    all_prompts = [d["prompt"] for d in train_data]

    rm_tok = AutoTokenizer.from_pretrained(cfg["reward_model"])
    rm_tok.padding_side = "right"
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token

    rm = RewardModel.load_frozen(cfg["reward_model"], use_8bit=cfg.get("use_8bit_frozen", True))
    if args.rm_ckpt:
        from utils.io import load_checkpoint
        load_checkpoint(rm, args.rm_ckpt, strict=False)

    def reward_fn(texts):
        return score_texts(rm, rm_tok, texts)

    optimizer    = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-5),
    )
    update_steps     = cfg.get("update_steps", 200)
    prompts_per_step = cfg.get("prompts_per_step", 8)
    K                = cfg.get("K", 4)
    log_every        = cfg.get("log_every", 5)

    for step in tqdm(range(update_steps), desc="GRPO"):
        batch_prompts = random.sample(all_prompts, prompts_per_step)
        clear_cache()

        rollout = grpo_rollout(
            policy_model=policy, ref_model=ref_model,
            reward_fn=reward_fn, policy_tokenizer=tok,
            prompts=batch_prompts, K=K,
            max_new_tokens=cfg.get("max_new_tokens", 128),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            device=device,
        )
        info = grpo_update(
            policy, ref_model, rollout, optimizer,
            epsilon=cfg.get("epsilon", 0.2),
            beta=cfg.get("beta", 0.1),
            device=device,
        )

        if (step + 1) % log_every == 0:
            log_metrics(logger, step + 1, info, prefix="GRPO")
            mlog.log(step + 1, info)


# ─────────────────────────────────────────────────────────────────────────────
# RLVR
# ─────────────────────────────────────────────────────────────────────────────

def _run_rlvr(cfg, policy, ref_model, tok, device, run_dir, mlog, logger, args):
    from data.gsm8k import load_gsm8k, GSM8KDataset
    from alignment.rlvr import rlvr_rollout, eval_gsm8k_pass_at_1
    from alignment.grpo import grpo_update
    import random

    train_items = load_gsm8k("train", max_samples=args.max_samples)
    test_items  = load_gsm8k("test")

    optimizer    = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-5),
    )
    update_steps     = cfg.get("update_steps", 300)
    prompts_per_step = cfg.get("prompts_per_step", 8)
    K                = cfg.get("K", 4)
    log_every        = cfg.get("log_every", 5)
    eval_every       = cfg.get("eval_every", 25)

    for step in tqdm(range(update_steps), desc="RLVR"):
        batch = random.sample(train_items, prompts_per_step)
        prompts      = [it["prompt"]      for it in batch]
        gold_answers = [it["gold_answer"] for it in batch]
        clear_cache()

        rollout = rlvr_rollout(
            policy_model=policy, ref_model=ref_model,
            policy_tokenizer=tok, prompts=prompts,
            gold_answers=gold_answers, K=K,
            max_new_tokens=cfg.get("max_new_tokens", 256),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            device=device,
        )
        info = grpo_update(
            policy, ref_model, rollout, optimizer,
            epsilon=cfg.get("epsilon", 0.2),
            beta=cfg.get("beta", 0.05),
            device=device,
        )

        if (step + 1) % log_every == 0:
            log_metrics(logger, step + 1, info, prefix="RLVR")
            mlog.log(step + 1, info)

        if (step + 1) % eval_every == 0:
            pass1 = eval_gsm8k_pass_at_1(
                policy, tok, test_items, max_new_tokens=cfg.get("max_new_tokens", 256),
                device=device, max_samples=100,
            )
            logger.info(f"[step {step+1}] GSM8K pass@1 = {pass1:.4f}")
            mlog.log(step + 1, {"pass_at_1": pass1})


if __name__ == "__main__":
    main(parse_args())
