#!/usr/bin/env python
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

from pa3.common.config import parse_args, load_config, apply_quick_debug_a
from pa3.common.seed import seed_everything
from pa3.common.device import get_device, print_device
from pa3.common.logging_utils import ensure_dirs, log_jsonl, save_csv
from pa3.common.checkpointing import load_checkpoint, save_checkpoint
from pa3.common.param_count import print_trainable_report
from pa3.common.text_ppl import compute_ppl
from pa3.common.vram import vram_stats
from pa3.common.timing import StepTimer, phase_timer
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.data.alpaca_replay import assert_no_visual_tokens, collate_alpaca
from pa3.models.part_a_connector import MLPConnector
from pa3.models.part_a_vlm import build_vqa_inputs
from pa3.models.lora_utils import apply_lora
from pa3.train.datasets import AVQADataset, collate_a_vqa, infinite_loader
from pa3.eval.part_a_eval import eval_a_vqa


def load_alpaca_cache(path):
    return json.loads(Path(path).read_text())


def run_condition(cfg, out, device, lam, condition, args):
    cache = out / "cached_data" / "part_a"
    rows = read_rows_jsonl(cache / "train_vqa.jsonl")
    val = read_rows_jsonl(cache / "val_vqa.jsonl")
    clip = torch.load(cache / "train_clip_patches.pt", map_location="cpu")
    val_clip = torch.load(cache / "test_clip_patches.pt", map_location="cpu")
    alpaca = load_alpaca_cache(cache / "alpaca_texts.json")
    ppl0 = json.loads((out / "tables" / "part_a_ppl0.json").read_text())["PPL0"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    base.config.use_cache = False
    lm = apply_lora(base, cfg["train"]["lora_r"], cfg["train"]["lora_alpha"], cfg["train"]["lora_dropout"])
    connector = MLPConnector().to(device)
    ckpt = load_checkpoint(out / "checkpoints" / "part_a_connector_phaseA1.pt")
    connector.load_state_dict(ckpt["connector"])
    report_lm = print_trainable_report(lm, f"A-C2 LM {condition}")
    report_conn = print_trainable_report(connector, f"A-C2 connector {condition}")
    assert report_lm["trainable_pct"] < 1.0, "LoRA trainable percent should be < 1%"
    vqa_loader = DataLoader(AVQADataset(clip, rows, tokenizer), batch_size=cfg["train"]["batch_a2"], shuffle=True, collate_fn=collate_a_vqa(tokenizer))
    text_loader = DataLoader(alpaca, batch_size=max(1, cfg["train"]["batch_a2"] // 2), shuffle=True, collate_fn=collate_alpaca(tokenizer, cfg["model"]["v_txt"]))
    text_iter = infinite_loader(text_loader)
    max_steps = args.max_steps or len(vqa_loader)
    opt = torch.optim.AdamW([p for p in list(lm.parameters()) + list(connector.parameters()) if p.requires_grad], lr=cfg["train"]["lr_a2"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["train"]["lr_a2"], total_steps=max(1, max_steps // cfg["train"]["grad_accum"]), pct_start=0.1)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    timer = StepTimer(); first = True; skipped = 0; global_step = 0
    for batch in tqdm(vqa_loader, desc=f"A-C2 {condition}"):
        global_step += 1
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available(), dtype=torch.float16):
            emb, labels, attn = build_vqa_inputs(lm, connector, tokenizer, batch["clip"], batch["q_ids"], batch["a_ids"], device)
            if first:
                txt0 = next(text_iter); print("first VQA batch:", tuple(emb.shape), tuple(labels.shape)); print("first Alpaca batch:", tuple(txt0["input_ids"].shape)); assert_no_visual_tokens(txt0["input_ids"], cfg["model"]["v_txt"]); first = False
            loss_vqa = lm(inputs_embeds=emb, attention_mask=attn, labels=labels).loss
            txt = next(text_iter); txt = {k: v.to(device) for k, v in txt.items()}
            loss_lm = lm(**txt, labels=txt["input_ids"]).loss
            loss = (loss_vqa + lam * loss_lm) / cfg["train"]["grad_accum"]
        if not torch.isfinite(loss):
            skipped += 1; continue
        scaler.scale(loss).backward()
        if global_step % cfg["train"]["grad_accum"] == 0:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in list(lm.parameters()) + list(connector.parameters()) if p.requires_grad], 1.0).item()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        else:
            grad_norm = 0.0
        dt, eps = timer.tick(cfg["train"]["batch_a2"])
        if global_step % 25 == 0 or global_step == 1:
            row = {"phase": "A-C2", "condition": condition, "epoch": 0, "global_step": global_step, "lr": opt.param_groups[0]["lr"], "loss_total": float((loss_vqa + lam * loss_lm).detach().cpu()), "loss_vqa": float(loss_vqa.detach().cpu()), "loss_lm": float(loss_lm.detach().cpu()), "grad_norm": grad_norm, "scaler_scale": scaler.get_scale(), "skipped_nan_step": False, "nan_skip_count": skipped, "step_time": dt, "examples_sec": eps, "elapsed_minutes": 0, **vram_stats()}
            print(row); log_jsonl(out / "logs" / f"part_a_phase2_{condition}.jsonl", row)
        if global_step % cfg["train"]["eval_every"] == 0:
            acc, _ = eval_a_vqa(lm, connector, tokenizer, val, val_clip, device, n=200)
            ppl, _ = compute_ppl(lm, tokenizer, alpaca[:100], device, desc=f"A-C2 {condition} PPL")
            print("eval:", {"acc": acc, "PPL_fine": ppl, "R": ppl / ppl0})
        if global_step >= max_steps: break
    acc, _ = eval_a_vqa(lm, connector, tokenizer, val, val_clip, device, n=200)
    ppl, _ = compute_ppl(lm, tokenizer, alpaca[:min(300, len(alpaca))], device, desc=f"A-C2 {condition} final PPL")
    result = {"condition": condition, "lambda": lam, "VQA_acc_percent": 100 * acc, "PPL0": ppl0, "PPL_fine": ppl, "R": ppl / ppl0, "trainable_params": report_lm["trainable_params"] + report_conn["trainable_params"], "peak_vram_gb": vram_stats()["vram_peak_gb"], "elapsed_minutes": 0}
    save_checkpoint(out / "checkpoints" / f"part_a_phaseA2_{condition}.pt", lm=lm.state_dict(), connector=connector.state_dict(), result=result)
    if condition == "baseline":
        save_checkpoint(out / "checkpoints" / "part_a_phaseA2.pt", lm=lm.state_dict(), connector=connector.state_dict(), result=result)
    return result


def main():
    args = parse_args("configs/part_a.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_a(cfg); args.max_steps = args.max_steps or 2
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    with phase_timer("A-C2 sft_replay"):
        print_device()
        rows = []
        for lam, cond in [(0.0, "no_replay"), (0.05, "weak"), (0.2, "baseline"), (0.5, "strong")]:
            rows.append(run_condition(cfg, out, device, lam, cond, args))
        save_csv(out / "tables" / "part_a_lambda_ablation.csv", rows)


if __name__ == "__main__":
    main()

