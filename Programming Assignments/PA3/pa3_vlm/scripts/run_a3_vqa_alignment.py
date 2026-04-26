#!/usr/bin/env python
import sys, json
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
from pa3.common.logging_utils import ensure_dirs, log_jsonl
from pa3.common.checkpointing import load_checkpoint, save_checkpoint
from pa3.common.param_count import print_trainable_report
from pa3.common.text_ppl import compute_ppl
from pa3.common.vram import vram_stats
from pa3.common.timing import phase_timer
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.models.part_a_connector import MLPConnector
from pa3.models.part_a_vlm import build_vqa_inputs
from pa3.models.lora_utils import apply_lora
from pa3.train.datasets import AVQADataset, collate_a_vqa
from pa3.eval.part_a_eval import eval_a_vqa


def main():
    args = parse_args("configs/part_a.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_a(cfg); args.max_steps = args.max_steps or 2
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    with phase_timer("A-C3 vqa_alignment"):
        print_device()
        cache = out / "cached_data" / "part_a"
        train = read_rows_jsonl(cache / "train_vqa.jsonl")[:10000]
        val = read_rows_jsonl(cache / "val_vqa.jsonl")
        clip = torch.load(cache / "train_clip_patches.pt", map_location="cpu")
        val_clip = torch.load(cache / "test_clip_patches.pt", map_location="cpu")
        alpaca = json.loads((cache / "alpaca_texts.json").read_text())
        ppl0 = json.loads((out / "tables" / "part_a_ppl0.json").read_text())["PPL0"]
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        base.config.use_cache = False
        lm = apply_lora(base, cfg["train"]["lora_r"], cfg["train"]["lora_alpha"], cfg["train"]["lora_dropout"])
        conn = MLPConnector().to(device)
        ckpt_path = out / "checkpoints" / "part_a_phaseA2.pt"
        ckpt = load_checkpoint(ckpt_path)
        lm.load_state_dict(ckpt["lm"], strict=False); conn.load_state_dict(ckpt["connector"])
        print("loaded checkpoint path:", ckpt_path)
        print_trainable_report(lm, "A-C3 LM"); print_trainable_report(conn, "A-C3 connector")
        loader = DataLoader(AVQADataset(clip, train, tokenizer), batch_size=cfg["train"]["batch_a2"], shuffle=True, collate_fn=collate_a_vqa(tokenizer))
        opt = torch.optim.AdamW([p for p in list(lm.parameters()) + list(conn.parameters()) if p.requires_grad], lr=cfg["train"]["lr_a3"])
        scaler = GradScaler(enabled=torch.cuda.is_available())
        step = 0
        for batch in tqdm(loader, desc="A-C3"):
            step += 1; opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available(), dtype=torch.float16):
                emb, labels, attn = build_vqa_inputs(lm, conn, tokenizer, batch["clip"], batch["q_ids"], batch["a_ids"], device)
                loss = lm(inputs_embeds=emb, attention_mask=attn, labels=labels).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in list(lm.parameters()) + list(conn.parameters()) if p.requires_grad], 1.0).item()
            scaler.step(opt); scaler.update()
            if step % 25 == 0 or step == 1:
                row = {"phase": "A-C3", "epoch": 0, "global_step": step, "lr": opt.param_groups[0]["lr"], "loss_total": float(loss.detach().cpu()), "loss_vqa": float(loss.detach().cpu()), "grad_norm": grad_norm, "scaler_scale": scaler.get_scale(), "skipped_nan_step": False, **vram_stats()}
                print(row); log_jsonl(out / "logs" / "part_a_phase3.jsonl", row)
            if args.max_steps and step >= args.max_steps: break
        acc, _ = eval_a_vqa(lm, conn, tokenizer, val, val_clip, device, n=200)
        ppl, _ = compute_ppl(lm, tokenizer, alpaca[:min(300, len(alpaca))], device, desc="A-C3 PPL_fine")
        result = {"phase2_acc": ckpt["result"]["VQA_acc_percent"], "phase2_R": ckpt["result"]["R"], "phase3_acc_percent": 100 * acc, "PPL_fine": ppl, "R": ppl / ppl0, "R_increased": (ppl / ppl0) > ckpt["result"]["R"]}
        print(result)
        save_checkpoint(out / "checkpoints" / "part_a_connector_phaseA3.pt", lm=lm.state_dict(), connector=conn.state_dict(), result=result)


if __name__ == "__main__":
    main()

