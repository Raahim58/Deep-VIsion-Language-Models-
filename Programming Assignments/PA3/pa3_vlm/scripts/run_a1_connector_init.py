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
from pa3.common.logging_utils import ensure_dirs, log_jsonl
from pa3.common.checkpointing import save_checkpoint
from pa3.common.param_count import print_trainable_report
from pa3.common.timing import StepTimer, phase_timer
from pa3.common.vram import vram_stats
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.models.part_a_connector import MLPConnector
from pa3.models.part_a_vlm import build_caption_inputs, freeze
from pa3.train.datasets import CaptionDataset, collate_caption


def rnorm(connector, lm, tokenizer, clip_tokens, captions, device):
    connector.eval()
    with torch.no_grad():
        v = connector(clip_tokens[:128].to(device)).float()
        ids = tokenizer(captions[:128], return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids.to(device)
        t = lm.get_input_embeddings()(ids).float()
        mask = ids.ne(tokenizer.pad_token_id)
        ratio = v.norm(dim=-1).mean().item() / max(t[mask].norm(dim=-1).mean().item(), 1e-8)
    print("rnorm:", ratio)
    return ratio


@torch.no_grad()
def generate_caption(lm, connector, tokenizer, clip_token, device):
    bos = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
    emb = torch.cat([lm.get_input_embeddings()(bos), connector(clip_token[None].to(device))], dim=1)
    out = lm.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=device, dtype=torch.long), max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    args = parse_args("configs/part_a.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug:
        cfg = apply_quick_debug_a(cfg)
        args.max_steps = args.max_steps or 2
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    t0 = time.time()
    with phase_timer("A-C1 connector_init"):
        print_device()
        cache = out / "cached_data" / "part_a"
        rows = read_rows_jsonl(cache / "train_captions.jsonl")
        test_rows = read_rows_jsonl(cache / "test_captions.jsonl")
        clip_tokens = torch.load(cache / "train_clip_patches.pt", map_location="cpu")
        test_tokens = torch.load(cache / "test_clip_patches.pt", map_location="cpu")
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        freeze(lm)
        connector = MLPConnector().to(device)
        report = print_trainable_report(connector, "Phase1 connector")
        loader = DataLoader(CaptionDataset(clip_tokens, rows, tokenizer), batch_size=cfg["train"]["batch_a1"], shuffle=True, collate_fn=collate_caption(tokenizer))
        opt = torch.optim.AdamW(connector.parameters(), lr=cfg["train"]["lr_a1"])
        scaler = GradScaler(enabled=torch.cuda.is_available())
        timer = StepTimer(); first = True; global_step = 0; skipped = 0
        for epoch in range(1):
            for batch in tqdm(loader, desc="A-C1"):
                global_step += 1
                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available(), dtype=torch.float16):
                    emb, labels, attn = build_caption_inputs(lm, connector, tokenizer, batch["clip"], batch["caption_ids"], device)
                    if first:
                        print("first input_embeds shape:", tuple(emb.shape))
                        print("first labels shape:", tuple(labels.shape))
                        print("ignored labels:", int((labels == -100).sum().item()))
                        first = False
                    loss = lm(inputs_embeds=emb, attention_mask=attn, labels=labels).loss
                if not torch.isfinite(loss):
                    skipped += 1; continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0).item()
                scaler.step(opt); scaler.update()
                dt, eps = timer.tick(batch["clip"].size(0))
                if global_step % 25 == 0 or global_step == 1:
                    row = {"phase": "A-C1", "epoch": epoch, "global_step": global_step, "lr": opt.param_groups[0]["lr"], "loss_total": float(loss.detach().cpu()), "grad_norm": grad_norm, "scaler_scale": scaler.get_scale(), "skipped_nan_step": False, "step_time": dt, "examples_sec": eps, "elapsed_minutes": (time.time()-t0)/60, **vram_stats()}
                    print(row); log_jsonl(out / "logs" / "part_a_phase1.jsonl", row)
                if args.max_steps and global_step >= args.max_steps: break
            if args.max_steps and global_step >= args.max_steps: break
        before = rnorm(connector, lm, tokenizer, clip_tokens, [r["caption"] for r in rows], device)
        after = before
        if before < 0.3 or before > 3.0:
            scale = 1.0 / max(before, 1e-8)
            with torch.no_grad():
                connector.net[-1].weight.mul_(scale); connector.net[-1].bias.mul_(scale)
            print("rescaled connector output by:", scale)
            after = rnorm(connector, lm, tokenizer, clip_tokens, [r["caption"] for r in rows], device)
        for i in range(min(5, len(test_tokens))):
            print("generated caption:", test_rows[i]["class"], generate_caption(lm, connector, tokenizer, test_tokens[i], device))
        save_checkpoint(out / "checkpoints" / "part_a_connector_phaseA1.pt", connector=connector.state_dict(), rnorm_before=before, rnorm_after=after, trainable_report=report)
        print("nan/skipped count:", skipped)


if __name__ == "__main__":
    main()

