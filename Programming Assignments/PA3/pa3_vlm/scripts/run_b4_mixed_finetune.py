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

from pa3.common.config import parse_args, load_config, apply_quick_debug_b
from pa3.common.seed import seed_everything
from pa3.common.device import get_device
from pa3.common.logging_utils import ensure_dirs, log_jsonl, save_csv
from pa3.common.checkpointing import load_checkpoint, save_checkpoint
from pa3.common.param_count import print_trainable_report
from pa3.common.text_ppl import compute_ppl
from pa3.common.vram import vram_stats
from pa3.common.timing import phase_timer
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.data.alpaca_replay import collate_alpaca, assert_no_visual_tokens
from pa3.data.multimodal_tokenization import visual_ids_from_codes, encode_multimodal, encode_imagegen
from pa3.models.overlay_embedding import install_overlay
from pa3.models.lora_utils import apply_lora
from pa3.train.datasets import TokenDataset, collate_token, infinite_loader
from pa3.eval.part_b_eval import eval_b_vqa
from pa3.common.plotting import save_show
import matplotlib.pyplot as plt


def build_encoded(tok, rows, imgrows, codes, cfg, token_ids):
    vqa, img = [], []
    for r in rows:
        v = visual_ids_from_codes(codes[r["image_idx"]], cfg["model"]["v_txt"])
        ids, lab = encode_multimodal(tok, v, r["question"], r["answer"], token_ids["image"], token_ids["end_image"])
        vqa.append({"input_ids": ids, "labels": lab, "row": r})
    for r in imgrows:
        v = visual_ids_from_codes(codes[r["image_idx"]], cfg["model"]["v_txt"])
        ids, lab = encode_imagegen(tok, v, r["prompt"], token_ids["image"], token_ids["end_image"])
        img.append({"input_ids": ids, "labels": lab, "row": r})
    return vqa, img


def load_model(cfg, out, device, r):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.add_tokens(["<image>", "</image>"] + [f"<vis_{i:03d}>" for i in range(256)], special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    lm.resize_token_embeddings(len(tok))
    overlay = install_overlay(lm, cfg["model"]["v_txt"], 258).to(device)
    overlay.overlay.load_state_dict(load_checkpoint(out / "checkpoints" / "part_b_projector_warmup.pt")["overlay"])
    lm.config.use_cache = False
    lm = apply_lora(lm, r, cfg["train"]["lora_alpha"], cfg["train"]["lora_dropout"])
    lm.get_input_embeddings().overlay.weight.requires_grad_(True)
    return lm, tok, {"image": cfg["model"]["v_txt"], "end_image": cfg["model"]["v_txt"] + 1, "visual_start": cfg["model"]["v_txt"] + 2}


def train_condition(cfg, out, device, lam, gamma, r, name, args):
    lm, tok, token_ids = load_model(cfg, out, device, r)
    if hasattr(lm, "gradient_checkpointing_enable"): lm.gradient_checkpointing_enable()
    report = print_trainable_report(lm, f"B-C4 {name}")
    assert report["trainable_pct"] < 1.0
    cache = out / "cached_data" / "part_b"
    rows = read_rows_jsonl(cache / "train_vqa.jsonl"); val = read_rows_jsonl(cache / "val_vqa.jsonl"); imgrows = read_rows_jsonl(cache / "imggen.jsonl")
    codes = torch.load(cache / "vq_codes.pt", map_location="cpu")["train_codes"]; val_codes = torch.load(cache / "vq_codes.pt", map_location="cpu")["val_codes"]
    vqa_enc, img_enc = build_encoded(tok, rows, imgrows, codes, cfg, token_ids)
    vqa_loader = DataLoader(TokenDataset(vqa_enc), batch_size=cfg["train"]["batch_lm"], shuffle=True, collate_fn=collate_token(tok))
    img_loader = DataLoader(TokenDataset(img_enc), batch_size=cfg["train"]["batch_lm"], shuffle=True, collate_fn=collate_token(tok))
    alpaca = json.loads((cache / "alpaca_texts.json").read_text())
    text_loader = DataLoader(alpaca, batch_size=max(1, cfg["train"]["batch_lm"]//2), shuffle=True, collate_fn=collate_alpaca(tok, token_ids["visual_start"]))
    vi, ii, ti = infinite_loader(vqa_loader), infinite_loader(img_loader), infinite_loader(text_loader)
    total = args.max_steps or cfg["train"]["epochs"] * max(len(vqa_loader), len(img_loader), len(text_loader))
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=cfg["train"]["lr_lora"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["train"]["lr_lora"], total_steps=max(1, total // cfg["train"]["grad_accum"]), pct_start=0.1)
    scaler = GradScaler(enabled=torch.cuda.is_available()); opt.zero_grad(set_to_none=True); skipped = 0
    for step in tqdm(range(1, total + 1), desc=f"B-C4 {name}"):
        b1, b2, b3 = next(vi), next(ii), next(ti)
        if step == 1:
            print("first VQA/img/Alpaca batch:", tuple(b1["input_ids"].shape), tuple(b2["input_ids"].shape), tuple(b3["input_ids"].shape)); assert_no_visual_tokens(b3["input_ids"], token_ids["visual_start"])
        b1 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b1.items()}; b2 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b2.items()}; b3 = {k: v.to(device) for k, v in b3.items()}
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available(), dtype=torch.float16):
            lv = lm(input_ids=b1["input_ids"], attention_mask=b1["attention_mask"], labels=b1["labels"]).loss; scaler.scale(lv / cfg["train"]["grad_accum"]).backward()
            li = lm(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"], labels=b2["labels"]).loss; scaler.scale(gamma * li / cfg["train"]["grad_accum"]).backward()
            lt = lm(**b3, labels=b3["input_ids"]).loss; scaler.scale(lam * lt / cfg["train"]["grad_accum"]).backward()
        if step % cfg["train"]["grad_accum"] == 0:
            scaler.unscale_(opt); grad_norm = torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0).item(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        else: grad_norm = 0.0
        if step % 25 == 0 or step == 1:
            row = {"phase": "B-C4", "condition": name, "epoch": 0, "global_step": step, "lr": opt.param_groups[0]["lr"], "loss_total": float((lv + gamma*li + lam*lt).detach().cpu()), "loss_vqa": float(lv.detach().cpu()), "loss_img": float(li.detach().cpu()), "loss_lm": float(lt.detach().cpu()), "grad_norm": grad_norm, "scaler_scale": scaler.get_scale(), "skipped_nan_step": False, "nan_skip_count": skipped, **vram_stats()}
            print(row); log_jsonl(out / "logs" / f"part_b_mixed_{name}.jsonl", row)
        if step % cfg["train"]["eval_every"] == 0:
            acc, _ = eval_b_vqa(lm, tok, val, val_codes, token_ids, device, 200)
            print("eval acc:", acc)
    acc, _ = eval_b_vqa(lm, tok, val, val_codes, token_ids, device, 200)
    ppl0 = json.loads((out / "tables" / "part_b_ppl0.json").read_text())["PPL0"]; ppl, _ = compute_ppl(lm, tok, alpaca[:min(100, len(alpaca))], device, desc=f"B {name} PPL")
    res = {"condition": name, "lambda": lam, "gamma_img": gamma, "rank": r, "VQA_acc_percent": 100*acc, "PPL0": ppl0, "PPL_fine": ppl, "R": ppl/ppl0}
    save_checkpoint(out / "checkpoints" / ("part_b_mixed.pt" if name == "baseline" else f"part_b_mixed_{name}.pt"), model=lm.state_dict(), token_ids=token_ids, result=res)
    return res


def main():
    args = parse_args("configs/part_b.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_b(cfg); args.max_steps = args.max_steps or 2
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    with phase_timer("B-C4 mixed"):
        rows = []
        for lam, gam, name, r in [(0,0,"no_replay",16),(0.05,0.05,"weak",16),(0.2,0.5,"baseline",16),(0.5,0.5,"strong",16),(0,0,"break_protection_r64",64)]:
            rows.append(train_condition(cfg, out, device, lam, gam, r, name, args))
        save_csv(out / "tables" / "part_b_mixed_ablation.csv", rows)
        save_csv(out / "tables" / "part_b_break_protection.csv", [r for r in rows if "break" in r["condition"]])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["condition"] for r in rows], [r["VQA_acc_percent"] for r in rows], marker="o", label="VQA_acc_percent")
        ax2 = ax.twinx(); ax2.plot([r["condition"] for r in rows], [r["R"] for r in rows], marker="x", color="orange", label="R")
        ax.set_xticklabels([r["condition"] for r in rows], rotation=30, ha="right")
        ax.set_title("Part B R and VQA by condition")
        save_show(fig, out / "plots" / "part_b_R_vqa_by_condition.png")
        fig, ax = plt.subplots(figsize=(7, 4)); ax.set_title("Part B training losses are logged in JSONL"); ax.axis("off")
        save_show(fig, out / "plots" / "part_b_training_losses.png")


if __name__ == "__main__":
    main()
