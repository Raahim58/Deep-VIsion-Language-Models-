#!/usr/bin/env python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from pa3.common.config import parse_args, load_config, apply_quick_debug_a
from pa3.common.seed import seed_everything
from pa3.common.device import get_device, print_device
from pa3.common.logging_utils import ensure_dirs, save_csv
from pa3.common.checkpointing import load_checkpoint
from pa3.common.plotting import save_show
from pa3.common.generation import topk_tokens
from pa3.common.vram import print_vram
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.models.part_a_connector import MLPConnector
from pa3.models.lora_utils import apply_lora
from pa3.eval.part_a_eval import eval_a_vqa, a_top5


def load_phase(cfg, out, device, phase):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    conn = MLPConnector().to(device)
    if phase == "phase1":
        ck = load_checkpoint(out / "checkpoints" / "part_a_connector_phaseA1.pt")
        conn.load_state_dict(ck["connector"])
        for p in lm.parameters(): p.requires_grad_(False)
    else:
        lm = apply_lora(lm, cfg["train"]["lora_r"], cfg["train"]["lora_alpha"], cfg["train"]["lora_dropout"])
        path = out / "checkpoints" / ("part_a_phaseA2.pt" if phase == "phase2" else "part_a_connector_phaseA3.pt")
        ck = load_checkpoint(path)
        lm.load_state_dict(ck["lm"], strict=False); conn.load_state_dict(ck["connector"])
    lm.eval(); conn.eval()
    return lm, conn, tok


def main():
    args = parse_args("configs/part_a.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_a(cfg)
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device(); print_device()
    rows = read_rows_jsonl(out / "cached_data" / "part_a" / "val_vqa.jsonl")
    if args.quick_debug: rows = rows[:50]
    clip = torch.load(out / "cached_data" / "part_a" / "test_clip_patches.pt", map_location="cpu")
    summary = []
    for phase in ["phase1", "phase2", "phase3"]:
        try:
            lm, conn, tok = load_phase(cfg, out, device, phase)
        except FileNotFoundError:
            continue
        acc, df = eval_a_vqa(lm, conn, tok, rows, clip, device)
        summary.append({"phase": phase, "VQA_acc_percent": 100 * acc, "R": load_checkpoint(out / "checkpoints" / ("part_a_phaseA2.pt" if phase=="phase2" else "part_a_connector_phaseA3.pt"), map_location="cpu").get("result", {}).get("R", None) if phase != "phase1" else None})
        save_csv(out / "tables" / f"part_a_{phase}_eval_rows.csv", df.to_dict("records"))
        save_csv(out / "tables" / "part_a_per_template.csv", df.groupby("template").correct.mean().reset_index(name="acc").to_dict("records"))
        save_csv(out / "tables" / "part_a_per_class.csv", df.groupby("class").correct.mean().reset_index(name="acc").to_dict("records"))
        sample = pd.concat([df[df.correct].head(4), df[~df.correct].head(2)]).head(6)
        if len(sample):
            top = a_top5(lm, conn, tok, sample.iloc[0].to_dict(), clip, device)
            print("top-5 logits:", top)
    save_csv(out / "tables" / "part_a_eval_summary.csv", summary)
    if summary:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["phase"] for r in summary], [r["VQA_acc_percent"] for r in summary], marker="o", label="VQA_acc_percent")
        ax2 = ax.twinx()
        ax2.plot([r["phase"] for r in summary], [r["R"] if r["R"] is not None else 0 for r in summary], marker="x", color="orange", label="R")
        ax.set_title("Part A accuracy and R by phase")
        save_show(fig, out / "plots" / "part_a_accuracy_R_by_phase.png")
    print("majority baseline:", df.answer.value_counts(normalize=True).iloc[0] if "df" in locals() else None)
    print_vram("A-C4")


if __name__ == "__main__":
    main()

