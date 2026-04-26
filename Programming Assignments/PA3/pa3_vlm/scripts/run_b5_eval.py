#!/usr/bin/env python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForCausalLM

from pa3.common.config import parse_args, load_config, apply_quick_debug_b
from pa3.common.seed import seed_everything
from pa3.common.device import get_device
from pa3.common.logging_utils import ensure_dirs, save_csv
from pa3.common.checkpointing import load_checkpoint
from pa3.common.plotting import save_show
from pa3.common.generation import topk_tokens
from pa3.common.vram import print_vram
from pa3.common.metrics import normalize_answer
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.models.overlay_embedding import install_overlay
from pa3.models.lora_utils import apply_lora
from pa3.models.vqvae import VQVAE
from pa3.models.logit_masks import mask_text_logits
from pa3.eval.part_b_eval import eval_b_vqa, generate_image_tokens
from pa3.data.part_b_templates import SYN_CLASSES


def load_model(cfg, out, device):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.add_tokens(["<image>", "</image>"] + [f"<vis_{i:03d}>" for i in range(256)], special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    lm.resize_token_embeddings(len(tok)); ov = install_overlay(lm, cfg["model"]["v_txt"], 258).to(device)
    lm = apply_lora(lm, cfg["train"]["lora_r"], cfg["train"]["lora_alpha"], cfg["train"]["lora_dropout"])
    ck = load_checkpoint(out / "checkpoints" / "part_b_mixed.pt"); lm.load_state_dict(ck["model"], strict=False)
    return lm.eval(), tok, ck["token_ids"]


@torch.no_grad()
def decode(vq, token_ids, visual_ids, device):
    idx = (visual_ids - token_ids["visual_start"]).clamp(0,255).view(1,4,4).to(device)
    z = vq.quantizer.codebook(idx.reshape(-1)).view(1,4,4,64).permute(0,3,1,2)
    return vq.decoder(z)[0].permute(1,2,0).cpu().numpy()


def main():
    args = parse_args("configs/part_b.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_b(cfg)
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    rows = read_rows_jsonl(out / "cached_data" / "part_b" / "val_vqa.jsonl")
    codes = torch.load(out / "cached_data" / "part_b" / "vq_codes.pt", map_location="cpu")["val_codes"]
    lm, tok, token_ids = load_model(cfg, out, device)
    n = len(rows) if args.full_eval else (50 if args.quick_debug else 500)
    acc, df = eval_b_vqa(lm, tok, rows, codes, token_ids, device, n)
    save_csv(out / "tables" / "part_b_vqa_eval.csv", df.to_dict("records"))
    save_csv(out / "tables" / "part_b_per_template.csv", df.groupby("template").correct.mean().reset_index(name="acc").to_dict("records"))
    save_csv(out / "tables" / "part_b_per_class.csv", df.groupby("class").correct.mean().reset_index(name="acc").to_dict("records"))
    print("overall VQA accuracy:", acc)
    print("majority baseline:", rows and max(np.mean([normalize_answer(r["answer"]) == a for r in rows]) for a in set(normalize_answer(r["answer"]) for r in rows)))
    shape = df[df.template == "shape"]
    if len(shape):
        cm = confusion_matrix(shape.answer.map(normalize_answer), shape.pred, labels=SYN_CLASSES)
        fig, ax = plt.subplots(figsize=(6,6)); ConfusionMatrixDisplay(cm, display_labels=SYN_CLASSES).plot(ax=ax, xticks_rotation=45); save_show(fig, out / "plots" / "part_b_confusion_matrix.png")
    vq = VQVAE(k=256,d=64,beta=0.25).to(device); vq.load_state_dict(load_checkpoint(out / "checkpoints" / "vqvae_best.pt")["model"]); vq.eval()
    fig, axes = plt.subplots(2,6,figsize=(10,4)); raw=masked=None
    for i, cls in enumerate(SYN_CLASSES):
        for rep in range(2):
            vt, raw, masked = generate_image_tokens(lm, tok, f"Generate a {cls} image.", token_ids, device, 1.0)
            axes[rep,i].imshow(decode(vq, token_ids, vt, device)); axes[rep,i].axis("off"); axes[rep,i].set_title(cls)
    save_show(fig, out / "samples" / "part_b_generated_12.png")
    fig, ax = plt.subplots(1,2,figsize=(8,3)); ax[0].hist(raw.numpy(), bins=50); ax[1].hist(masked.numpy(), bins=50); save_show(fig, out / "plots" / "part_b_logit_hist_before_after.png")
    fig, axes = plt.subplots(1,3,figsize=(8,3))
    for j,T in enumerate([0.5,1.0,1.5]):
        vt,_,_ = generate_image_tokens(lm, tok, "Generate a circle image.", token_ids, device, T); axes[j].imshow(decode(vq, token_ids, vt, device)); axes[j].axis("off"); axes[j].set_title(f"T={T}")
    save_show(fig, out / "samples" / "part_b_temperature_sweep.png")
    if len(df):
        r = df.iloc[0].to_dict(); v = codes[r["image_idx"]].long().to(device) + token_ids["visual_start"]; q = tok(r["question"]+" Answer:", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        prefix = torch.cat([torch.tensor([[tok.bos_token_id or tok.eos_token_id, token_ids["image"]]], device=device), v[None], torch.tensor([[token_ids["end_image"]]], device=device), q], 1)
        print("top-5 logits:", topk_tokens(mask_text_logits(lm(input_ids=prefix).logits[0,-1], token_ids["visual_start"], 256), tok))
    print_vram("B-C5")


if __name__ == "__main__":
    main()

