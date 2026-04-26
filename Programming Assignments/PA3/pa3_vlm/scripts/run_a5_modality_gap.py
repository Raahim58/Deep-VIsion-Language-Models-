#!/usr/bin/env python
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

from pa3.common.config import parse_args, load_config, apply_quick_debug_a
from pa3.common.seed import seed_everything
from pa3.common.device import get_device
from pa3.common.logging_utils import ensure_dirs, save_csv
from pa3.common.checkpointing import load_checkpoint
from pa3.common.plotting import save_show
from pa3.common.vram import print_vram
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.models.part_a_connector import MLPConnector
from pa3.models.lora_utils import apply_lora


@torch.no_grad()
def stats(conn, lm, tok, clip, rows, img_idx, q_idx, device):
    v = conn(clip[img_idx].to(device)).mean(1).float()
    q = [rows[i]["question"] for i in q_idx]
    ids = tok(q, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    t = lm.get_input_embeddings()(ids).float()
    mask = ids.ne(tok.pad_token_id)
    t = torch.stack([t[i, mask[i]].mean(0) for i in range(t.size(0))])
    v = F.normalize(v, dim=-1); t = F.normalize(t, dim=-1)
    eyev = ~torch.eye(v.size(0), dtype=torch.bool, device=device)
    return {"MG": float((v.mean(0)-t.mean(0)).norm().cpu()), "within_visual": float((v@v.T)[eyev].mean().cpu()), "within_text": float((t@t.T)[eyev].mean().cpu()), "cross_modal": float((v@t.T).mean().cpu())}, v.cpu(), t.cpu()


def main():
    args = parse_args("configs/part_a.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_a(cfg)
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    rows = read_rows_jsonl(out / "cached_data" / "part_a" / "val_vqa.jsonl")
    clip = torch.load(out / "cached_data" / "part_a" / "test_clip_patches.pt", map_location="cpu")
    rng = np.random.default_rng(42)
    n = min(200, len(clip), len(rows))
    img_idx = rng.choice(len(clip), n, replace=False).tolist()
    q_idx = rng.choice(len(rows), n, replace=False).tolist()
    print("fixed indices preview:", img_idx[:10], q_idx[:10], "hash:", hashlib.md5(str((img_idx, q_idx)).encode()).hexdigest())
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    results = []
    for phase, path in [("phase1", "part_a_connector_phaseA1.pt"), ("phase2", "part_a_phaseA2.pt"), ("phase3", "part_a_connector_phaseA3.pt")]:
        ckpath = out / "checkpoints" / path
        if not ckpath.exists(): continue
        lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        if phase != "phase1":
            lm = apply_lora(lm, cfg["train"]["lora_r"], cfg["train"]["lora_alpha"], cfg["train"]["lora_dropout"])
        conn = MLPConnector().to(device); ck = load_checkpoint(ckpath)
        conn.load_state_dict(ck["connector"])
        if "lm" in ck: lm.load_state_dict(ck["lm"], strict=False)
        row, v, t = stats(conn, lm, tok, clip, rows, img_idx, q_idx, device); row["phase"] = phase
        print(row); results.append(row)
        xy = PCA(n_components=2, random_state=42).fit_transform(torch.cat([v, t]).numpy())
        fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(xy[:n,0], xy[:n,1], s=10, label="visual"); ax.scatter(xy[n:,0], xy[n:,1], s=10, label="text"); ax.legend(); ax.set_title(phase)
        save_show(fig, out / "plots" / f"part_a_embedding_pca_{phase}.png")
    save_csv(out / "tables" / "part_a_modality_gap.csv", results)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot([r["phase"] for r in results], [r["MG"] for r in results], marker="o"); ax.set_title("Part A MG by phase")
    save_show(fig, out / "plots" / "part_a_MG_by_phase.png")
    print("Lnorm comparison: run A-C2 with Lnorm term weight=1e-2; hook point is loss_vqa + lambda*loss_lm in run_a2_sft_replay.py.")
    print_vram("A-C5")


if __name__ == "__main__":
    main()

