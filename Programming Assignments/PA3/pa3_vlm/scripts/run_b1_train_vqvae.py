#!/usr/bin/env python
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pa3.common.config import parse_args, load_config, apply_quick_debug_b
from pa3.common.seed import seed_everything
from pa3.common.device import get_device
from pa3.common.logging_utils import ensure_dirs, log_jsonl, save_csv
from pa3.common.checkpointing import save_checkpoint
from pa3.common.param_count import print_trainable_report
from pa3.common.plotting import save_show
from pa3.common.vram import vram_stats, print_vram
from pa3.common.timing import phase_timer
from pa3.data.part_b_synthetic import ImageTensorDataset
from pa3.data.part_b_templates import SYN_CLASSES
from pa3.models.vqvae import VQVAE


@torch.no_grad()
def analyse(model, val_ds, out, name="baseline"):
    model.eval(); device = next(model.parameters()).device
    x = torch.stack([val_ds[np.where(val_ds.y.numpy() == cid)[0][0]][0] for cid in range(6)]).to(device)
    z = model.encoder(x); pre = model.decoder(z); outq = model(x); post = outq["recon"]
    gap = float((F.mse_loss(post, x) - F.mse_loss(pre, x)).cpu())
    print("quantization gap:", gap)
    idx_all = []
    for i in range(0, len(val_ds), 128):
        idx_all.append(model(val_ds.x[i:i+128].to(device))["idx"].reshape(-1).cpu())
    usage = torch.bincount(torch.cat(idx_all), minlength=model.quantizer.k).numpy()
    fig, ax = plt.subplots(figsize=(8, 3)); ax.bar(range(len(usage)), usage); ax.set_title("Codebook usage"); save_show(fig, out / "plots" / "part_b_codebook_usage.png")
    w = F.normalize(model.quantizer.codebook.weight.detach().float().cpu(), dim=-1); fig, ax = plt.subplots(figsize=(5,5)); im = ax.imshow((w@w.T).numpy(), vmin=-1, vmax=1, cmap="coolwarm"); fig.colorbar(im, ax=ax); save_show(fig, out / "plots" / "part_b_codebook_cosine_heatmap.png")
    fig, axes = plt.subplots(6, 3, figsize=(7, 10))
    for i in range(6):
        axes[i,0].imshow(x[i].permute(1,2,0).cpu()); axes[i,1].imshow(post[i].permute(1,2,0).cpu()); axes[i,2].imshow(outq["idx"][i].cpu(), cmap="tab20")
        for j in range(3): axes[i,j].axis("off")
    save_show(fig, out / "samples" / "part_b_token_maps.png")
    return gap


def train_one(cfg, out, device, name, k, beta, ema, args):
    tr_x = np.load(out / "cached_data" / "part_b" / "train_images.npy"); tr_y = np.load(out / "cached_data" / "part_b" / "train_labels.npy")
    va_x = np.load(out / "cached_data" / "part_b" / "val_images.npy"); va_y = np.load(out / "cached_data" / "part_b" / "val_labels.npy")
    tr, va = ImageTensorDataset(tr_x, tr_y), ImageTensorDataset(va_x, va_y)
    model = VQVAE(k=k, d=cfg["vqvae"]["d"], beta=beta, ema=ema, decay=cfg["vqvae"]["ema_decay"], dead_threshold=cfg["vqvae"]["dead_threshold"]).to(device)
    print_trainable_report(model, f"VQVAE {name}")
    params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + ([] if ema else list(model.quantizer.codebook.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg["vqvae"]["lr"])
    loader = DataLoader(tr, batch_size=cfg["vqvae"]["batch_size"], shuffle=True)
    val_loader = DataLoader(va, batch_size=cfg["vqvae"]["batch_size"])
    best = 1e9; rows = []
    for ep in range(cfg["vqvae"]["epochs"]):
        for step, (x, _) in enumerate(tqdm(loader, desc=f"VQVAE {name} ep{ep}")):
            x = x.to(device); opt.zero_grad(set_to_none=True); o = model(x)
            if ep == 0 and step == 0:
                print("ze/zq/idx shapes:", tuple(o["ze"].shape), tuple(o["zq"].shape), tuple(o["idx"].shape))
            o["loss"].backward(); grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item(); opt.step()
            if args.max_steps and step + 1 >= args.max_steps: break
        vals = []
        with torch.no_grad():
            for x, _ in val_loader:
                vals.append(float(model(x.to(device))["recon_mse"].cpu()))
        rec = {"phase": "B-C1", "config": name, "epoch": ep, "global_step": ep, "lr": cfg["vqvae"]["lr"], "loss_total": float(o["loss"].detach().cpu()), "recon_mse": float(np.mean(vals)), "codebook_loss": float(o["codebook_loss"].detach().cpu()), "commitment_loss": float(o["commitment_loss"].detach().cpu()), "codebook_perplexity": float(o["codebook_perplexity"].detach().cpu()), "dead_codes": o["dead_codes"], "grad_norm": grad_norm, "scaler_scale": 1.0, "skipped_nan_step": False, **vram_stats()}
        print(rec); rows.append(rec); log_jsonl(out / "logs" / f"part_b_vqvae_{name}.jsonl", rec)
        if rec["recon_mse"] < best:
            best = rec["recon_mse"]; save_checkpoint(out / "checkpoints" / ("vqvae_best.pt" if name == "baseline" else f"vqvae_{name}.pt"), model=model.state_dict(), config={"k": k, "beta": beta, "ema": ema})
    gap = analyse(model, va, out, name) if name == "baseline" else float("nan")
    return {"config": name, "K": k, "beta": beta, "update": "EMA" if ema else "Gradient", "final_recon_mse": best, "perplexity": rows[-1]["codebook_perplexity"], "dead_codes": rows[-1]["dead_codes"], "quantization_gap": gap}


def main():
    args = parse_args("configs/part_b.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_b(cfg); args.max_steps = args.max_steps or 2
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    with phase_timer("B-C1 vqvae"):
        rows = []
        for name, k, beta, ema in [("baseline",256,0.25,False), ("K128",128,0.25,False), ("beta1",256,1.0,False), ("EMA",256,0.25,True)]:
            rows.append(train_one(cfg, out, device, name, k, beta, ema, args))
        save_csv(out / "tables" / "part_b_vqvae_ablation.csv", rows)
        print_vram("B-C1")


if __name__ == "__main__":
    main()

