#!/usr/bin/env python
import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from pa3.common.config import parse_args, load_config, apply_quick_debug_b
from pa3.common.seed import seed_everything
from pa3.common.device import get_device
from pa3.common.logging_utils import ensure_dirs, save_csv
from pa3.common.checkpointing import load_checkpoint, save_checkpoint
from pa3.common.param_count import print_trainable_report
from pa3.common.timing import phase_timer
from pa3.common.vram import print_vram
from pa3.models.vqvae import VQVAE
from pa3.models.overlay_embedding import install_overlay, visual_text_norm_ratio


class Projector(nn.Module):
    def __init__(self, d=64, out=960):
        super().__init__(); self.proj = nn.Linear(d, out); nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5)); nn.init.zeros_(self.proj.bias)
    def forward(self, x): return self.proj(x)


def main():
    args = parse_args("configs/part_b.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_b(cfg)
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    with phase_timer("B-C2 vocab_projector"):
        tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        print("original vocab size:", len(tok))
        added = tok.add_tokens(["<image>", "</image>"] + [f"<vis_{i:03d}>" for i in range(256)], special_tokens=False)
        print("added:", added, "new vocab size:", len(tok))
        assert len(tok) == 49410
        ids = {"image": tok.convert_tokens_to_ids("<image>"), "end_image": tok.convert_tokens_to_ids("</image>"), "first_visual": cfg["model"]["v_txt"] + 2, "last_visual": cfg["model"]["v_txt"] + 257}
        print("token IDs:", ids)
        lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        lm.resize_token_embeddings(len(tok))
        overlay = install_overlay(lm, cfg["model"]["v_txt"], 258).to(device)
        print("overlay embedding shape:", tuple(overlay.overlay.weight.shape))
        vq = VQVAE(k=256, d=64, beta=0.25).to(device); vq.load_state_dict(load_checkpoint(out / "checkpoints" / "vqvae_best.pt")["model"]); vq.eval()
        codebook = vq.quantizer.codebook.weight.detach().float()
        proj = Projector(64, cfg["model"]["d_lm"]).to(device)
        print_trainable_report(proj, "projector before transplant")
        opt = torch.optim.AdamW(proj.parameters(), lr=1e-3)
        target = overlay.base.weight[:cfg["model"]["v_txt"]].float().norm(dim=-1).mean().detach()
        for step in range(10 if args.quick_debug else 200):
            opt.zero_grad(set_to_none=True); z = proj(codebook.to(device)); loss = (z.norm(dim=-1).mean() - target).pow(2); loss.backward(); opt.step()
            if step % 25 == 0 or step == 0: print({"step": step, "norm_loss": float(loss.detach().cpu()), "visual_text_ratio": float(z.norm(dim=-1).mean().detach().cpu()/target.cpu())})
        with torch.no_grad():
            overlay.overlay.weight[2:].copy_(proj(codebook.to(device)).to(overlay.overlay.weight.dtype))
        ratio = visual_text_norm_ratio(overlay, 256)
        if ratio < 0.2 or ratio > 5:
            with torch.no_grad(): overlay.overlay.weight[2:].mul_(1.0 / max(ratio, 1e-8))
            ratio = visual_text_norm_ratio(overlay, 256)
        print_trainable_report(overlay, "overlay after transplant")
        save_checkpoint(out / "checkpoints" / "part_b_projector_warmup.pt", overlay=overlay.overlay.state_dict(), projector=proj.state_dict(), token_ids=ids, norm_ratio=ratio)
        save_csv(out / "tables" / "part_b_vocab_projector_summary.csv", [{"original_vocab": cfg["model"]["v_txt"], "new_vocab": len(tok), **ids, "norm_ratio": ratio}])
        print_vram("B-C2")


if __name__ == "__main__":
    main()

