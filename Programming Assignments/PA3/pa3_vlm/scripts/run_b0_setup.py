#!/usr/bin/env python
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pa3.common.config import parse_args, load_config, apply_quick_debug_b
from pa3.common.seed import seed_everything
from pa3.common.device import get_device, print_device
from pa3.common.logging_utils import ensure_dirs, save_csv, save_json
from pa3.common.text_ppl import compute_ppl
from pa3.common.timing import phase_timer
from pa3.common.vram import print_vram
from pa3.data.part_b_synthetic import generate_dataset, stratified_split, save_grid
from pa3.data.part_b_templates import make_b_vqa, make_img_prompts
from pa3.data.part_a_cifar import save_rows_jsonl
from pa3.data.alpaca_replay import load_alpaca_texts


def main():
    args = parse_args("configs/part_b.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_b(cfg)
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = get_device()
    with phase_timer("B-C0 setup"):
        print_device()
        cache = out / "cached_data" / "part_b"; cache.mkdir(parents=True, exist_ok=True)
        images, labels = generate_dataset(cfg["data"]["n_per_class"], cfg["seed"])
        tr_x, tr_y, va_x, va_y = stratified_split(images, labels, cfg["seed"])
        np.save(cache / "train_images.npy", tr_x); np.save(cache / "train_labels.npy", tr_y)
        np.save(cache / "val_images.npy", va_x); np.save(cache / "val_labels.npy", va_y)
        save_grid(tr_x, tr_y, out / "plots" / "part_b_synthetic_grid.png")
        train_vqa, val_vqa = make_b_vqa(torch.tensor(tr_y)), make_b_vqa(torch.tensor(va_y))
        imggen = make_img_prompts(torch.tensor(tr_y))
        print("Example B VQA:", train_vqa[:5])
        print("Example image-gen prompts:", imggen[:5])
        print("B VQA sizes:", len(train_vqa), len(val_vqa))
        save_rows_jsonl(cache / "train_vqa.jsonl", train_vqa); save_rows_jsonl(cache / "val_vqa.jsonl", val_vqa); save_rows_jsonl(cache / "imggen.jsonl", imggen)
        tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"; print("tokenizer padding side:", tok.padding_side)
        lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        print("SmolLM2 vocab/hidden:", lm.config.vocab_size, lm.config.hidden_size)
        assert lm.config.vocab_size == cfg["model"]["v_txt"]; assert lm.config.hidden_size == cfg["model"]["d_lm"]
        alpaca = load_alpaca_texts(cfg["data"]["alpaca_n"]); save_json(cache / "alpaca_texts.json", alpaca)
        ppl0, loss0 = compute_ppl(lm, tok, alpaca, device, desc="B PPL0")
        save_json(out / "tables" / "part_b_ppl0.json", {"PPL0": ppl0, "loss0": loss0})
        save_csv(out / "tables" / "part_b_data_summary.csv", [{"train": len(tr_y), "val": len(va_y), "vqa_train": len(train_vqa), "vqa_val": len(val_vqa), "PPL0": ppl0}])
        print_vram("B-C0")


if __name__ == "__main__":
    main()

