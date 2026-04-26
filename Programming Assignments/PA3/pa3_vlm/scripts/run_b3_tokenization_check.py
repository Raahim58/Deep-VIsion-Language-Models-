#!/usr/bin/env python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import AutoTokenizer

from pa3.common.config import parse_args, load_config, apply_quick_debug_b
from pa3.common.seed import seed_everything
from pa3.common.logging_utils import ensure_dirs, save_csv
from pa3.common.checkpointing import load_checkpoint
from pa3.data.part_a_cifar import read_rows_jsonl
from pa3.data.multimodal_tokenization import visual_ids_from_codes, encode_multimodal, encode_imagegen, token_type_sequence
from pa3.data.alpaca_replay import assert_no_visual_tokens
from pa3.data.part_b_synthetic import ImageTensorDataset
from pa3.models.vqvae import VQVAE


@torch.no_grad()
def preencode(model, images, labels, device):
    ds = ImageTensorDataset(images, labels); toks = []
    model.to(device).eval()
    for i in range(0, len(ds), 128):
        toks.append(model(ds.x[i:i+128].to(device))["idx"].reshape(-1, 16).cpu())
    model.to("cpu")
    return torch.cat(toks)


def main():
    args = parse_args("configs/part_b.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug: cfg = apply_quick_debug_b(cfg)
    out = Path(cfg["output_dir"]); ensure_dirs(out); seed_everything(cfg["seed"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = out / "cached_data" / "part_b"
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.add_tokens(["<image>", "</image>"] + [f"<vis_{i:03d}>" for i in range(256)], special_tokens=False)
    image_id, end_id, visual_start = tok.convert_tokens_to_ids("<image>"), tok.convert_tokens_to_ids("</image>"), cfg["model"]["v_txt"] + 2
    assert image_id == cfg["model"]["v_txt"] and end_id == cfg["model"]["v_txt"] + 1
    tr_x, tr_y = np.load(cache / "train_images.npy"), np.load(cache / "train_labels.npy")
    va_x, va_y = np.load(cache / "val_images.npy"), np.load(cache / "val_labels.npy")
    vq = VQVAE(k=256, d=64, beta=0.25); vq.load_state_dict(load_checkpoint(out / "checkpoints" / "vqvae_best.pt")["model"])
    train_codes, val_codes = preencode(vq, tr_x, tr_y, device), preencode(vq, va_x, va_y, device)
    torch.save({"train_codes": train_codes, "val_codes": val_codes}, cache / "vq_codes.pt")
    rows, imgrows = read_rows_jsonl(cache / "train_vqa.jsonl"), read_rows_jsonl(cache / "imggen.jsonl")
    examples = []
    txt = []
    for i in range(3):
        v_ids = visual_ids_from_codes(train_codes[rows[i]["image_idx"]], cfg["model"]["v_txt"])
        assert int(v_ids.min()) >= visual_start and int(v_ids.max()) <= visual_start + 255
        ids, lab = encode_multimodal(tok, v_ids, rows[i]["question"], rows[i]["answer"], image_id, end_id)
        ids2, lab2 = encode_imagegen(tok, v_ids, imgrows[i]["prompt"], image_id, end_id)
        for mode, a, b in [("vqa", ids, lab), ("img", ids2, lab2)]:
            line = f"{mode} {i}: types={token_type_sequence(a, tok, image_id, end_id, visual_start, 256)} input_ids={a.tolist()} labels={b.tolist()}"
            print(line); txt.append(line); examples.append({"mode": mode, "example": i, "min_visual": int(v_ids.min()), "max_visual": int(v_ids.max()), "padding_side": tok.padding_side})
    alpaca = json.loads((cache / "alpaca_texts.json").read_text())
    enc = tok(alpaca[:2], return_tensors="pt", padding=True, truncation=True)
    assert_no_visual_tokens(enc["input_ids"], visual_start)
    save_csv(out / "tables" / "part_b_tokenization_checks.csv", examples)
    (out / "logs" / "part_b_tokenization_examples.txt").write_text("\n".join(txt), encoding="utf-8")


if __name__ == "__main__":
    main()

