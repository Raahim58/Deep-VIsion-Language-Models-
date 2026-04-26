#!/usr/bin/env python
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import CLIPImageProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

from pa3.common.config import parse_args, load_config, apply_quick_debug_a
from pa3.common.seed import seed_everything
from pa3.common.device import get_device, print_device
from pa3.common.logging_utils import ensure_dirs, save_csv, save_json
from pa3.common.timing import phase_timer
from pa3.common.vram import print_vram
from pa3.common.text_ppl import compute_ppl
from pa3.data.part_a_cifar import load_cifar_subsets, cache_clip_pixels, cache_clip_patch_tokens, save_rows_jsonl
from pa3.data.part_a_templates import make_captions, make_vqa
from pa3.data.alpaca_replay import load_alpaca_texts
from pa3.models.part_a_vlm import freeze


def main():
    args = parse_args("configs/part_a.yaml")
    cfg = load_config(args.config, args.output_dir)
    if args.quick_debug:
        cfg = apply_quick_debug_a(cfg)
    out = Path(cfg["output_dir"])
    ensure_dirs(out)
    seed_everything(cfg["seed"])
    device = get_device()
    t0 = time.time()
    with phase_timer("A-C0 setup"):
        print_device()
        cache = out / "cached_data" / "part_a"
        cache.mkdir(parents=True, exist_ok=True)
        processor = CLIPImageProcessor.from_pretrained(cfg["model"]["clip_name"])
        print("CLIP mean/std:", processor.image_mean, processor.image_std)
        print("ImageNet mean/std:", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        print("CLIP differs from ImageNet:", processor.image_mean != [0.485, 0.456, 0.406] or processor.image_std != [0.229, 0.224, 0.225])
        train, test, train_idx, test_idx = load_cifar_subsets(cache / "torchvision", cfg["data"]["train_per_class"], cfg["data"]["test_per_class"], cfg["seed"])
        train_pixels, train_labels = cache_clip_pixels(train, train_idx, processor, cache / "train_clip_pixels.pt")
        test_pixels, test_labels = cache_clip_pixels(test, test_idx, processor, cache / "test_clip_pixels.pt")
        captions = make_captions(train_labels)
        test_captions = make_captions(test_labels)
        train_vqa = make_vqa(train_labels)
        val_vqa = make_vqa(test_labels)
        print("Example captions:", captions[:3])
        print("Example VQA pairs:", train_vqa[:5])
        print("Expected/actual VQA sizes:", len(train_vqa), len(val_vqa))
        save_rows_jsonl(cache / "train_captions.jsonl", captions)
        save_rows_jsonl(cache / "test_captions.jsonl", test_captions)
        save_rows_jsonl(cache / "train_vqa.jsonl", train_vqa)
        save_rows_jsonl(cache / "val_vqa.jsonl", val_vqa)
        save_csv(out / "tables" / "part_a_data_summary.csv", [{"train_images": len(train_labels), "test_images": len(test_labels), "train_vqa": len(train_vqa), "val_vqa": len(val_vqa)}])

        clip = CLIPModel.from_pretrained(cfg["model"]["clip_name"]).to(device).float()
        freeze(clip)
        with torch.no_grad():
            h = clip.vision_model(pixel_values=train_pixels[:2].to(device)).last_hidden_state
        print("CLIP hidden before CLS discard:", tuple(h.shape))
        print("CLIP hidden after CLS discard:", tuple(h[:, 1:, :].shape))
        cache_clip_patch_tokens(clip, train_pixels, device, cache / "train_clip_patches.pt")
        cache_clip_patch_tokens(clip, test_pixels, device, cache / "test_clip_patches.pt")

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["lm_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lm_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        lm = AutoModelForCausalLM.from_pretrained(cfg["model"]["lm_name"], torch_dtype=lm_dtype).to(device)
        print("SmolLM2 hidden/vocab:", lm.config.hidden_size, lm.config.vocab_size)
        assert lm.config.hidden_size == cfg["model"]["d_lm"]
        assert lm.config.vocab_size == cfg["model"]["v_txt"]
        alpaca = load_alpaca_texts(cfg["data"]["alpaca_n"])
        save_json(cache / "alpaca_texts.json", alpaca)
        ppl0, loss0 = compute_ppl(lm, tokenizer, alpaca, device, desc="A PPL0")
        save_json(out / "tables" / "part_a_ppl0.json", {"PPL0": ppl0, "loss0": loss0})
        print_vram("A-C0 peak")
        print("elapsed_minutes:", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()

