#!/usr/bin/env python
import argparse, subprocess, sys

SCRIPTS = [
    "scripts/run_b0_setup.py",
    "scripts/run_b1_train_vqvae.py",
    "scripts/run_b2_vocab_projector.py",
    "scripts/run_b3_tokenization_check.py",
    "scripts/run_b4_mixed_finetune.py",
    "scripts/run_b5_eval.py",
    "scripts/run_b6_ablation.py",
]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--quick_debug", action="store_true"); p.add_argument("--output_dir", default="outputs"); p.add_argument("--config", default="configs/part_b.yaml")
    args = p.parse_args()
    for s in SCRIPTS:
        cmd = [sys.executable, s, "--config", args.config, "--output_dir", args.output_dir]
        if args.quick_debug: cmd += ["--quick_debug", "--max_steps", "2"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

