#!/usr/bin/env python
import argparse, subprocess, sys

SCRIPTS = [
    "scripts/run_a0_setup.py",
    "scripts/run_a1_connector_init.py",
    "scripts/run_a2_sft_replay.py",
    "scripts/run_a3_vqa_alignment.py",
    "scripts/run_a4_eval.py",
    "scripts/run_a5_modality_gap.py",
    "scripts/run_a6_ablation.py",
]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--quick_debug", action="store_true"); p.add_argument("--output_dir", default="outputs"); p.add_argument("--config", default="configs/part_a.yaml")
    args = p.parse_args()
    for s in SCRIPTS:
        cmd = [sys.executable, s, "--config", args.config, "--output_dir", args.output_dir]
        if args.quick_debug: cmd += ["--quick_debug", "--max_steps", "2"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

