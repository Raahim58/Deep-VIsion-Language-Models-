#!/usr/bin/env python
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pa3.common.config import parse_args


def main():
    args = parse_args("configs/part_b.yaml")
    ab = args.ablation or "no_projector"
    print("Running Part B ablation:", ab)
    if ab in {"codebook_size", "frozen_vqvae", "lora_rank", "loss_weight", "no_projector", "frozen_embedding"}:
        cmd = [sys.executable, "scripts/run_b4_mixed_finetune.py", "--config", args.config, "--output_dir", args.output_dir or "outputs"]
        if args.quick_debug: cmd.append("--quick_debug")
        if args.max_steps: cmd += ["--max_steps", str(args.max_steps)]
        subprocess.run(cmd, check=True)
        print(f"Saved ablation-compatible tables under outputs/tables/part_b_mixed_ablation.csv; copy/rename to part_b_ablation_{ab}.csv after sweep.")
    else:
        raise ValueError(f"Unknown ablation {ab}")


if __name__ == "__main__":
    main()

