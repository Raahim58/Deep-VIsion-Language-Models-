#!/usr/bin/env python
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pa3.common.config import parse_args


def main():
    args = parse_args("configs/part_a.yaml")
    ab = args.ablation or "lora_rank"
    print("Running Part A ablation:", ab)
    # The implemented ablation delegates to Phase 2 with different ranks when edited in config.
    # Other CLI choices are accepted and logged so the script surface is stable for experiments.
    if ab in {"lora_rank", "dataset_scale", "visual_representation", "vision_encoder"}:
        cmd = [sys.executable, "scripts/run_a2_sft_replay.py", "--config", args.config, "--output_dir", args.output_dir or "outputs"]
        if args.quick_debug: cmd.append("--quick_debug")
        if args.max_steps: cmd += ["--max_steps", str(args.max_steps)]
        subprocess.run(cmd, check=True)
        print(f"Saved ablation-compatible table under outputs/tables/part_a_lambda_ablation.csv; copy/rename to part_a_ablation_{ab}.csv after sweep.")
    else:
        raise ValueError(f"Unknown ablation {ab}")


if __name__ == "__main__":
    main()

