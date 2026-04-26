# AI623 PA3 Vision Language Models

Script-first implementation for the coding pipeline only. The existing notebook in the parent directory is left untouched; this project lives under `pa3_vlm/`.

## Install

```bash
cd pa3_vlm
pip install -r requirements.txt
```

## Quick Debug

```bash
python scripts/run_all_part_a.py --quick_debug
python scripts/run_all_part_b.py --quick_debug
```

Quick debug uses tiny data and two batches per phase to check shapes, labels, token IDs, checkpoints, and logs.

## Full Runs

```bash
python scripts/run_all_part_a.py
python scripts/run_all_part_b.py
```

Individual phases support:

```bash
python scripts/run_a1_connector_init.py --config configs/part_a.yaml --resume --max_steps 100 --output_dir outputs
python scripts/run_b4_mixed_finetune.py --config configs/part_b.yaml --resume --max_steps 300 --output_dir outputs
```

## Notebook Runner

```bash
jupyter notebook notebooks/PA3_runner.ipynb
```

The notebook runner calls the scripts, shows log snippets, plots, summary tables, and samples. It does not hide implementation logic.

## Outputs

All generated files are saved under:

- `outputs/logs/*.jsonl`
- `outputs/checkpoints/*.pt`
- `outputs/plots/*.png`
- `outputs/tables/*.csv`
- `outputs/samples/*.png`
- `outputs/cached_data/`

## Expected Runtime and VRAM

- Part A: about 25-30 min on a Kaggle T4, under 2.5 GB peak VRAM.
- Part B: about 100 min on a Kaggle T4, under 3.5 GB peak VRAM.

## Common Failure Fixes

- OOM: reduce batch size to 8 and increase gradient accumulation to 8.
- Wrong CLIP features: use `CLIPImageProcessor` and discard CLS with `last_hidden_state[:, 1:, :]`.
- VQA loss near zero: labels probably include only `-100` or are shifted incorrectly.
- Visual IDs in text: logit mask or Alpaca guard is missing.
- High forgetting ratio `R`: Alpaca batches are contaminated with visual tokens or replay weight is too low.
- Resize token embedding errors: call `resize_token_embeddings` before `get_peft_model`.
- NaNs: use `torch.amp.autocast`, `GradScaler`, gradient clipping, and log skipped steps.

