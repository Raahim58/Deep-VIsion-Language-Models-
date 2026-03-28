# DVLM PA2 — LLM Alignment

Full implementation of SFT, Reward Model, DPO, PPO, GRPO, and RLVR for AI623 Programming Assignment 2.

## Model Configuration (T4-Safe Defaults)

| Component | Model | Notes |
|---|---|---|
| Policy (trainable) | `HuggingFaceTB/SmolLM2-360M-Instruct` | LoRA r=8 |
| Reference πref (frozen) | Same SmolLM2 checkpoint | 8-bit optional |
| Reward model | `meta-llama/Llama-3.2-1B-Instruct` | Sequence classifier head |
| Value model (PPO) | `meta-llama/Llama-3.2-1B-Instruct` | Frozen backbone + scalar head |

## Quickstart — Colab

```python
# Cell 1: Clone
!git clone https://github.com/YOUR_USERNAME/dvlm_pa2.git
%cd dvlm_pa2

# Cell 2: Install
!pip install -r requirements.txt -q

# Cell 3: HF login (needed for Llama-3.2)
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")   # or set HF_TOKEN env var

# Cell 4: Open and run notebooks/run_colab.ipynb top to bottom
```

## Quickstart — Local

```bash
git clone https://github.com/YOUR_USERNAME/dvlm_pa2.git
cd dvlm_pa2
pip install -e .
pip install -r requirements.txt

# Run SFT
python train_sft.py --max_samples 5000

# Run RM
python train_rm.py --max_samples 5000

# Run DPO (after SFT)
python train_rl.py --method dpo --sft_ckpt runs/sft/sft_adapter

# Run GRPO (after SFT + RM)
python train_rl.py --method grpo --sft_ckpt runs/sft/sft_adapter --rm_ckpt runs/rm/rm.pt

# Run PPO (after SFT + RM)
python train_rl.py --method ppo --sft_ckpt runs/sft/sft_adapter --rm_ckpt runs/rm/rm.pt

# Run RLVR (after SFT, no RM needed)
python train_rl.py --method rlvr --sft_ckpt runs/sft/sft_adapter

# Evaluate all methods
python eval.py --methods dpo grpo ppo --sft_ckpt runs/sft/sft_adapter

# Package source
python tools/make_scripts_zip.py
```

## File Structure

```
dvlm_pa2/
  configs/          YAML config files (one per method + shared default)
  data/             HH-RLHF and GSM8K loading, parsing, collators
  model/            Policy/RM/value loading, LoRA, generation, log-probs
  alignment/        DPO, PPO, GRPO, RLVR, KL, GAE implementations
  train_sft.py      SFT warm-up training loop
  train_rm.py       Reward model training loop
  train_rl.py       DPO / PPO / GRPO / RLVR training dispatcher
  eval.py           Win-rate, KL, sample table evaluation
  utils/            Config, logging, IO, plotting, metrics helpers
  notebooks/        run_colab.ipynb — one-button Colab runner
  tools/            make_scripts_zip.py
  runs/             Output checkpoints and logs (gitignored)
```

## Recommended Training Order

1. **C0** — Data + model loading (sanity checks in notebook)
2. **C1** — `train_rm.py` — reward model (target ≥60% pref accuracy)
3. **C2** — `train_sft.py` — SFT warm-up (produces πref and π⁰_θ)
4. **C4** — `train_rl.py --method dpo` — easiest, offline, no RM needed during training
5. **C5** — `train_rl.py --method grpo` — online, no critic
6. **C3** — `train_rl.py --method ppo` — most memory-intensive
7. **C6** — `train_rl.py --method rlvr` — GSM8K with binary reward

## VRAM Budget (bfloat16, T4 = 15 GB)

| Method | Approx VRAM |
|---|---|
| SFT | ~4 GB |
| RM training | ~5 GB |
| DPO | ~6 GB |
| GRPO | ~9 GB |
| PPO | ~11 GB |

**Tips to reduce VRAM:**
- Set `use_8bit_frozen: true` in `configs/default.yaml`
- Reduce `max_new_tokens` to 64 for ablations
- Reduce `prompts_per_step` to 4
- Enable `torch.backends.cuda.matmul.allow_tf32 = True`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HF_TOKEN` missing / 401 | `huggingface_hub.login()` or `export HF_TOKEN=...` |
| Llama access denied | Accept license at huggingface.co/meta-llama/Llama-3.2-1B-Instruct |
| CUDA OOM | Reduce `prompts_per_step`, `max_new_tokens`; enable 8-bit frozen models |
| DPO stuck at 50% pref acc | Check ref model is frozen; check prompt token masking in log-prob sum |
| PPO ratio ≠ 1 at epoch start | You are recomputing old log-probs — must cache at rollout time |
| GRPO all advantages zero | All K completions got same reward — lower temperature or increase K |
| SFT perplexity suspiciously low (< 5) | Loss leaking into prompt tokens — check `labels[prompt]=-100` masking |
| Generation produces only EOS | `padding_side` is `right` instead of `left` for decoder-only model |

## Constraints (from assignment)

- No TRL, trlX, OpenRLHF, RL4LMs, or any ready-made alignment trainer
- No `Trainer` API for RM or RL training
- All alignment logic (GAE, clipping, DPO loss, KL shaping) is implemented from scratch
- `AutoModelForSequenceClassification` used for RM backbone (as required)
