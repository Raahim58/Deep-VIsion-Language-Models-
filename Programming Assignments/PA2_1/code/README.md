# Colab-First RLHF / Alignment Assignment Repo

Compact, runnable reference code for the AI623 alignment assignment. The repo is designed for:

- GitHub push/clone with clean structure
- Colab-first execution from a single notebook
- Lightweight but real implementations of SFT, reward modeling, DPO, PPO, GRPO, and RLVR
- Practical defaults for a Colab T4

The default stack is:

- Policy: `HuggingFaceTB/SmolLM2-360M`
- Reference: frozen copy of the SFT policy checkpoint
- Reward model backbone: `meta-llama/Llama-3.2-1B-Instruct`
- Value model backbone: `meta-llama/Llama-3.2-1B-Instruct`
- LoRA: `r=8`, `alpha=16`, `target_modules=[q_proj, v_proj]`, `dropout=0.05`

## What This Repo Covers

- HH-RLHF harmless data loading and prompt/response parsing
- SFT warm-up on response tokens only
- Reward model training on pairwise preferences
- DPO with response-token masking
- PPO with cached old logprobs, per-token KL shaping, GAE, clipped surrogate, and critic loss
- GRPO with grouped rollouts and group-relative advantages
- RLVR on GSM8K with exact-answer verification
- Evaluation, plotting, sample tables, and zip packaging

## Quick Setup

```bash
git clone <your-repo-url>
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Colab, the notebook installs dependencies for you.

## Hugging Face Access

The reward/value backbones use `meta-llama/Llama-3.2-1B-Instruct`. You need:

1. A valid `HF_TOKEN`
2. Accepted access terms for the Llama model on Hugging Face

If you do not have access, the repo fails with a clear model-loading error. For local debugging you can temporarily replace those model names in `configs/default.yaml` with an open checkpoint.

## Colab Flow

1. Open [notebooks/run_colab.ipynb](/Users/Raahim/Documents/LUMS/Junior/spring_semester/coursework/AI623/Deep VIsion Language Models /Programming Assignments/PA2/code/notebooks/run_colab.ipynb)
2. Set the repo path or GitHub URL
3. Optionally set `HF_TOKEN`
4. Toggle `RUN_SFT`, `RUN_RM`, `RUN_DPO`, `RUN_PPO`, `RUN_GRPO`, `RUN_RLVR`, `RUN_EVAL`
5. Run the notebook top to bottom

Outputs are written under `runs/`.

## Repo Layout

```text
code/
  configs/
  data/
  model/
  alignment/
  utils/
  notebooks/
  tools/
  train_sft.py
  train_rm.py
  train_rl.py
  eval.py
```

## Commands

Train SFT:

```bash
python train_sft.py --config configs/default.yaml --config configs/sft.yaml
```

Train reward model:

```bash
python train_rm.py --config configs/default.yaml --config configs/rm.yaml
```

Train DPO:

```bash
python train_rl.py --method dpo --config configs/default.yaml --config configs/dpo.yaml
```

Train PPO:

```bash
python train_rl.py --method ppo --config configs/default.yaml --config configs/ppo.yaml
```

Train GRPO:

```bash
python train_rl.py --method grpo --config configs/default.yaml --config configs/grpo.yaml
```

Train RLVR:

```bash
python train_rl.py --method rlvr --config configs/default.yaml --config configs/rlvr.yaml
```

Run evaluation:

```bash
python eval.py --config configs/default.yaml --policy-checkpoint runs/<method_run_dir> --reference-checkpoint runs/<sft_run_dir>
```

Package source files:

```bash
python tools/make_scripts_zip.py
```

## Practical Notes

- Policy tokenizer uses left padding with `pad_token = eos_token`.
- Reward tokenizer uses right padding.
- Frozen models can load in 8-bit when bitsandbytes is available.
- The code avoids `Trainer`, TRL, or other high-level alignment libraries.
- PPO/GRPO defaults are intentionally conservative for Colab.

## Troubleshooting

- Missing `HF_TOKEN`: set `os.environ["HF_TOKEN"]` in Colab or run `huggingface-cli login`
- Llama access denied: accept the model license or swap to an open reward/value backbone
- CUDA OOM: reduce batch size, prompts per step, or `max_new_tokens`; keep frozen models in 8-bit
- DPO stuck near 50%: verify response masking and that policy/ref both start from the same SFT checkpoint
- PPO ratio not 1.0 at rollout start: verify old logprobs are cached before update steps
- GRPO zero advantages: inspect reward dispersion inside each group; exact ties produce zero-variance groups
- Broken generation: verify policy tokenizer uses left padding and `pad_token = eos_token`
