"""
Model and tokenizer loading utilities.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def _resolve_device_map():
    if torch.cuda.is_available():
        return "cuda"   # explicit, not "auto"
    return "cpu"

def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from {list(mapping)}")
    return mapping[dtype_str]


def get_tokenizer(model_name: str, padding_side: str = "left") -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_policy(
    model_name: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """Load the trainable policy model (SmolLM2-360M-Instruct)."""
    torch_dtype = get_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.config.use_cache = False  # Required for gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # Needed for PEFT + grad checkpointing
    print(f"[Policy] Loaded {model_name}: {_param_count(model):.1f}M params")
    return model


def load_reward_backbone(
    model_name: str,
    dtype: str = "bfloat16",
    use_8bit: bool = False,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    Load the backbone for reward model / value model.
    Uses 8-bit quantisation when use_8bit=True to save VRAM on frozen copies.
    """
    if use_8bit:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
        )
    else:
        torch_dtype = get_dtype(dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    print(f"[Backbone] Loaded {model_name}: {_param_count(model):.1f}M params  (8bit={use_8bit})")
    return model


def _param_count(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
