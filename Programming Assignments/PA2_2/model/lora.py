"""
LoRA application and freezing utilities.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, PeftModel


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> PeftModel:
    """
    Wrap a model with LoRA adapters.
    Automatically falls back to a minimal set of modules if the specified ones
    are not present in the model.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Check which target modules actually exist
    existing_modules = {name.split(".")[-1] for name, _ in model.named_modules()}
    valid_targets = [m for m in target_modules if m in existing_modules]
    if not valid_targets:
        # Fallback: find any linear layers
        valid_targets = _find_linear_names(model)[:2]
        print(f"[LoRA] Specified modules not found; using fallback: {valid_targets}")
    else:
        print(f"[LoRA] Applying to: {valid_targets}")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=valid_targets,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def freeze_model(model) -> None:
    """Freeze all parameters in-place. Does not return a copy."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def get_frozen_ref(peft_model) -> object:
    """
    Return a context manager that disables adapters, giving access to πref.
    Usage:
        with peft_model.disable_adapter():
            logits = peft_model(input_ids)
    """
    # We just return the model itself; callers use .disable_adapter() context manager.
    return peft_model


def _find_linear_names(model) -> list[str]:
    """Find names of all Linear layers in the model."""
    import torch.nn as nn
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = name.split(".")[-1]
            if short not in names:
                names.append(short)
    return names
