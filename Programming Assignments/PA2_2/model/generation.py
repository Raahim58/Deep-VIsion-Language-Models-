"""
Batch generation utilities.
"""
from __future__ import annotations

import torch
from typing import List


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda",
) -> List[str]:
    """
    Generate one response per prompt string.
    Returns decoded response strings (without the prompt prefix).
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024 - max_new_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out_ids = model.generate(**gen_kwargs)
    response_ids = out_ids[:, prompt_len:]  # strip prompt tokens

    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    return responses


@torch.no_grad()
def generate_k_responses(
    model,
    tokenizer,
    prompt: str,
    K: int,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> List[str]:
    """Generate K responses for a single prompt (used in GRPO rollouts)."""
    return generate_responses(
        model, tokenizer,
        prompts=[prompt] * K,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        device=device,
    )
