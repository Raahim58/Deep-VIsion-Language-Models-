from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[str]:
    device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    output = model.generate(
        **generation_kwargs,
    )
    prompt_length = encoded["input_ids"].shape[1]
    completions = output[:, prompt_length:]
    return tokenizer.batch_decode(completions, skip_special_tokens=True)
