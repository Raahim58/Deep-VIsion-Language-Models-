from __future__ import annotations

from typing import Any


ASSISTANT_MARKER = "\n\nAssistant:"
ASSISTANT_FALLBACK = "Assistant:"


def split_hh_prompt_response(text: str) -> tuple[str, str]:
    marker_index = text.rfind(ASSISTANT_MARKER)
    if marker_index == -1:
        marker_index = text.rfind(ASSISTANT_FALLBACK)
        if marker_index == -1:
            raise ValueError("Could not find assistant marker in HH-RLHF sample.")
        prompt = text[: marker_index + len(ASSISTANT_FALLBACK)]
        response = text[marker_index + len(ASSISTANT_FALLBACK) :]
        return prompt, response.strip()

    prompt_end = marker_index + len(ASSISTANT_MARKER)
    prompt = text[:prompt_end]
    response = text[prompt_end:]
    return prompt, response.strip()


def _truncate_to_last_assistant(text: str) -> str:
    trimmed = text.rstrip()
    marker_index = trimmed.rfind(ASSISTANT_FALLBACK)
    if marker_index == -1:
        return ""
    return trimmed[: marker_index + len(ASSISTANT_FALLBACK)]


def _shared_prompt_prefix(chosen_prompt: str, rejected_prompt: str) -> str:
    common_len = 0
    for chosen_char, rejected_char in zip(chosen_prompt, rejected_prompt):
        if chosen_char != rejected_char:
            break
        common_len += 1

    shared = chosen_prompt[:common_len].rstrip()
    if shared.endswith(ASSISTANT_FALLBACK):
        return shared

    shared = _truncate_to_last_assistant(shared)
    if shared:
        return shared

    shorter_prompt = min((chosen_prompt, rejected_prompt), key=len)
    shared = _truncate_to_last_assistant(shorter_prompt)
    if shared:
        return shared

    raise ValueError(
        "Could not derive a shared prompt prefix for HH-RLHF sample; "
        "no valid assistant boundary remained after fallback."
    )


def parse_hh_example(example: dict[str, Any]) -> dict[str, str]:
    chosen_prompt, chosen_response = split_hh_prompt_response(example["chosen"])
    rejected_prompt, rejected_response = split_hh_prompt_response(example["rejected"])
    prompt = chosen_prompt if chosen_prompt == rejected_prompt else _shared_prompt_prefix(chosen_prompt, rejected_prompt)
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "chosen_full": example["chosen"],
        "rejected_full": example["rejected"],
    }
