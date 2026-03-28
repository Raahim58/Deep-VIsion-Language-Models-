from __future__ import annotations

from typing import Any


ASSISTANT_MARKER = "\n\nAssistant:"


def split_hh_prompt_response(text: str) -> tuple[str, str]:
    marker_index = text.rfind(ASSISTANT_MARKER)
    if marker_index == -1:
        marker_index = text.rfind("Assistant:")
        if marker_index == -1:
            raise ValueError("Could not find assistant marker in HH-RLHF sample.")
        prompt = text[: marker_index + len("Assistant:")]
        response = text[marker_index + len("Assistant:") :]
        return prompt, response.strip()

    prompt_end = marker_index + len(ASSISTANT_MARKER)
    prompt = text[:prompt_end]
    response = text[prompt_end:]
    return prompt, response.strip()


def parse_hh_example(example: dict[str, Any]) -> dict[str, str]:
    chosen_prompt, chosen_response = split_hh_prompt_response(example["chosen"])
    rejected_prompt, rejected_response = split_hh_prompt_response(example["rejected"])
    if chosen_prompt != rejected_prompt:
        raise ValueError("Chosen/rejected prompts do not match after parsing.")
    return {
        "prompt": chosen_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "chosen_full": example["chosen"],
        "rejected_full": example["rejected"],
    }
