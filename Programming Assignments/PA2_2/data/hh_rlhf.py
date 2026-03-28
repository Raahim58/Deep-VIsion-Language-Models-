"""
HH-RLHF dataset loading and torch Dataset wrappers.
"""
from __future__ import annotations

from typing import List, Optional
from torch.utils.data import Dataset
from datasets import load_dataset

from .parsing import parse_hh_example


def load_hh_rlhf(split: str = "train", max_samples: Optional[int] = None) -> List[dict]:
    """
    Load the harmless-base split of Anthropic/hh-rlhf and parse each example
    into {prompt, chosen, rejected}.

    Args:
        split:       'train' or 'test'
        max_samples: cap for quick debugging

    Returns:
        List of dicts with keys: prompt, chosen, rejected
    """
    raw = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    parsed = []
    for ex in raw:
        result = parse_hh_example(ex)
        if result is not None:
            parsed.append(result)
        if max_samples and len(parsed) >= max_samples:
            break
    print(f"[HH-RLHF] {split}: loaded {len(parsed)} examples")
    return parsed


class HHRLHFDataset(Dataset):
    """Base dataset holding parsed (prompt, chosen, rejected) triples."""

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SFTDataset(Dataset):
    """
    Each item: (prompt, chosen_response).
    Labels mask prompt tokens with -100 so loss is response-only.
    """

    def __init__(self, data: List[dict], tokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []

        for d in data:
            full_text = d["prompt"] + " " + d["chosen"]
            prompt_ids = tokenizer.encode(d["prompt"], add_special_tokens=False)
            self.items.append({
                "full_text": full_text,
                "prompt_len": len(prompt_ids),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RMDataset(Dataset):
    """Each item: (prompt, chosen, rejected) — raw strings for RM tokenization."""

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "prompt":   d["prompt"],
            "chosen":   d["chosen"],
            "rejected": d["rejected"],
        }


class DPODataset(Dataset):
    """Each item: full (prompt, chosen, rejected) triple — raw strings."""

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PromptDataset(Dataset):
    """
    Prompt-only dataset for PPO/GRPO rollouts.
    Returns the raw prompt string.
    """

    def __init__(self, data: List[dict]):
        self.prompts = [d["prompt"] for d in data]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
