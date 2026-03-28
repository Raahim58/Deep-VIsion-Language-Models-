"""
GSM8K dataset loading for RLVR.
"""
from __future__ import annotations

from typing import List, Optional
from torch.utils.data import Dataset
from datasets import load_dataset

from .parsing import extract_gsm8k_answer

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    "At the end, write your final answer as a single number.\n\n"
    "Problem: {question}\n\nSolution:"
)


def load_gsm8k(split: str = "train", max_samples: Optional[int] = None) -> List[dict]:
    """
    Load openai/gsm8k and format each example into:
      {prompt, question, gold_answer (float)}
    """
    raw = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for ex in raw:
        gold = extract_gsm8k_answer(ex["answer"])
        if gold is None:
            continue
        items.append({
            "question":    ex["question"],
            "gold_answer": gold,
            "gold_text":   ex["answer"],
            "prompt":      PROMPT_TEMPLATE.format(question=ex["question"]),
        })
        if max_samples and len(items) >= max_samples:
            break
    print(f"[GSM8K] {split}: loaded {len(items)} examples")
    return items


class GSM8KDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer, max_prompt_tokens: int = 200):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.data = []
        for d in data:
            # Truncate the prompt if necessary
            ids = tokenizer.encode(d["prompt"], add_special_tokens=True)
            if len(ids) > max_prompt_tokens:
                ids = ids[:max_prompt_tokens]
                d = dict(d)
                d["prompt"] = tokenizer.decode(ids, skip_special_tokens=False)
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
