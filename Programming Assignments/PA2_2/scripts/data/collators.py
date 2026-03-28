"""
DataLoader collators for SFT, RM, and DPO batches.
"""
from __future__ import annotations

import torch
from typing import List


class SFTCollator:
    """
    Tokenises (prompt + chosen) and builds labels with prompt tokens masked.
    Uses LEFT padding (required for decoder-only generation compatibility).
    """

    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[dict]):
        full_texts  = [b["full_text"]  for b in batch]
        prompt_lens = [b["prompt_len"] for b in batch]

        enc = self.tok(
            full_texts,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        labels = enc["input_ids"].clone()
        # Mask padding tokens
        labels[labels == self.tok.pad_token_id] = -100

        # Mask prompt tokens for each item individually
        for i, plen in enumerate(prompt_lens):
            # After left-padding, real tokens start at some offset.
            # Find first non-pad token.
            mask = enc["attention_mask"][i]
            start = (mask == 1).nonzero(as_tuple=False)[0].item()
            # Mask prompt portion (from start to start+plen, clamped to seq len)
            end = min(start + plen, self.max_len)
            labels[i, start:end] = -100

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
        }


class RMCollator:
    """
    Returns tokenised (prompt+chosen) and (prompt+rejected) pairs.
    Uses RIGHT padding — we read the last real token for the scalar head.
    """

    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[dict]):
        chosen_texts   = [b["prompt"] + " " + b["chosen"]   for b in batch]
        rejected_texts = [b["prompt"] + " " + b["rejected"] for b in batch]

        enc_pos = self.tok(
            chosen_texts, truncation=True, max_length=self.max_len,
            padding="longest", return_tensors="pt",
        )
        enc_neg = self.tok(
            rejected_texts, truncation=True, max_length=self.max_len,
            padding="longest", return_tensors="pt",
        )

        return {
            "input_ids_pos":      enc_pos["input_ids"],
            "attention_mask_pos": enc_pos["attention_mask"],
            "input_ids_neg":      enc_neg["input_ids"],
            "attention_mask_neg": enc_neg["attention_mask"],
        }


class DPOCollator:
    """
    Tokenises (prompt+chosen) and (prompt+rejected) separately.
    Stores prompt lengths so the training loop can build response masks.
    Uses LEFT padding (policy tokeniser).
    """

    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[dict]):
        prompts   = [b["prompt"]   for b in batch]
        chosens   = [b["chosen"]   for b in batch]
        rejecteds = [b["rejected"] for b in batch]

        chosen_texts   = [p + " " + c for p, c in zip(prompts, chosens)]
        rejected_texts = [p + " " + r for p, r in zip(prompts, rejecteds)]

        # Encode prompts to find response-start positions
        prompt_ids = [
            self.tok.encode(p, add_special_tokens=False) for p in prompts
        ]
        prompt_lens = [len(p) for p in prompt_ids]

        enc_pos = self.tok(
            chosen_texts, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        enc_neg = self.tok(
            rejected_texts, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )

        return {
            "input_ids_pos":      enc_pos["input_ids"],
            "attention_mask_pos": enc_pos["attention_mask"],
            "input_ids_neg":      enc_neg["input_ids"],
            "attention_mask_neg": enc_neg["attention_mask"],
            "prompt_lens":        torch.tensor(prompt_lens, dtype=torch.long),
        }
