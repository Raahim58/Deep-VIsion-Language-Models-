"""
Reward model: Llama-3.2-1B backbone + scalar regression head.
Uses AutoModelForSequenceClassification (num_labels=1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from model.loading import _resolve_device_map


class RewardModel(nn.Module):
    """
    Thin wrapper around AutoModelForSequenceClassification with num_labels=1.
    Reads the hidden state at the last non-pad token and projects to a scalar.
    """

    def __init__(self, backbone_name: str, use_8bit: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        device_map = _resolve_device_map()
        if use_8bit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                backbone_name,
                num_labels=1,
                quantization_config=bnb_cfg,
                # device_map="auto",
                device_map=device_map,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                backbone_name,
                num_labels=1,
                torch_dtype=dtype,
                # device_map="auto",
                device_map=device_map,
            )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.squeeze(-1)

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @classmethod
    def load_frozen(cls, backbone_name: str, use_8bit: bool = True) -> "RewardModel":
        rm = cls(backbone_name, use_8bit=use_8bit)
        rm.freeze()
        return rm


def score_texts(rm, rm_tokenizer, texts: list, max_len: int = 1024) -> list:
    """Score a list of full (prompt+response) strings with the reward model."""
    enc = rm_tokenizer(
        texts, truncation=True, max_length=max_len,
        padding="longest", return_tensors="pt",
    )
    rm_device = next(rm.parameters()).device
    with torch.no_grad():
        scores = rm.score(enc["input_ids"].to(rm_device), enc["attention_mask"].to(rm_device))
    return scores.float().cpu().tolist()
