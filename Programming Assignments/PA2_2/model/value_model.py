"""
Value model: Llama-3.2-1B backbone + scalar head for PPO critic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, BitsAndBytesConfig


class ValueModel(nn.Module):
    """
    Critic for PPO. Wraps the Llama backbone with a scalar head V(s).
    The backbone can be frozen (train head only) or LoRA-adapted.
    """

    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool = True,
        use_8bit: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if use_8bit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.backbone = AutoModel.from_pretrained(
                backbone_name, quantization_config=bnb_cfg, device_map="auto",
            )
        else:
            self.backbone = AutoModel.from_pretrained(
                backbone_name, torch_dtype=dtype, device_map="auto",
            )

        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.01)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Move value head to same device as backbone
        backbone_device = next(self.backbone.parameters()).device
        self.value_head = self.value_head.to(backbone_device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns per-token value estimates.
        Shape: (batch_size, seq_len)
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)
        values = self.value_head(hidden).squeeze(-1)  # (B, T)
        return values

    def value_at_last(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns value estimate at the last real (non-pad) token. Shape: (B,)"""
        values = self.forward(input_ids, attention_mask)
        # Find last non-pad position per sequence
        lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(values.size(0), device=values.device)
        return values[batch_idx, lengths]
