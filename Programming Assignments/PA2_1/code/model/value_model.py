from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class ValueModel(nn.Module):
    def __init__(self, backbone_name: str, torch_dtype: torch.dtype, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.backbone.config.use_cache = False
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch_dtype)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.to(self.value_head.weight.dtype)
        return self.value_head(hidden).squeeze(-1)
