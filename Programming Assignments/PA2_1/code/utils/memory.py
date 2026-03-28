from __future__ import annotations

from contextlib import nullcontext
from typing import Iterator

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torch_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_context(dtype: torch.dtype):
    if torch.cuda.is_available() and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def count_parameters(model) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def format_parameter_count(model) -> str:
    counts = count_parameters(model)
    return f"total={counts['total']:,} trainable={counts['trainable']:,}"


def get_gpu_report() -> dict[str, float | str]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "total_gb": 0.0, "reserved_gb": 0.0, "allocated_gb": 0.0}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / 1024**3
    reserved_gb = torch.cuda.memory_reserved(index) / 1024**3
    allocated_gb = torch.cuda.memory_allocated(index) / 1024**3
    return {
        "device": props.name,
        "total_gb": round(total_gb, 2),
        "reserved_gb": round(reserved_gb, 2),
        "allocated_gb": round(allocated_gb, 2),
    }
