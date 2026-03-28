"""Checkpoint I/O helpers."""
from __future__ import annotations
import os
import torch
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(model, path: str | Path, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[IO] Saved checkpoint → {path}")


def load_checkpoint(model, path: str | Path, strict: bool = True):
    payload = torch.load(path, map_location="cpu")
    sd = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(f"[IO] Missing keys: {missing}")
    if unexpected:
        print(f"[IO] Unexpected keys: {unexpected}")
    print(f"[IO] Loaded checkpoint ← {path}")
    return payload


def save_peft_checkpoint(model, path: str | Path):
    """Save only LoRA adapter weights (much smaller)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    print(f"[IO] Saved PEFT adapters → {path}")
