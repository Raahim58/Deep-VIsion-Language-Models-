"""YAML config loading and merging."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: dict) -> dict:
    """Later configs override earlier ones (shallow merge)."""
    merged = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


class Config:
    """Attribute-access wrapper around a plain dict."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self):
        return f"Config({vars(self)})"
