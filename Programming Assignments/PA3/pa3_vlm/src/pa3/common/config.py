import argparse
from pathlib import Path
try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def parse_args(default_config):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick_debug", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ablation", default=None)
    p.add_argument("--full_eval", action="store_true")
    p.add_argument("--unfreeze_last4_clip", action="store_true")
    return p.parse_args()


def load_config(path, output_dir=None):
    with Path(path).open("r", encoding="utf-8") as f:
        if yaml is not None:
            cfg = yaml.safe_load(f)
        else:
            cfg = _simple_yaml(f.read())
    if output_dir:
        cfg["output_dir"] = output_dir
    return cfg


def _coerce(value):
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml(text):
    root = {}
    current = root
    stack = [(0, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            node = {}
            current[key] = node
            stack.append((indent + 2, node))
        else:
            current[key] = _coerce(value)
    return root


def apply_quick_debug_a(cfg):
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["train"] = dict(cfg["train"])
    cfg["data"]["train_per_class"] = 10
    cfg["data"]["test_per_class"] = 2
    cfg["data"]["alpaca_n"] = 20
    cfg["train"]["batch_a1"] = 8
    cfg["train"]["batch_a2"] = 4
    return cfg


def apply_quick_debug_b(cfg):
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["train"] = dict(cfg["train"])
    cfg["vqvae"] = dict(cfg["vqvae"])
    cfg["data"]["n_per_class"] = 10
    cfg["data"]["alpaca_n"] = 20
    cfg["vqvae"]["epochs"] = 1
    cfg["vqvae"]["batch_size"] = 12
    cfg["train"]["batch_lm"] = 4
    cfg["train"]["epochs"] = 1
    return cfg
