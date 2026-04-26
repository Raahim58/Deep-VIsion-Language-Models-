from pathlib import Path
import torch


def save_checkpoint(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print("Saved checkpoint:", path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    print("Loaded checkpoint:", path)
    return ckpt

