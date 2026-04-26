from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision.datasets import CIFAR10


def stratified_indices(targets, per_class, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    idxs = []
    for c in sorted(np.unique(targets)):
        cls = np.where(targets == c)[0]
        idxs.extend(rng.choice(cls, size=per_class, replace=False).tolist())
    rng.shuffle(idxs)
    return idxs


def load_cifar_subsets(root, train_per_class=1000, test_per_class=200, seed=42):
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)
    train_idx = stratified_indices(train.targets, train_per_class, seed)
    test_idx = stratified_indices(test.targets, test_per_class, seed)
    print("CIFAR train class counts:", Counter([train.targets[i] for i in train_idx]))
    print("CIFAR test class counts:", Counter([test.targets[i] for i in test_idx]))
    return train, test, train_idx, test_idx


def cache_clip_pixels(dataset, indices, processor, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        print("Loaded cached pixels:", cache_path, tuple(obj["pixel_values"].shape))
        return obj["pixel_values"], obj["labels"]
    pixels, labels = [], []
    for idx in tqdm(indices, desc=f"CLIP preprocess {cache_path.name}"):
        image, y = dataset[idx]
        pv = processor(images=image, return_tensors="pt")["pixel_values"][0]
        pixels.append(pv)
        labels.append(y)
    out = {"pixel_values": torch.stack(pixels), "labels": torch.tensor(labels).long()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    print("Saved cached pixels:", cache_path, tuple(out["pixel_values"].shape))
    return out["pixel_values"], out["labels"]


@torch.no_grad()
def cache_clip_patch_tokens(clip_model, pixel_values, device, cache_path, batch_size=64):
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = torch.load(cache_path, map_location="cpu")
        print("Loaded cached CLIP patches:", cache_path, tuple(z.shape))
        return z
    outs = []
    clip_model.eval()
    for i in tqdm(range(0, len(pixel_values), batch_size), desc="CLIP patches"):
        batch = pixel_values[i:i + batch_size].to(device)
        hidden = clip_model.vision_model(pixel_values=batch).last_hidden_state
        assert hidden.shape[1] == 50, f"Expected 50 CLIP tokens before CLS discard, got {hidden.shape}"
        outs.append(hidden[:, 1:, :].float().cpu())
    z = torch.cat(outs)
    assert z.shape[1:] == (49, 768), f"Expected [N,49,768], got {z.shape}"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, cache_path)
    print("Saved cached CLIP patches:", cache_path, tuple(z.shape))
    return z


def save_rows_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_rows_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

