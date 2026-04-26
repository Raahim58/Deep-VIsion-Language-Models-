from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from pa3.common.plotting import save_show
from .part_b_templates import SYN_CLASSES


def draw_synthetic(cls, size=16, rng=None):
    rng = rng or np.random.default_rng()
    img = np.zeros((size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = (size - 1) / 2 + rng.uniform(-0.6, 0.6), (size - 1) / 2 + rng.uniform(-0.6, 0.6)
    color = rng.uniform(0.45, 1.0, size=3)
    img[:] = rng.uniform(0.0, 0.15, size=3)
    if cls == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rng.uniform(4.0, 6.0) ** 2
        img[mask] = color
    elif cls == "triangle":
        top, left, right = np.array([cy - 5, cx]), np.array([cy + 5, cx - 6]), np.array([cy + 5, cx + 6])
        def edge(p1, p2):
            return (xx - p1[1]) * (p2[0] - p1[0]) - (yy - p1[0]) * (p2[1] - p1[1])
        e1, e2, e3 = edge(top, left), edge(left, right), edge(right, top)
        img[((e1 >= 0) & (e2 >= 0) & (e3 >= 0)) | ((e1 <= 0) & (e2 <= 0) & (e3 <= 0))] = color
    elif cls == "cross":
        w = rng.integers(2, 4)
        img[(np.abs(xx - cx) <= w) | (np.abs(yy - cy) <= w)] = color
    elif cls == "checkerboard":
        block = rng.choice([2, 4])
        mask = ((xx // block + yy // block) % 2) == 0
        img[mask] = color
        img[~mask] = 1 - color * 0.5
    elif cls == "gradient":
        grad = (xx + yy) / (2 * (size - 1))
        img = np.stack([grad, np.flipud(grad), 0.5 * grad + 0.2], axis=-1).astype(np.float32)
    elif cls == "spiral":
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        img[(np.mod(theta + r * 1.4, 2 * np.pi) < 0.45) & (r < 7.2)] = color
    return np.clip(img + rng.normal(0, 0.025, img.shape).astype(np.float32), 0, 1)


def generate_dataset(n_per_class=1000, seed=42):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cid, cls in enumerate(SYN_CLASSES):
        for _ in range(n_per_class):
            images.append(draw_synthetic(cls, rng=rng))
            labels.append(cid)
    images = np.stack(images)
    labels = np.asarray(labels)
    order = rng.permutation(len(labels))
    return images[order], labels[order]


def stratified_split(images, labels, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(images, labels))
    print("Synthetic train counts:", Counter(labels[train_idx].tolist()))
    print("Synthetic val counts:", Counter(labels[val_idx].tolist()))
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def save_grid(images, labels, path):
    fig, axes = plt.subplots(6, 5, figsize=(8, 9))
    for cid, cls in enumerate(SYN_CLASSES):
        idxs = np.where(labels == cid)[0][:5]
        for j, idx in enumerate(idxs):
            axes[cid, j].imshow(images[idx])
            axes[cid, j].axis("off")
            if j == 0:
                axes[cid, j].set_title(cls)
    save_show(fig, path)


class ImageTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.x = torch.tensor(images).permute(0, 3, 1, 2).float()
        self.y = torch.tensor(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

