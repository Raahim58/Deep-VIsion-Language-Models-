from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics_frame(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_metric_curves(metrics_path: str | Path, out_path: str | Path, x_key: str = "step") -> Path:
    df = load_metrics_frame(metrics_path)
    numeric_cols = [col for col in df.columns if col != x_key and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise ValueError(f"No numeric metrics found in {metrics_path}")

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8, max(3, 3 * len(numeric_cols))), squeeze=False)
    for axis, col in zip(axes.flatten(), numeric_cols):
        axis.plot(df[x_key], df[col])
        axis.set_title(col)
        axis.set_xlabel(x_key)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out
