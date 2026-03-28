"""Training curve and distribution plots."""
from __future__ import annotations
from pathlib import Path
import json


def _load_jsonl(path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_training_curves(log_path: str | Path, save_path: str | Path | None = None):
    import matplotlib.pyplot as plt
    records = _load_jsonl(log_path)
    if not records:
        print("[Plot] No records found.")
        return

    steps = [r["step"] for r in records]
    keys  = [k for k in records[0] if k not in ("step", "ts") and isinstance(records[0][k], (int, float))]

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        vals = [r.get(key, float("nan")) for r in records]
        ax.plot(steps, vals, linewidth=1.5)
        ax.set_ylabel(key)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    plt.suptitle(str(log_path), fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_reward_distribution(pos_rewards: list, neg_rewards: list, save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pos_rewards, bins=30, alpha=0.6, label="chosen (r+)", color="steelblue")
    ax.hist(neg_rewards, bins=30, alpha=0.6, label="rejected (r−)", color="tomato")
    ax.set_xlabel("Reward score")
    ax.set_ylabel("Count")
    ax.set_title("Reward model score distribution")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_method_comparison(results: dict, metric: str = "win_rate", save_path=None):
    """Bar chart comparing methods on a given metric."""
    import matplotlib.pyplot as plt
    methods = list(results.keys())
    values  = [results[m].get(metric, 0) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel(metric)
    ax.set_title(f"Method comparison — {metric}")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
