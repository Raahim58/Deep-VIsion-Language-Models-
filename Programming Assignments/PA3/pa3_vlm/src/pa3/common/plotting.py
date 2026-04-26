from pathlib import Path
import matplotlib.pyplot as plt


def save_show(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    print("Saved plot:", path)
    plt.close(fig)

