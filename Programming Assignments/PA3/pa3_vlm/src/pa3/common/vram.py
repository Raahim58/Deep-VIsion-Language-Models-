import torch


def vram_stats():
    if not torch.cuda.is_available():
        return {"vram_allocated_gb": 0.0, "vram_reserved_gb": 0.0, "vram_peak_gb": 0.0}
    return {
        "vram_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "vram_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def print_vram(prefix="VRAM"):
    stats = vram_stats()
    print(prefix, {k: round(v, 4) for k, v in stats.items()})
    return stats

