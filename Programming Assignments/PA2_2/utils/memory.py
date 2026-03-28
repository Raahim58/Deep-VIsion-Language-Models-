"""GPU memory helpers."""
import torch


def memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {
        "allocated_GB": round(allocated, 2),
        "reserved_GB":  round(reserved,  2),
        "total_GB":     round(total,     2),
        "free_GB":      round(total - reserved, 2),
    }


def print_memory():
    stats = memory_stats()
    if not stats.get("cuda"):
        print("[Memory] No CUDA device")
        return
    print(
        f"[Memory] allocated={stats['allocated_GB']:.2f}GB  "
        f"reserved={stats['reserved_GB']:.2f}GB  "
        f"total={stats['total_GB']:.2f}GB  "
        f"free={stats['free_GB']:.2f}GB"
    )


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
