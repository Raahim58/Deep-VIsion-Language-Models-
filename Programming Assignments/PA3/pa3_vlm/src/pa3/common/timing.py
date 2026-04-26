import time
from contextlib import contextmanager
from .vram import reset_peak_vram, print_vram


@contextmanager
def phase_timer(name: str):
    reset_peak_vram()
    t0 = time.time()
    print(f"[{name}] start")
    try:
        yield
    finally:
        elapsed = (time.time() - t0) / 60
        print_vram(f"[{name}]")
        print(f"[{name}] elapsed_minutes={elapsed:.3f}")


class StepTimer:
    def __init__(self):
        self.last = time.time()

    def tick(self, examples=0):
        now = time.time()
        dt = now - self.last
        self.last = now
        eps = examples / dt if dt > 0 and examples else 0.0
        return dt, eps

