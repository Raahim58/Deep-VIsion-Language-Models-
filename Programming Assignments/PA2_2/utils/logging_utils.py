"""Logging utilities."""
from __future__ import annotations
import logging
import json
from pathlib import Path
from datetime import datetime


def get_logger(name: str = "dvlm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class MetricsLogger:
    """Accumulates and persists training metrics to JSONL."""

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a")

    def log(self, step: int, metrics: dict):
        record = {"step": step, "ts": datetime.utcnow().isoformat(), **metrics}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()

    def load(self) -> list[dict]:
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def log_metrics(logger: logging.Logger, step: int, metrics: dict, prefix: str = ""):
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    logger.info(f"[step {step}] {prefix} " + "  ".join(parts))
