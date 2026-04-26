import math
import numpy as np


def normalize_answer(text):
    text = str(text).strip().lower()
    for ch in [".", ",", ";", ":", "!", "?"]:
        text = text.replace(ch, "")
    return text.split()[0] if text.split() else ""


def exact_match(preds, refs):
    if not refs:
        return 0.0
    return float(np.mean([normalize_answer(p) == normalize_answer(r) for p, r in zip(preds, refs)]))


def perplexity(loss):
    return float(math.exp(min(float(loss), 20.0)))

