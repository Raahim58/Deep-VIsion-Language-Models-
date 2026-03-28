from __future__ import annotations

from typing import Iterable

import numpy as np

from utils.text import extract_gsm8k_gold_answer, extract_gsm8k_pred_answer


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def gsm8k_exact_match(prediction: str, gold_solution: str) -> bool:
    return extract_gsm8k_pred_answer(prediction) == extract_gsm8k_gold_answer(gold_solution)


def pairwise_accuracy(chosen_scores, rejected_scores) -> float:
    chosen = np.asarray(chosen_scores)
    rejected = np.asarray(rejected_scores)
    return float((chosen > rejected).mean())
