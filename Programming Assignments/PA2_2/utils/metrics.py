"""Evaluation metric helpers."""
from __future__ import annotations
from typing import List


def preference_accuracy(scores_pos: List[float], scores_neg: List[float]) -> float:
    """Fraction of pairs where r+ > r-."""
    assert len(scores_pos) == len(scores_neg)
    correct = sum(p > n for p, n in zip(scores_pos, scores_neg))
    return correct / len(scores_pos)


def win_rate(
    scores_aligned: List[float],
    scores_baseline: List[float],
) -> float:
    """Fraction of prompts where aligned model scores higher than baseline."""
    assert len(scores_aligned) == len(scores_baseline)
    wins = sum(a > b for a, b in zip(scores_aligned, scores_baseline))
    return wins / len(scores_aligned)
