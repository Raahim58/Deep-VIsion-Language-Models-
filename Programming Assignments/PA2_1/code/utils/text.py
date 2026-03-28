from __future__ import annotations

import re


ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_special_tokens(text: str) -> str:
    return text.replace("<|endoftext|>", "").strip()


def extract_gsm8k_gold_answer(solution: str) -> str:
    match = ANSWER_RE.search(solution)
    if match:
        return normalize_numeric_string(match.group(1))
    numbers = NUMBER_RE.findall(solution)
    return normalize_numeric_string(numbers[-1]) if numbers else ""


def extract_gsm8k_pred_answer(text: str) -> str:
    answer_patterns = [
        r"####\s*([^\n]+)",
        r"answer\s*[:=]\s*([^\n]+)",
        r"final answer\s*[:=]\s*([^\n]+)",
    ]
    lower = text.lower()
    for pattern in answer_patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            return normalize_numeric_string(match.group(1))
    numbers = NUMBER_RE.findall(text)
    return normalize_numeric_string(numbers[-1]) if numbers else ""


def normalize_numeric_string(text: str) -> str:
    cleaned = text.strip().replace(",", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned
