"""
Parsing utilities for HH-RLHF and GSM8K.
"""
import re
from typing import Optional


# ── HH-RLHF ─────────────────────────────────────────────────────────────────

def parse_hh_example(example: dict) -> dict:
    """
    Parse one HH-RLHF example into (prompt, chosen, rejected).

    HH-RLHF stores each field as a single string with alternating
    'Human: ...' / 'Assistant: ...' turns. The final 'Assistant:' block
    is the response; everything before it is the prompt.

    Returns a dict with keys: prompt, chosen, rejected.
    Returns None if parsing fails.
    """
    chosen_raw: str = example.get("chosen", "") or ""
    rejected_raw: str = example.get("rejected", "") or ""

    prompt, chosen = _split_last_assistant(chosen_raw)
    _,      rejected = _split_last_assistant(rejected_raw)

    if not prompt or not chosen or not rejected:
        return None

    return {"prompt": prompt.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()}


def _split_last_assistant(text: str):
    """
    Split text at the *last* occurrence of 'Assistant:'.
    Returns (context_before, response_after).
    """
    marker = "Assistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    prompt = text[:idx + len(marker)].strip()
    response = text[idx + len(marker):].strip()
    return prompt, response


# ── GSM8K ────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
)

def extract_gsm8k_answer(text: str) -> Optional[float]:
    """
    Extract the final numeric answer from a model completion or ground-truth
    GSM8K solution.

    Priority order:
      1. After '####'                    (ground-truth format)
      2. After 'The answer is'           (common model format)
      3. After 'boxed{' / '\\boxed{'     (LaTeX format)
      4. Last number in the string       (fallback)

    Returns a float, or None if no valid number is found.
    """
    text = text.strip()

    # 1. #### {number}
    m = re.search(r"####\s*([-+]?\d[\d,\.]*)", text)
    if m:
        return _parse_number(m.group(1))

    # 2. "The answer is …"
    m = re.search(r"[Tt]he\s+answer\s+is\s*[:=]?\s*([-+]?\d[\d,\.]*)", text)
    if m:
        return _parse_number(m.group(1))

    # 3. \boxed{…}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return _parse_number(m.group(1))

    # 4. Last number in text
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def _parse_number(s: str) -> Optional[float]:
    """Strip commas and convert to float."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def answers_match(pred: Optional[float], gold: Optional[float], tol: float = 1e-6) -> bool:
    """Check whether two extracted answers are equal within tolerance."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < tol
