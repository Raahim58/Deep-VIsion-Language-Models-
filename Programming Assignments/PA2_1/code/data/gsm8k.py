from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from utils.text import extract_gsm8k_gold_answer


GSM8K_SYSTEM_PROMPT = (
    "You are a careful math solver. Show your reasoning briefly and end with a line in the form "
    "'#### <final answer>'."
)


def _limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def format_gsm8k_prompt(question: str) -> str:
    return (
        f"{GSM8K_SYSTEM_PROMPT}\n\n"
        f"Question: {question.strip()}\n\n"
        "Assistant:"
    )


def load_gsm8k_dataset(config: dict[str, Any], split: str, max_samples: int | None = None) -> Dataset:
    dataset = load_dataset(
        config["data"]["gsm_dataset_name"],
        config["data"]["gsm_config"],
        split=split,
    )
    dataset = _limit_dataset(dataset, max_samples, config["seed"])
    return dataset.map(
        lambda row: {
            "question": row["question"],
            "answer": row["answer"],
            "gold_answer": extract_gsm8k_gold_answer(row["answer"]),
            "prompt": format_gsm8k_prompt(row["question"]),
        },
        remove_columns=dataset.column_names,
    )
