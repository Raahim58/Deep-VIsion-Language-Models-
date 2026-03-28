from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from data.parsing import parse_hh_example


def _limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def load_hh_dataset(config: dict[str, Any], split: str, max_samples: int | None = None) -> Dataset:
    dataset = load_dataset(
        config["data"]["hh_dataset_name"],
        data_dir=config["data"]["hh_data_dir"],
        split=split,
    )
    dataset = _limit_dataset(dataset, max_samples, config["seed"])
    return dataset.map(parse_hh_example, remove_columns=dataset.column_names)


def make_sft_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(lambda row: {"prompt": row["prompt"], "response": row["chosen"]})


def make_dpo_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda row: {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
    )


def make_prompt_dataset(dataset: Dataset) -> Dataset:
    return dataset.map(lambda row: {"prompt": row["prompt"]})
