from __future__ import annotations

from dataclasses import dataclass

import torch


def _left_pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    output = []
    for seq in sequences:
        output.append([pad_value] * (max_len - len(seq)) + seq)
    return torch.tensor(output, dtype=torch.long)


def _right_pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    output = []
    for seq in sequences:
        output.append(seq + [pad_value] * (max_len - len(seq)))
    return torch.tensor(output, dtype=torch.long)


def tokenize_prompt_response(tokenizer, prompt: str, response: str, max_length: int) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False, truncation=True, max_length=max_length)[
        "input_ids"
    ]
    response_mask = [0] * len(full_ids)
    response_tokens = max(0, len(full_ids) - min(len(prompt_ids), len(full_ids)))
    if response_tokens:
        response_mask[-response_tokens:] = [1] * response_tokens
    return full_ids, response_mask


@dataclass
class SFTCollator:
    tokenizer: any
    max_length: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_masks = []
        for row in batch:
            full_ids, response_mask = tokenize_prompt_response(
                self.tokenizer, row["prompt"], row["response"], self.max_length
            )
            labels = [token if mask else -100 for token, mask in zip(full_ids, response_mask)]
            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_masks.append([1] * len(full_ids))
        return {
            "input_ids": _left_pad_sequences(input_ids_list, self.tokenizer.pad_token_id),
            "attention_mask": _left_pad_sequences(attention_masks, 0),
            "labels": _left_pad_sequences(labels_list, -100),
        }


@dataclass
class PairwiseRewardCollator:
    tokenizer: any
    max_length: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        chosen = [
            self.tokenizer(
                row["prompt"] + row["chosen"],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            for row in batch
        ]
        rejected = [
            self.tokenizer(
                row["prompt"] + row["rejected"],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            for row in batch
        ]
        return {
            "chosen_input_ids": _right_pad_sequences(chosen, self.tokenizer.pad_token_id),
            "chosen_attention_mask": _right_pad_sequences([[1] * len(ids) for ids in chosen], 0),
            "rejected_input_ids": _right_pad_sequences(rejected, self.tokenizer.pad_token_id),
            "rejected_attention_mask": _right_pad_sequences([[1] * len(ids) for ids in rejected], 0),
        }


@dataclass
class DPOCollator:
    tokenizer: any
    max_length: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        chosen_input_ids = []
        chosen_masks = []
        chosen_resp_masks = []
        rejected_input_ids = []
        rejected_masks = []
        rejected_resp_masks = []
        for row in batch:
            chosen_ids, chosen_resp = tokenize_prompt_response(
                self.tokenizer, row["prompt"], row["chosen"], self.max_length
            )
            rejected_ids, rejected_resp = tokenize_prompt_response(
                self.tokenizer, row["prompt"], row["rejected"], self.max_length
            )
            chosen_input_ids.append(chosen_ids)
            chosen_masks.append([1] * len(chosen_ids))
            chosen_resp_masks.append(chosen_resp)
            rejected_input_ids.append(rejected_ids)
            rejected_masks.append([1] * len(rejected_ids))
            rejected_resp_masks.append(rejected_resp)
        return {
            "chosen_input_ids": _left_pad_sequences(chosen_input_ids, self.tokenizer.pad_token_id),
            "chosen_attention_mask": _left_pad_sequences(chosen_masks, 0),
            "chosen_response_mask": _left_pad_sequences(chosen_resp_masks, 0),
            "rejected_input_ids": _left_pad_sequences(rejected_input_ids, self.tokenizer.pad_token_id),
            "rejected_attention_mask": _left_pad_sequences(rejected_masks, 0),
            "rejected_response_mask": _left_pad_sequences(rejected_resp_masks, 0),
        }


@dataclass
class PromptCollator:
    tokenizer: any
    max_length: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        encoded = [
            self.tokenizer(
                row["prompt"], truncation=True, max_length=self.max_length, add_special_tokens=False
            )["input_ids"]
            for row in batch
        ]
        return {
            "input_ids": _left_pad_sequences(encoded, self.tokenizer.pad_token_id),
            "attention_mask": _left_pad_sequences([[1] * len(ids) for ids in encoded], 0),
        }
