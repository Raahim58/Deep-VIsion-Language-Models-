import torch
from torch.utils.data import Dataset


def pad_right(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


class CaptionDataset(Dataset):
    def __init__(self, clip_tokens, rows, tokenizer, max_len=32):
        self.clip_tokens, self.rows, self.tokenizer, self.max_len = clip_tokens, rows, tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = self.tokenizer(row["caption"], add_special_tokens=False, truncation=True, max_length=self.max_len, return_tensors="pt").input_ids[0]
        return {"clip": self.clip_tokens[row["image_idx"]], "caption_ids": ids, "row": row}


def collate_caption(tokenizer):
    def _collate(batch):
        return {"clip": torch.stack([b["clip"] for b in batch]), "caption_ids": pad_right([b["caption_ids"] for b in batch], tokenizer.pad_token_id), "rows": [b["row"] for b in batch]}
    return _collate


class AVQADataset(Dataset):
    def __init__(self, clip_tokens, rows, tokenizer, max_q=32, max_a=12):
        self.clip_tokens, self.rows, self.tokenizer = clip_tokens, rows, tokenizer
        self.max_q, self.max_a = max_q, max_a

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        q = self.tokenizer(row["question"] + " Answer:", add_special_tokens=False, truncation=True, max_length=self.max_q, return_tensors="pt").input_ids[0]
        a = self.tokenizer(" " + row["answer"] + (self.tokenizer.eos_token or ""), add_special_tokens=False, truncation=True, max_length=self.max_a, return_tensors="pt").input_ids[0]
        return {"clip": self.clip_tokens[row["image_idx"]], "q_ids": q, "a_ids": a, "row": row}


def collate_a_vqa(tokenizer):
    def _collate(batch):
        return {
            "clip": torch.stack([b["clip"] for b in batch]),
            "q_ids": pad_right([b["q_ids"] for b in batch], tokenizer.pad_token_id),
            "a_ids": pad_right([b["a_ids"] for b in batch], tokenizer.pad_token_id),
            "rows": [b["row"] for b in batch],
        }
    return _collate


class TokenDataset(Dataset):
    def __init__(self, encoded):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate_token(tokenizer):
    def _collate(batch):
        ids, labels = [b["input_ids"] for b in batch], [b["labels"] for b in batch]
        max_len = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
        lab = torch.full((len(ids), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(ids), max_len), dtype=torch.long)
        for i, (x, y) in enumerate(zip(ids, labels)):
            input_ids[i, -len(x):] = x
            lab[i, -len(y):] = y
            attn[i, -len(x):] = 1
        return {"input_ids": input_ids, "labels": lab, "attention_mask": attn, "rows": [b.get("row") for b in batch]}
    return _collate


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

