import torch


def topk_tokens(logits, tokenizer, k=5):
    vals, ids = torch.topk(logits.detach().float().cpu(), k)
    rows = []
    for rank, (v, idx) in enumerate(zip(vals.tolist(), ids.tolist()), 1):
        rows.append({"rank": rank, "token_id": idx, "token": tokenizer.decode([idx]), "logit": v})
    return rows

