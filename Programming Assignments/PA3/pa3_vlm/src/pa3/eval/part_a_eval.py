import torch
import pandas as pd
from tqdm.auto import tqdm
from pa3.common.metrics import normalize_answer, exact_match
from pa3.common.generation import topk_tokens


@torch.no_grad()
def generate_a_answer(lm, connector, tokenizer, clip_token, question, device, max_new_tokens=6, text_only=False):
    bos = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
    q = tokenizer(question + " Answer:", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if text_only:
        prefix = torch.cat([bos, q], dim=1)
        out = lm.generate(input_ids=prefix, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new = out[0, prefix.size(1):]
    else:
        v = connector(clip_token[None].to(device))
        emb = torch.cat([lm.get_input_embeddings()(bos), v, lm.get_input_embeddings()(q)], dim=1)
        out = lm.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=device), max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new = out[0]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


@torch.no_grad()
def eval_a_vqa(lm, connector, tokenizer, rows, clip_tokens, device, n=None, text_only=False):
    rows = rows[:n] if n else rows
    preds = []
    for r in tqdm(rows, desc="A VQA eval"):
        preds.append(generate_a_answer(lm, connector, tokenizer, clip_tokens[r["image_idx"]], r["question"], device, text_only=text_only))
    df = pd.DataFrame([{**r, "pred": normalize_answer(p), "raw_pred": p, "correct": normalize_answer(p) == normalize_answer(r["answer"])} for r, p in zip(rows, preds)])
    acc = float(df.correct.mean()) if len(df) else 0.0
    print("A exact-match accuracy:", acc)
    return acc, df


@torch.no_grad()
def a_top5(lm, connector, tokenizer, row, clip_tokens, device):
    bos = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
    q = tokenizer(row["question"] + " Answer:", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    v = connector(clip_tokens[row["image_idx"]][None].to(device))
    emb = torch.cat([lm.get_input_embeddings()(bos), v, lm.get_input_embeddings()(q)], dim=1)
    logits = lm(inputs_embeds=emb).logits[0, -1]
    return topk_tokens(logits, tokenizer)

