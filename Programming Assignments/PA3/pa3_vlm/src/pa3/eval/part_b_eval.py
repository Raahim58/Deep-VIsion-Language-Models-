import torch
import pandas as pd
from tqdm.auto import tqdm
from pa3.common.metrics import normalize_answer
from pa3.models.logit_masks import mask_text_logits, mask_image_logits


@torch.no_grad()
def eval_b_vqa(model, tokenizer, rows, codes, token_ids, device, n=500):
    model.eval(); model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
    rows = rows[:min(n, len(rows))]
    preds = []
    for r in tqdm(rows, desc="B VQA"):
        v = codes[r["image_idx"]].long().to(device) + token_ids["visual_start"]
        q = tokenizer(r["question"] + " Answer:", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        prefix = torch.cat([torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id, token_ids["image"]]], device=device), v[None], torch.tensor([[token_ids["end_image"]]], device=device), q], dim=1)
        cur = prefix
        for _ in range(6):
            logits = mask_text_logits(model(input_ids=cur).logits[:, -1], token_ids["visual_start"], 256)
            nxt = logits.argmax(-1, keepdim=True); cur = torch.cat([cur, nxt], 1)
            if int(nxt.item()) == tokenizer.eos_token_id: break
        preds.append(normalize_answer(tokenizer.decode(cur[0, prefix.size(1):], skip_special_tokens=True)))
    df = pd.DataFrame([{**r, "pred": p, "correct": p == normalize_answer(r["answer"])} for r, p in zip(rows, preds)])
    print("B VQA accuracy:", float(df.correct.mean()) if len(df) else 0.0)
    return float(df.correct.mean()) if len(df) else 0.0, df


@torch.no_grad()
def generate_image_tokens(model, tokenizer, prompt, token_ids, device, temperature=1.0, n_tokens=16):
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    cur = torch.cat([torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device), ids, torch.tensor([[token_ids["image"]]], device=device)], 1)
    raw_hist = masked_hist = None
    for i in range(n_tokens):
        logits = model(input_ids=cur).logits[:, -1] / temperature
        if i == 0: raw_hist = logits.detach().float().cpu()
        masked = mask_image_logits(logits, token_ids["visual_start"], 256, token_ids["end_image"])
        if i == 0: masked_hist = masked[torch.isfinite(masked)].detach().float().cpu()
        nxt = torch.multinomial(torch.softmax(masked, -1), 1)
        cur = torch.cat([cur, nxt], 1)
    return cur[0, -n_tokens:].cpu(), raw_hist, masked_hist

