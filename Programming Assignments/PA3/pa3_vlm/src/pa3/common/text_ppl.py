import numpy as np
import torch
from tqdm.auto import tqdm
from .metrics import perplexity


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, device, batch_size=4, max_length=256, desc="PPL"):
    model.eval()
    losses = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        losses.append(float(out.loss.detach().cpu()))
    loss = float(np.mean(losses)) if losses else float("nan")
    ppl = perplexity(loss)
    print(f"{desc}: loss={loss:.4f} ppl={ppl:.3f}")
    return ppl, loss

