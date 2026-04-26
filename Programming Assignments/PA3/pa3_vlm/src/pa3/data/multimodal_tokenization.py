import torch


def pad_left(seqs, pad_id, label=False):
    max_len = max(len(s) for s in seqs)
    fill = -100 if label else pad_id
    out = torch.full((len(seqs), max_len), fill, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, -len(s):] = s
        attn[i, -len(s):] = 1
    return out, attn


def visual_ids_from_codes(codes, v_txt=49152):
    return codes.long() + v_txt + 2


def encode_multimodal(tokenizer, v_ids, question, answer, image_id, end_image_id):
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    q = tokenizer(question + " Answer:", add_special_tokens=False, return_tensors="pt").input_ids[0]
    a = tokenizer(" " + answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
    eos = torch.tensor([tokenizer.eos_token_id])
    ids = torch.cat([torch.tensor([bos, image_id]), v_ids.long(), torch.tensor([end_image_id]), q, a, eos])
    prefix = 1 + 1 + len(v_ids) + 1 + len(q)
    labels = torch.cat([torch.full((prefix,), -100), a.long(), eos.long()])
    return ids.long(), labels.long()


def encode_imagegen(tokenizer, v_ids, prompt, image_id, end_image_id):
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    p = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    eos = torch.tensor([tokenizer.eos_token_id])
    ids = torch.cat([torch.tensor([bos]), p, torch.tensor([image_id]), v_ids.long(), torch.tensor([end_image_id]), eos])
    prefix = 1 + len(p) + 1
    labels = torch.cat([torch.full((prefix,), -100), v_ids.long(), torch.tensor([end_image_id]), eos.long()])
    return ids.long(), labels.long()


def token_type_sequence(ids, tokenizer, image_id, end_image_id, visual_start, k):
    out = []
    for tid in ids.tolist():
        if tid == tokenizer.pad_token_id:
            out.append("PAD")
        elif tid == (tokenizer.bos_token_id or tokenizer.eos_token_id):
            out.append("BOS")
        elif tid == tokenizer.eos_token_id:
            out.append("EOS")
        elif tid == image_id:
            out.append("<image>")
        elif tid == end_image_id:
            out.append("</image>")
        elif visual_start <= tid < visual_start + k:
            out.append("VIS")
        else:
            out.append("TEXT")
    return out

