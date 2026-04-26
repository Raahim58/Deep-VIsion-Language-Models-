import torch


def build_caption_inputs(lm, connector, tokenizer, clip_tokens, caption_ids, device):
    bsz = clip_tokens.size(0)
    clip_tokens = clip_tokens.to(device)
    caption_ids = caption_ids.to(device)
    bos = torch.full((bsz, 1), tokenizer.bos_token_id or tokenizer.eos_token_id, dtype=torch.long, device=device)
    bos_emb = lm.get_input_embeddings()(bos)
    v_emb = connector(clip_tokens)
    cap_emb = lm.get_input_embeddings()(caption_ids)
    inputs_embeds = torch.cat([bos_emb, v_emb, cap_emb], dim=1)
    labels = torch.cat([torch.full((bsz, 1 + v_emb.size(1)), -100, device=device, dtype=torch.long), caption_ids], dim=1)
    labels[:, 1 + v_emb.size(1):][caption_ids == tokenizer.pad_token_id] = -100
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, labels, attn


def build_vqa_inputs(lm, connector, tokenizer, clip_tokens, q_ids, a_ids, device):
    bsz = clip_tokens.size(0)
    clip_tokens, q_ids, a_ids = clip_tokens.to(device), q_ids.to(device), a_ids.to(device)
    bos = torch.full((bsz, 1), tokenizer.bos_token_id or tokenizer.eos_token_id, dtype=torch.long, device=device)
    bos_emb = lm.get_input_embeddings()(bos)
    v_emb = connector(clip_tokens)
    q_emb = lm.get_input_embeddings()(q_ids)
    a_emb = lm.get_input_embeddings()(a_ids)
    inputs_embeds = torch.cat([bos_emb, v_emb, q_emb, a_emb], dim=1)
    prefix_len = 1 + v_emb.size(1) + q_ids.size(1)
    labels = torch.cat([torch.full((bsz, prefix_len), -100, dtype=torch.long, device=device), a_ids.clone()], dim=1)
    labels[:, prefix_len:][a_ids == tokenizer.pad_token_id] = -100
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, labels, attn


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model

