import torch


def _neg(dtype):
    return torch.finfo(dtype).min


def mask_text_logits(logits, visual_start, k):
    out = logits.clone()
    out[..., visual_start:visual_start + k] = _neg(out.dtype)
    return out


def mask_image_logits(logits, visual_start, k, end_image_id=None):
    out = torch.full_like(logits, _neg(logits.dtype))
    out[..., visual_start:visual_start + k] = logits[..., visual_start:visual_start + k]
    if end_image_id is not None:
        out[..., end_image_id] = logits[..., end_image_id]
    return out

