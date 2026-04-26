import torch
import torch.nn as nn


class OverlayEmbedding(nn.Module):
    def __init__(self, base_embedding, v_txt=49152, n_new=258):
        super().__init__()
        self.base = base_embedding
        self.v_txt = v_txt
        self.overlay = nn.Embedding(n_new, base_embedding.embedding_dim)
        self.embedding_dim = base_embedding.embedding_dim
        self.num_embeddings = v_txt + n_new
        self.base.weight.requires_grad_(False)
        with torch.no_grad():
            mean = self.base.weight[:v_txt].float().mean(0)
            self.overlay.weight[:2].copy_(mean)
            self.overlay.weight[2:].zero_()

    @property
    def weight(self):
        return torch.cat([self.base.weight[:self.v_txt], self.overlay.weight], dim=0)

    def forward(self, input_ids):
        base_ids = input_ids.clamp(max=self.v_txt - 1)
        out = self.base(base_ids)
        mask = input_ids >= self.v_txt
        if mask.any():
            out = out.clone()
            out[mask] = self.overlay((input_ids[mask] - self.v_txt).long()).to(out.dtype)
        return out


def install_overlay(model, v_txt=49152, n_new=258):
    overlay = OverlayEmbedding(model.get_input_embeddings(), v_txt=v_txt, n_new=n_new)
    model.set_input_embeddings(overlay)
    return overlay


def visual_text_norm_ratio(overlay, k=256):
    with torch.no_grad():
        text = overlay.base.weight[:overlay.v_txt].float().norm(dim=-1).mean().item()
        visual = overlay.overlay.weight[2:2 + k].float().norm(dim=-1).mean().item()
    ratio = visual / max(text, 1e-8)
    print(f"visual/text norm ratio={ratio:.4f} visual={visual:.4f} text={text:.4f}")
    return ratio

