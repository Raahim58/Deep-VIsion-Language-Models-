import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, k=256, d=64, beta=0.25, ema=False, decay=0.99, dead_threshold=2):
        super().__init__()
        self.k, self.d, self.beta = k, d, beta
        self.ema, self.decay, self.dead_threshold = ema, decay, dead_threshold
        self.codebook = nn.Embedding(k, d)
        self.codebook.weight.data.uniform_(-1 / k, 1 / k)
        self.register_buffer("ema_count", torch.zeros(k))
        self.register_buffer("ema_sum", torch.zeros(k, d))

    def forward(self, ze):
        z = ze.permute(0, 2, 3, 1).contiguous()
        flat = z.view(-1, self.d)
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(1)
        idx = dist.argmin(1)
        zq_flat = self.codebook(idx)
        zq = zq_flat.view_as(z).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(zq, ze.detach())
        commitment_loss = F.mse_loss(ze, zq.detach())
        q_loss = codebook_loss + self.beta * commitment_loss
        zq_st = ze + (zq - ze).detach()
        usage, perplexity, dead = self.stats(idx)
        if self.ema and self.training:
            self.ema_update(idx, flat)
        return zq_st, idx.view(ze.size(0), ze.size(2), ze.size(3)), q_loss, codebook_loss, commitment_loss, perplexity, dead, usage

    @torch.no_grad()
    def stats(self, idx):
        usage = torch.bincount(idx, minlength=self.k).float()
        probs = usage / usage.sum().clamp_min(1)
        nz = probs > 0
        perplexity = torch.exp(-(probs[nz] * probs[nz].log()).sum())
        dead = int((usage < self.dead_threshold).sum().item())
        return usage, perplexity, dead

    @torch.no_grad()
    def ema_update(self, idx, flat):
        usage = torch.bincount(idx, minlength=self.k).float()
        sums = torch.zeros_like(self.ema_sum)
        sums.index_add_(0, idx, flat.detach())
        self.ema_count.mul_(self.decay).add_(usage, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        total = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-5) / (total + self.k * 1e-5) * total
        self.codebook.weight.data.copy_(self.ema_sum / smoothed.unsqueeze(1).clamp_min(1e-5))
        dead = torch.where(self.ema_count < self.dead_threshold)[0]
        if len(dead) > 0 and flat.size(0) > 0:
            rand = torch.randint(0, flat.size(0), (len(dead),), device=flat.device)
            self.codebook.weight.data[dead] = flat[rand]
            self.ema_sum[dead] = flat[rand]
            self.ema_count[dead] = self.dead_threshold + 1

