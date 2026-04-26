import torch.nn as nn
import torch.nn.functional as F
from .vector_quantizer import VectorQuantizer


class Encoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, d, 3, 1, 1), nn.GroupNorm(8, d), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(self, k=256, d=64, beta=0.25, ema=False, decay=0.99, dead_threshold=2):
        super().__init__()
        self.encoder = Encoder(d)
        self.quantizer = VectorQuantizer(k, d, beta, ema=ema, decay=decay, dead_threshold=dead_threshold)
        self.decoder = Decoder(d)

    def forward(self, x):
        ze = self.encoder(x)
        zq, idx, q_loss, cb, com, perp, dead, usage = self.quantizer(ze)
        recon = self.decoder(zq)
        recon_mse = F.mse_loss(recon, x)
        return {
            "ze": ze, "zq": zq, "idx": idx, "recon": recon,
            "loss": recon_mse + q_loss, "recon_mse": recon_mse,
            "codebook_loss": cb, "commitment_loss": com,
            "codebook_perplexity": perp, "dead_codes": dead, "usage": usage,
        }

