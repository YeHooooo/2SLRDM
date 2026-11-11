from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResBlock, group_norm


class DiagonalGaussian:
    def __init__(self, moments: torch.Tensor):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-20.0, 10.0)

    def sample(self) -> torch.Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * (self.mean.square() + self.logvar.exp() - 1.0 - self.logvar).mean()


class CTVAE(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_mult: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        latent_scale: float = 1.0,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_mult]
        enc: list[nn.Module] = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        current = channels[0]
        for i, out in enumerate(channels):
            enc.extend([ResBlock(current, out), ResBlock(out, out)])
            current = out
            if i < len(channels) - 1:
                enc.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
        self.encoder = nn.Sequential(*enc)
        self.to_moments = nn.Conv2d(current, latent_channels * 2, 3, padding=1)

        self.from_latent = nn.Conv2d(latent_channels, current, 3, padding=1)
        dec: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        for i, out in enumerate(reversed_channels):
            dec.extend([ResBlock(current, out), ResBlock(out, out)])
            current = out
            if i < len(reversed_channels) - 1:
                next_channels = reversed_channels[i + 1]
                dec.extend([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(current, next_channels, 3, padding=1)])
                current = next_channels
        self.decoder = nn.Sequential(*dec)
        self.out = nn.Sequential(group_norm(current), nn.SiLU(), nn.Conv2d(current, in_channels, 3, padding=1), nn.Tanh())
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** (len(channels) - 1)
        self.latent_scale = float(latent_scale)

    def posterior(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.to_moments(self.encoder(x)))

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        posterior = self.posterior(x)
        z = posterior.sample() if sample else posterior.mode()
        return z * self.latent_scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.latent_scale
        return self.out(self.decoder(self.from_latent(z)))

    def forward(self, x: torch.Tensor, sample: bool = True):
        posterior = self.posterior(x)
        z = posterior.sample() if sample else posterior.mode()
        return self.decode(z * self.latent_scale), posterior


def build_vae(cfg) -> CTVAE:
    if cfg.type != "custom":
        raise ValueError(
            "This training package currently supports vae.type=custom. "
            "Foundation video VAEs use a 5D latent layout and require a matching 3D diffusion backbone."
        )
    return CTVAE(
        in_channels=cfg.in_channels,
        latent_channels=cfg.latent_channels,
        base_channels=cfg.base_channels,
        channel_mult=cfg.channel_mult,
        latent_scale=cfg.latent_scale,
    )


def load_frozen_vae(cfg, device: torch.device, checkpoint: str | Path | None = None) -> CTVAE:
    model = build_vae(cfg).to(device)
    path = Path(checkpoint or cfg.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {path}. Run train_vae.py first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model

