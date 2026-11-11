from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int | None = None):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.time = nn.Linear(time_dim, out_channels) if time_dim is not None else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time is not None:
            if temb is None:
                raise ValueError("time embedding is required")
            h = h + self.time(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        out, _ = self.attn(q, q, q, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000) / max(1, half - 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        phase = t.float()[:, None] * freq[None]
        emb = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

