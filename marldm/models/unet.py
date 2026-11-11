from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResBlock, SinusoidalTimeEmbedding, SpatialAttention, group_norm


class ConditionalUNet(nn.Module):

    def __init__(self, latent_channels: int = 4, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.latent_channels = latent_channels
        input_channels = latent_channels * 3 + 2
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.down1 = nn.Sequential()
        self.d1a = ResBlock(base_channels, base_channels, time_dim)
        self.d1b = ResBlock(base_channels, base_channels, time_dim)
        self.ds1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.d2a = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.d2b = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.ds2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.attn = SpatialAttention(base_channels * 4, heads=4)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.us2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.u2a = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.u2b = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.us1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.u1a = ResBlock(base_channels * 2, base_channels, time_dim)
        self.u1b = ResBlock(base_channels, base_channels, time_dim)
        self.output = nn.Sequential(group_norm(base_channels), nn.SiLU(), nn.Conv2d(base_channels, latent_channels, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_li: torch.Tensor,
        z_raw: torch.Tensor,
        mask: torch.Tensor,
        li_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask.shape[-2:] != z_t.shape[-2:]:
            mask = F.interpolate(mask, z_t.shape[-2:], mode="bilinear", align_corners=False)
        if z_li.shape[-2:] != z_t.shape[-2:]:
            z_li = F.interpolate(z_li, z_t.shape[-2:], mode="bilinear", align_corners=False)
        if z_raw.shape[-2:] != z_t.shape[-2:]:
            z_raw = F.interpolate(z_raw, z_t.shape[-2:], mode="bilinear", align_corners=False)
        if li_delta is None:
            li_delta = torch.zeros_like(mask)
        if li_delta.shape[-2:] != z_t.shape[-2:]:
            li_delta = F.interpolate(
                li_delta, z_t.shape[-2:], mode="bilinear", align_corners=False
            )
        temb = self.time_embed(t)
        h = self.input(torch.cat([z_t, z_li, z_raw, mask, li_delta], dim=1))
        s1 = self.d1b(self.d1a(h, temb), temb)
        h = self.ds1(s1)
        s2 = self.d2b(self.d2a(h, temb), temb)
        h = self.ds2(s2)
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)
        h = self.us2(h)
        if h.shape[-2:] != s2.shape[-2:]:
            h = F.interpolate(h, s2.shape[-2:], mode="nearest")
        h = self.u2b(self.u2a(torch.cat([h, s2], dim=1), temb), temb)
        h = self.us1(h)
        if h.shape[-2:] != s1.shape[-2:]:
            h = F.interpolate(h, s1.shape[-2:], mode="nearest")
        h = self.u1b(self.u1a(torch.cat([h, s1], dim=1), temb), temb)
        return self.output(h)
