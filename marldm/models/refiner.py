from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import ResBlock, group_norm


class PixelResidualRefiner(nn.Module):

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 16,
        channel_mult=(1, 2, 4, 8),
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        channels = [int(base_channels * value) for value in channel_mult]
        if len(channels) < 3:
            raise ValueError("channel_mult must contain at least three levels")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        input_channels = image_channels * 3 + 3
        self.input = nn.Conv2d(input_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level in range(len(channels) - 1):
            current = channels[level]
            self.down_blocks.append(nn.ModuleList([
                ResBlock(current, current), ResBlock(current, current),
            ]))
            self.downsamples.append(
                nn.Conv2d(current, channels[level + 1], 3, stride=2, padding=1)
            )
        self.mid = nn.ModuleList([
            ResBlock(channels[-1], channels[-1]),
            ResBlock(channels[-1], channels[-1]),
        ])
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channels) - 1)):
            incoming, skip = channels[level + 1], channels[level]
            self.upsamples.append(
                nn.ConvTranspose2d(incoming, skip, 4, stride=2, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([
                ResBlock(skip * 2, skip), ResBlock(skip, skip),
            ]))
        self.output = nn.Sequential(
            group_norm(channels[0]), nn.SiLU(),
            nn.Conv2d(channels[0], image_channels, 3, padding=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.output[-2].weight)
        nn.init.zeros_(self.output[-2].bias)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(
        self,
        source: torch.Tensor,
        coarse: torch.Tensor,
        li_image: torch.Tensor,
        confidence: torch.Tensor,
        support: torch.Tensor,
        anatomy_edge: torch.Tensor,
    ) -> torch.Tensor:
        size = source.shape[-2:]
        tensors = [coarse, li_image, confidence, support, anatomy_edge]
        tensors = [
            value if value.shape[-2:] == size else F.interpolate(
                value, size=size, mode="bilinear", align_corners=False
            )
            for value in tensors
        ]
        h = self.input(torch.cat((source, *tensors), dim=1))
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            h = self._run(blocks[0], h)
            h = self._run(blocks[1], h)
            skips.append(h)
            h = downsample(h)
        h = self._run(self.mid[0], h)
        h = self._run(self.mid[1], h)
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = upsample(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, skip.shape[-2:], mode="nearest")
            h = self._run(blocks[0], torch.cat((h, skip), dim=1))
            h = self._run(blocks[1], h)
        return self.output(h)
