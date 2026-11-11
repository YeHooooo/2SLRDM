from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .utils import weighted_l1


def separated_transfer_mask(
    support: torch.Tensor,
    confidence: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    return (
        support.float().clamp(0.0, 1.0)
        * confidence.float().clamp(0.0, 1.0).pow(max(0.05, float(gamma)))
    ).clamp(0.0, 1.0)


def directional_gradient_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    losses = []
    for dy, dx in directions:
        y0 = max(0, -dy)
        y1 = pred.shape[-2] - max(0, dy)
        x0 = max(0, -dx)
        x1 = pred.shape[-1] - max(0, dx)
        shifted_y0, shifted_y1 = y0 + dy, y1 + dy
        shifted_x0, shifted_x1 = x0 + dx, x1 + dx
        pred_diff = pred[..., shifted_y0:shifted_y1, shifted_x0:shifted_x1] - pred[..., y0:y1, x0:x1]
        target_diff = target[..., shifted_y0:shifted_y1, shifted_x0:shifted_x1] - target[..., y0:y1, x0:x1]
        error = (pred_diff - target_diff).abs()
        if weight is None:
            losses.append(error.mean())
        else:
            local = weight[..., shifted_y0:shifted_y1, shifted_x0:shifted_x1]
            if local.shape[1] == 1 and error.shape[1] != 1:
                local = local.expand(-1, error.shape[1], -1, -1)
            losses.append((error * local).sum() / local.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def laplacian_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    channels = pred.shape[1]
    kernel = pred.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    pred_lap = F.conv2d(pred, kernel, padding=1, groups=channels)
    target_lap = F.conv2d(target, kernel, padding=1, groups=channels)
    return weighted_l1(pred_lap, target_lap, weight)


def _rotate(x: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    theta = x.new_tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0]])
    theta = theta.unsqueeze(0).expand(x.shape[0], -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def projection_consistency_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    metal_mask: torch.Tensor,
    angles=(0, 30, 60, 90, 120, 150),
    image_size: int = 96,
    trace_threshold: float = 0.02,
) -> torch.Tensor:
    center = pred.shape[1] // 2
    pred = pred[:, center : center + 1].float()
    target = target[:, center : center + 1].float()
    if metal_mask.shape[1] != 1:
        metal_mask = metal_mask[:, center : center + 1]
    size = (int(image_size), int(image_size))
    pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False, antialias=True)
    target = F.interpolate(target, size=size, mode="bilinear", align_corners=False, antialias=True)
    metal = F.interpolate(metal_mask.float(), size=size, mode="nearest")
    losses = []
    for angle in angles:
        pred_projection = _rotate(pred, angle).mean(dim=-2)
        target_projection = _rotate(target, angle).mean(dim=-2)
        metal_trace = _rotate(metal, angle).amax(dim=-2)
        valid = (metal_trace < float(trace_threshold)).to(pred.dtype)
        error = (pred_projection - target_projection).abs()
        losses.append((error * valid).sum() / valid.sum().clamp_min(1.0))
    return torch.stack(losses).mean()
