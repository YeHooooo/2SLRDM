from __future__ import annotations

import torch

from .masks import soft_circular_expand


@torch.no_grad()
def transfer_real_artifact_residual(
    clean: torch.Tensor,
    donor_metal: torch.Tensor,
    donor_li: torch.Tensor,
    artifact_mask: torch.Tensor,
    metal_core_mask: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    base_residual_hu: float = 1000.0,
    max_residual_hu: float = 1800.0,
    severity: torch.Tensor | None = None,
    severity_gain: float = 0.75,
    dark_residual_gain: float = 1.0,
    min_scale: float = 0.6,
    max_scale: float = 1.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.shape != donor_metal.shape or clean.shape != donor_li.shape:
        raise ValueError("clean, donor_metal and donor_li must have the same shape")
    mask = artifact_mask
    core = metal_core_mask
    if mask.shape[1] == 1 and clean.shape[1] != 1:
        mask = mask.expand(-1, clean.shape[1], -1, -1)
    if core.shape[1] == 1 and clean.shape[1] != 1:
        core = core.expand(-1, clean.shape[1], -1, -1)
    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    residual_hu = (donor_metal.float() - donor_li.float()) * hu_per_unit
    dark_gain = max(0.0, float(dark_residual_gain))
    residual_hu = torch.where(residual_hu < 0.0, residual_hu * dark_gain, residual_hu)
    base_limit = max(1.0, float(base_residual_hu))
    maximum_limit = max(base_limit, float(max_residual_hu))
    if severity is None:
        limit = torch.full(
            (clean.shape[0], 1, 1, 1), base_limit, device=clean.device, dtype=torch.float32
        )
    else:
        limit = base_limit * (1.0 + float(severity_gain) * severity.float())
        limit = limit.clamp(min=base_limit, max=maximum_limit)
    residual_hu = limit * torch.tanh(residual_hu / limit)
    residual = residual_hu / hu_per_unit
    residual = residual * mask * (1.0 - (core > 0.5).to(residual.dtype))
    scales = torch.empty(clean.shape[0], 1, 1, 1, device=clean.device).uniform_(
        float(min_scale), float(max_scale)
    )
    transferred = residual * scales
    pseudo_raw = (clean.float() + transferred).clamp(-1.0, 1.0)
    return pseudo_raw.to(clean.dtype), transferred.to(clean.dtype)


@torch.no_grad()
def li_delta_hint(raw_condition: torch.Tensor, li_condition: torch.Tensor, scale_hu: float, hu_min: float, hu_max: float):
    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    center = raw_condition.shape[1] // 2
    delta = (
        raw_condition[:, center : center + 1].float()
        - li_condition[:, center : center + 1].float()
    )
    return (delta * hu_per_unit / max(1.0, float(scale_hu))).clamp(-1.0, 1.0)


@torch.no_grad()
def dark_tail_evidence(
    signed_li_delta: torch.Tensor,
    support: torch.Tensor,
    *,
    delta_scale_hu: float,
    threshold_hu: float = 35.0,
    softness_hu: float = 12.0,
) -> torch.Tensor:
    if signed_li_delta.ndim != 4 or support.ndim != 4:
        raise ValueError("signed_li_delta and support must be BCHW tensors")
    if signed_li_delta.shape[0] != support.shape[0] or signed_li_delta.shape[-2:] != support.shape[-2:]:
        raise ValueError("signed_li_delta and support must have matching batch/spatial shapes")
    scale = max(1.0, float(delta_scale_hu))
    threshold = float(threshold_hu) / scale
    softness = max(1.0, float(softness_hu)) / scale
    dark = torch.sigmoid((-signed_li_delta.float() - threshold) / softness)
    return (dark * support.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)


@torch.no_grad()
def signed_residual_focus_maps(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    streak_threshold_hu: float = 40.0,
    dark_threshold_hu: float = 40.0,
    softness_hu: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.ndim != 4 or support.ndim != 4:
        raise ValueError("residual and support must be BCHW tensors")
    center = residual.shape[1] // 2
    residual_hu = residual[:, center : center + 1].float()
    residual_hu = residual_hu * (0.5 * (float(hu_max) - float(hu_min)))
    softness = max(1.0, float(softness_hu))
    support = support.float().clamp(0.0, 1.0)
    streak = torch.sigmoid(
        (residual_hu.abs() - float(streak_threshold_hu)) / softness
    ) * support
    dark = torch.sigmoid(
        (-residual_hu - float(dark_threshold_hu)) / softness
    ) * support
    return streak.clamp(0.0, 1.0), dark.clamp(0.0, 1.0)


@torch.no_grad()
def expanded_sampling_mask(
    mask: torch.Tensor,
    threshold: float = 0.05,
    dilate_kernel: int = 7,
    blur_kernel: int = 5,
) -> torch.Tensor:
    return soft_circular_expand(mask, threshold, dilate_kernel, blur_kernel)
