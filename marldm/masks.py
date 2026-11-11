from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _disk_footprint(kernel_size: int) -> np.ndarray:
    kernel_size = _odd_kernel(kernel_size)
    radius = kernel_size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy <= radius * radius)


def metal_artifact_masks(
    hu: np.ndarray,
    threshold: float = 2500.0,
    mask_dilate_kernel: int = 21,
    li_dilate_kernel: int = 15,
    streak_kernel: int = 31,
    streak_quantile: float = 0.975,
    min_body_hu: float = -850.0,
) -> tuple[np.ndarray, np.ndarray]:
    hu = np.asarray(hu, dtype=np.float32)
    core = hu > threshold
    if not core.any():
        zeros = np.zeros_like(hu, dtype=np.float32)
        return zeros, zeros.copy()

    def odd(value: int, minimum: int) -> int:
        value = max(minimum, int(value))
        return value if value % 2 else value + 1

    mask_dilate_kernel = odd(mask_dilate_kernel, 3)
    li_dilate_kernel = odd(li_dilate_kernel, 3)
    streak_kernel = odd(streak_kernel, 5)

    core_float = core.astype(np.float32)
    core_dilated = ndimage.maximum_filter(
        core_float, footprint=_disk_footprint(mask_dilate_kernel), mode="nearest"
    )
    li_core = ndimage.maximum_filter(
        core_float, footprint=_disk_footprint(li_dilate_kernel), mode="nearest"
    )
    li_mask = ndimage.gaussian_filter(li_core, sigma=1.0, mode="nearest")
    li_mask = np.clip(li_mask, 0.0, 1.0)

    body = ndimage.binary_fill_holes(hu > min_body_hu)
    body_eroded = ndimage.binary_erosion(body, structure=np.ones((21, 21), dtype=bool), border_value=0)
    local_mean = ndimage.uniform_filter(hu, size=streak_kernel, mode="nearest")
    contrast = np.abs(hu - local_mean) * body_eroded.astype(np.float32)
    quantile = float(np.quantile(contrast.reshape(-1), streak_quantile))
    streak = ((contrast > quantile) & body_eroded).astype(np.float32)
    streak = ndimage.gaussian_filter(streak, sigma=max(1.0, mask_dilate_kernel / 6.0), mode="nearest")

    artifact = np.clip(core_dilated + streak, 0.0, 1.0)
    artifact = ndimage.gaussian_filter(artifact, sigma=11.0 / 6.0, mode="nearest")
    artifact = np.clip(np.maximum(artifact, li_mask), 0.0, 1.0)
    return artifact.astype(np.float32), li_mask.astype(np.float32)


def metal_artifact_mask(hu: np.ndarray, **kwargs) -> np.ndarray:
    return metal_artifact_masks(hu, **kwargs)[0]


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _torch_disk_kernel(kernel_size: int, reference: torch.Tensor) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size, minimum=1)
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, device=reference.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx * xx + yy * yy) <= radius * radius).to(reference.dtype)[None, None]


def circular_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError("mask must be BCHW")
    kernel_size = _odd_kernel(kernel_size, minimum=1)
    if kernel_size == 1:
        return (mask > 0.5).to(mask.dtype)
    values = (mask > 0.5).to(mask.dtype)
    channels = values.shape[1]
    kernel = _torch_disk_kernel(kernel_size, values).expand(channels, 1, -1, -1)
    expanded = F.conv2d(values, kernel, padding=kernel_size // 2, groups=channels)
    return (expanded > 0).to(mask.dtype)


def gaussian_blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError("mask must be BCHW")
    kernel_size = _odd_kernel(kernel_size, minimum=1)
    if kernel_size == 1:
        return mask.clamp(0.0, 1.0)
    radius = kernel_size // 2
    sigma = max(0.5, kernel_size / 6.0)
    coordinates = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    one_dimensional = torch.exp(-0.5 * (coordinates / sigma).square())
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = (one_dimensional[:, None] * one_dimensional[None, :])[None, None]
    channels = mask.shape[1]
    kernel = kernel.expand(channels, 1, -1, -1)
    return F.conv2d(mask, kernel, padding=radius, groups=channels).clamp(0.0, 1.0)


def soft_circular_expand(
    mask: torch.Tensor,
    threshold: float = 0.05,
    dilate_kernel: int = 7,
    blur_kernel: int = 5,
) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError("mask must be BCHW")
    floor = min(max(float(threshold), 0.0), 0.99)
    soft = ((mask.float() - floor).clamp_min(0.0) / (1.0 - floor)).clamp(0.0, 1.0)
    dilate_kernel = _odd_kernel(dilate_kernel, minimum=1)
    if dilate_kernel > 1:
        channels = soft.shape[1]
        disk = _torch_disk_kernel(dilate_kernel, soft).expand(channels, 1, -1, -1)
        soft = F.conv2d(soft, disk, padding=dilate_kernel // 2, groups=channels).clamp(0.0, 1.0)
    soft = gaussian_blur_mask(soft, blur_kernel)
    return torch.maximum(soft, mask.float().clamp(0.0, 1.0)).to(mask.dtype)


@torch.no_grad()
def metal_fill_mask(metal_mask: torch.Tensor, dilate_kernel: int = 9, blur_kernel: int = 5) -> torch.Tensor:
    if metal_mask.ndim != 4:
        raise ValueError("metal_mask must be BCHW")
    if metal_mask.shape[1] != 1:
        metal_mask = metal_mask[:, metal_mask.shape[1] // 2 : metal_mask.shape[1] // 2 + 1]
    core = (metal_mask > 0.5).float()
    expanded = circular_dilate(core, dilate_kernel)
    softened = gaussian_blur_mask(expanded, blur_kernel)
    return torch.maximum(core, softened).clamp(0.0, 1.0)


@torch.no_grad()
def near_metal_ring_mask(
    metal_mask: torch.Tensor,
    outer_kernel: int = 61,
    blur_kernel: int = 7,
) -> torch.Tensor:
    if metal_mask.ndim != 4:
        raise ValueError("metal_mask must be BCHW")
    if metal_mask.shape[1] != 1:
        center = metal_mask.shape[1] // 2
        metal_mask = metal_mask[:, center : center + 1]
    core = (metal_mask > 0.5).float()
    outer = circular_dilate(core, outer_kernel)
    outer = gaussian_blur_mask(outer, blur_kernel)
    return (outer * (1.0 - core)).clamp(0.0, 1.0)


def _replicate_local_mean(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size, minimum=1)
    if kernel_size == 1:
        return image
    radius = kernel_size // 2
    horizontal = F.avg_pool2d(
        F.pad(image, (radius, radius, 0, 0), mode="replicate"),
        kernel_size=(1, kernel_size),
        stride=1,
    )
    return F.avg_pool2d(
        F.pad(horizontal, (0, 0, radius, radius), mode="replicate"),
        kernel_size=(kernel_size, 1),
        stride=1,
    )


@torch.no_grad()
def residual_artifact_mask(
    source: torch.Tensor,
    li_image: torch.Tensor,
    metal_mask: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    residual_threshold_hu: float = 80.0,
    residual_softness_hu: float = 20.0,
    core_dilate_kernel: int = 15,
    blur_kernel: int = 9,
    min_body_hu: float = -950.0,
) -> torch.Tensor:
    if source.ndim != 4 or li_image.shape != source.shape:
        raise ValueError("source and li_image must be same-shaped BCHW tensors")
    center = source.shape[1] // 2
    source_center = source[:, center : center + 1].float()
    li_center = li_image[:, center : center + 1].float()
    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    residual_hu = (source_center - li_center).abs() * hu_per_unit
    source_hu = (source_center + 1.0) * hu_per_unit + float(hu_min)
    body = (source_hu > float(min_body_hu)).float()

    softness = max(1.0, float(residual_softness_hu))
    score = torch.sigmoid((residual_hu - float(residual_threshold_hu)) / softness) * body
    if metal_mask.shape[1] != 1:
        metal_mask = metal_mask[:, center : center + 1]
    metal_core = (metal_mask > 0.5).float()
    core = circular_dilate(metal_core, core_dilate_kernel)
    core = gaussian_blur_mask(core, blur_kernel)
    mask = torch.maximum(score, core)
    mask = gaussian_blur_mask(mask, blur_kernel)
    return torch.maximum(mask, metal_core).clamp(0.0, 1.0)


@torch.no_grad()
def residual_artifact_masks(
    source: torch.Tensor,
    li_image: torch.Tensor,
    metal_mask: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    low_threshold_hu: float = 50.0,
    high_threshold_hu: float = 180.0,
    residual_softness_hu: float = 25.0,
    core_dilate_kernel: int = 11,
    confidence_blur_kernel: int = 9,
    support_dilate_kernel: int = 15,
    support_blur_kernel: int = 5,
    hysteresis_steps: int = 12,
    min_body_hu: float = -950.0,
    anatomy_edge_scale_hu: float = 180.0,
    direct_threshold_hu: float = 100.0,
    dark_threshold_hu: float = 70.0,
    dark_softness_hu: float = 15.0,
    dark_blur_kernel: int = 15,
    edge_confidence_floor: float = 0.35,
    severity_scale_hu: float = 300.0,
    near_metal_outer_kernel: int = 61,
    near_metal_ring_blur_kernel: int = 7,
    near_metal_local_kernel: int = 61,
    near_metal_dark_threshold_hu: float = 100.0,
    near_metal_dark_softness_hu: float = 20.0,
    near_metal_reference_min_hu: float = -600.0,
    near_metal_reference_softness_hu: float = 80.0,
    near_metal_reference_max_hu: float = 500.0,
    near_metal_max_confidence: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source.ndim != 4 or li_image.shape != source.shape:
        raise ValueError("source and li_image must be same-shaped BCHW tensors")
    if float(high_threshold_hu) <= float(low_threshold_hu):
        raise ValueError("high_threshold_hu must be greater than low_threshold_hu")

    center = source.shape[1] // 2
    source_center = source[:, center : center + 1].float()
    li_center = li_image[:, center : center + 1].float()
    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    signed_residual_hu = (source_center - li_center) * hu_per_unit
    residual_hu = signed_residual_hu.abs()
    source_hu = (source_center + 1.0) * hu_per_unit + float(hu_min)
    body = (source_hu > float(min_body_hu)).float()

    if metal_mask.shape[1] != 1:
        metal_mask = metal_mask[:, center : center + 1]
    metal_core = (metal_mask > 0.5).float()
    strong = (residual_hu >= float(high_threshold_hu)).float() * body
    weak = (residual_hu >= float(low_threshold_hu)).float() * body
    connected = torch.maximum(strong, metal_core)
    for _ in range(max(0, int(hysteresis_steps))):
        grown = circular_dilate(connected, 3) * weak
        connected = torch.maximum(connected, grown)

    softness = max(1.0, float(residual_softness_hu))
    weak_score = torch.sigmoid((residual_hu - float(low_threshold_hu)) / softness) * body
    strong_score = torch.sigmoid(
        (residual_hu - float(high_threshold_hu)) / softness
    ) * body
    direct_score = torch.sigmoid(
        (residual_hu - float(direct_threshold_hu)) / softness
    ) * body

    dark_softness = max(1.0, float(dark_softness_hu))
    dark_score = torch.sigmoid(
        (-signed_residual_hu - float(dark_threshold_hu)) / dark_softness
    ) * body
    dark_score = torch.maximum(
        dark_score, gaussian_blur_mask(dark_score, dark_blur_kernel)
    )

    ring = near_metal_ring_mask(
        metal_core,
        outer_kernel=near_metal_outer_kernel,
        blur_kernel=near_metal_ring_blur_kernel,
    )
    reference_ceiling = float(near_metal_reference_max_hu)
    reference_input = torch.maximum(
        source_hu.clamp(max=reference_ceiling),
        ((li_center + 1.0) * hu_per_unit + float(hu_min)).clamp(
            max=reference_ceiling
        ),
    )
    local_reference_hu = _replicate_local_mean(
        reference_input, near_metal_local_kernel
    )
    local_darkness_hu = local_reference_hu - source_hu
    near_softness = max(1.0, float(near_metal_dark_softness_hu))
    reference_softness = max(1.0, float(near_metal_reference_softness_hu))
    tissue_context = torch.sigmoid(
        (local_reference_hu - float(near_metal_reference_min_hu))
        / reference_softness
    )
    near_dark_score = torch.sigmoid(
        (local_darkness_hu - float(near_metal_dark_threshold_hu))
        / near_softness
    )
    near_dark_score = (
        near_dark_score
        * tissue_context
        * ring
        * min(max(float(near_metal_max_confidence), 0.0), 1.0)
    ).clamp(0.0, 1.0)

    support_binary = circular_dilate(connected, support_dilate_kernel)
    support = gaussian_blur_mask(support_binary, support_blur_kernel)
    support = torch.maximum(support, connected)
    support = torch.maximum(support, direct_score)
    support = torch.maximum(support, dark_score).clamp(0.0, 1.0)
    support = torch.maximum(support, near_dark_score).clamp(0.0, 1.0)

    dx = F.pad((source_center[..., :, 1:] - source_center[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((source_center[..., 1:, :] - source_center[..., :-1, :]).abs(), (0, 0, 0, 1))
    anatomy_edge_hu = (dx + dy) * hu_per_unit
    edge_gate = torch.exp(-anatomy_edge_hu / max(1.0, float(anatomy_edge_scale_hu)))
    edge_protection = strong_score + (1.0 - strong_score) * edge_gate
    edge_floor = min(max(float(edge_confidence_floor), 0.0), 1.0)
    protected_edge_weight = edge_floor + (1.0 - edge_floor) * edge_protection
    direct_edge_weight = edge_floor + (1.0 - edge_floor) * direct_score

    core_soft = gaussian_blur_mask(
        circular_dilate(metal_core, core_dilate_kernel), confidence_blur_kernel
    )
    connected_confidence = support * weak_score * protected_edge_weight
    direct_confidence = direct_score * direct_edge_weight
    confidence = torch.maximum(connected_confidence, direct_confidence)
    confidence = torch.maximum(confidence, dark_score)
    confidence = torch.maximum(confidence, near_dark_score)
    confidence = torch.maximum(confidence, core_soft)
    confidence = gaussian_blur_mask(confidence, confidence_blur_kernel)
    confidence = torch.maximum(confidence, direct_confidence)
    confidence = torch.maximum(confidence, dark_score)
    confidence = torch.maximum(confidence, near_dark_score)
    confidence = torch.maximum(confidence, core_soft)
    confidence = torch.maximum(confidence, metal_core).clamp(0.0, 1.0)
    support = torch.maximum(support, confidence).clamp(0.0, 1.0)

    support_sum = support.flatten(1).sum(dim=1).clamp_min(1.0)
    rms_hu = torch.sqrt(
        (residual_hu.square() * support).flatten(1).sum(dim=1) / support_sum
    )
    severity = (rms_hu / max(1.0, float(severity_scale_hu))).clamp(0.0, 3.0)
    return confidence, support, severity[:, None, None, None]
