from __future__ import annotations

import torch
import torch.nn.functional as F

from .masks import circular_dilate, gaussian_blur_mask, near_metal_ring_mask


def _expand_like(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and image.shape[1] != 1:
        return mask.expand(-1, image.shape[1], -1, -1)
    return mask


@torch.no_grad()
def li_guided_metal_removal(
    output: torch.Tensor,
    li_image: torch.Tensor,
    metal_core: torch.Tensor,
    *,
    dilate_kernel: int = 3,
    blur_kernel: int = 7,
    transition_gain: float = 1.0,
) -> torch.Tensor:
    if output.shape != li_image.shape:
        raise ValueError("output and li_image must have the same shape")
    if output.ndim != 4 or metal_core.ndim != 4:
        raise ValueError("output, li_image and metal_core must be BCHW tensors")
    core_single = (
        metal_core[:, metal_core.shape[1] // 2 : metal_core.shape[1] // 2 + 1]
        if metal_core.shape[1] != 1
        else metal_core
    )
    core_single = (core_single > 0.5).to(output.dtype)
    hard_removal = circular_dilate(core_single, int(dilate_kernel))
    transition = gaussian_blur_mask(hard_removal, int(blur_kernel))
    gain = min(max(float(transition_gain), 0.0), 1.0)
    blend_single = torch.maximum(
        hard_removal,
        gain * transition * (1.0 - hard_removal),
    )
    blend = _expand_like(blend_single, output)
    return (
        output.float() * (1.0 - blend.float())
        + li_image.float() * blend.float()
    ).clamp(-1.0, 1.0).to(output.dtype)


@torch.no_grad()
def adaptive_dual_candidate(
    source: torch.Tensor,
    corrected_decode: torch.Tensor,
    raw_decode: torch.Tensor,
    artifact_confidence: torch.Tensor,
    *,
    full_candidate_threshold: float = 0.55,
    full_candidate_gamma: float = 0.75,
    dark_tail: torch.Tensor | None = None,
    dark_tail_full_candidate_gain: float = 0.0,
) -> torch.Tensor:
    if source.shape != corrected_decode.shape or source.shape != raw_decode.shape:
        raise ValueError("source, corrected_decode and raw_decode must have the same shape")
    confidence = _expand_like(artifact_confidence.float(), source)
    threshold = min(max(float(full_candidate_threshold), 0.0), 0.99)
    gamma = max(0.05, float(full_candidate_gamma))
    full_gate = ((confidence - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    full_gate = full_gate.pow(gamma)
    if dark_tail is not None and float(dark_tail_full_candidate_gain) > 0.0:
        tail = _expand_like(dark_tail.float().clamp(0.0, 1.0), source)
        tail_gain = min(max(float(dark_tail_full_candidate_gain), 0.0), 1.0)
        full_gate = torch.maximum(full_gate, tail_gain * tail)

    delta_candidate = source.float() + corrected_decode.float() - raw_decode.float()
    candidate = delta_candidate + full_gate * (corrected_decode.float() - delta_candidate)
    return candidate.clamp(-1.0, 1.0).to(source.dtype)


def anatomy_edge_map(
    source: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    edge_scale_hu: float = 120.0,
    dilate_kernel: int = 3,
) -> torch.Tensor:
    source = source.float()
    dx = F.pad((source[..., :, 1:] - source[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((source[..., 1:, :] - source[..., :-1, :]).abs(), (0, 0, 0, 1))
    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    gradient_hu = torch.sqrt(dx.square() + dy.square()) * hu_per_unit
    scale = max(1.0, float(edge_scale_hu))
    edge = gradient_hu / (gradient_hu + scale)
    kernel = max(1, int(dilate_kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1:
        edge = F.max_pool2d(edge, kernel_size=kernel, stride=1, padding=kernel // 2)
    return edge.clamp(0.0, 1.0)


@torch.no_grad()
def safe_li_residual_fusion(
    source: torch.Tensor,
    li_image: torch.Tensor,
    prediction: torch.Tensor,
    artifact_mask: torch.Tensor,
    metal_core: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    max_correction_hu: float = 250.0,
    confidence_scale_hu: float = 250.0,
) -> torch.Tensor:
    if source.shape != li_image.shape or source.shape != prediction.shape:
        raise ValueError("source, li_image and prediction must have the same shape")
    if artifact_mask.shape[1] == 1 and source.shape[1] != 1:
        artifact_mask = artifact_mask.expand(-1, source.shape[1], -1, -1)
    if metal_core.shape[1] == 1 and source.shape[1] != 1:
        metal_core = metal_core.expand(-1, source.shape[1], -1, -1)

    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    delta_hu = (prediction.float() - li_image.float()) * hu_per_unit
    limit = max(1.0, float(max_correction_hu))
    bounded_delta = limit * torch.tanh(delta_hu / limit)
    confidence = torch.exp(-delta_hu.abs() / max(1.0, float(confidence_scale_hu)))
    candidate = li_image.float() + bounded_delta * confidence / hu_per_unit
    output = source.float() * (1.0 - artifact_mask) + candidate * artifact_mask
    output = torch.where(metal_core > 0.5, source.float(), output)
    return output.clamp(-1.0, 1.0).to(source.dtype)


@torch.no_grad()
def near_metal_dark_rescue(
    source: torch.Tensor,
    li_image: torch.Tensor,
    output: torch.Tensor,
    artifact_confidence: torch.Tensor,
    metal_core: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    outer_kernel: int = 61,
    ring_blur_kernel: int = 7,
    residual_threshold_hu: float = 60.0,
    residual_softness_hu: float = 20.0,
    output_gap_threshold_hu: float = 60.0,
    confidence_threshold: float = 0.70,
    confidence_transition: float = 0.10,
    confidence_gamma: float = 0.50,
    output_dark_threshold_hu: float = -25.0,
    output_dark_softness_hu: float = 30.0,
    min_body_hu: float = -1025.0,
    li_target_min_hu: float = -200.0,
    li_target_softness_hu: float = 40.0,
    max_correction_hu: float = 1200.0,
    blend_gain: float = 1.0,
    mask_blur_kernel: int = 5,
    rescue_passes: int = 1,
) -> torch.Tensor:
    if source.shape != li_image.shape or source.shape != output.shape:
        raise ValueError("source, li_image and output must have the same shape")
    if source.ndim != 4:
        raise ValueError("source, li_image and output must be BCHW tensors")

    confidence = _expand_like(artifact_confidence.float(), source)
    metal = _expand_like((metal_core > 0.5).float(), source)
    ring = _expand_like(
        near_metal_ring_mask(
            metal_core,
            outer_kernel=int(outer_kernel),
            blur_kernel=int(ring_blur_kernel),
        ),
        source,
    )

    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    source_hu = (source.float() + 1.0) * hu_per_unit + float(hu_min)
    li_hu = (li_image.float() + 1.0) * hu_per_unit + float(hu_min)
    output_hu = (output.float() + 1.0) * hu_per_unit + float(hu_min)

    residual_softness = max(1.0, float(residual_softness_hu))
    source_residual_evidence = torch.sigmoid(
        (
            li_hu
            - source_hu
            - float(residual_threshold_hu)
        )
        / residual_softness
    )
    confidence_floor = min(max(float(confidence_threshold), 0.0), 0.99)
    confidence_activation = (
        (confidence - confidence_floor)
        / max(0.01, float(confidence_transition))
    ).clamp(0.0, 1.0)
    confidence_gate = confidence_activation * confidence.clamp(
        0.0, 1.0
    ).pow(max(0.05, float(confidence_gamma)))
    body = (source_hu > float(min_body_hu)).to(source.dtype)
    li_context = torch.sigmoid(
        (
            li_hu
            - float(li_target_min_hu)
        )
        / max(1.0, float(li_target_softness_hu))
    )

    gain = min(max(float(blend_gain), 0.0), 1.0)
    dark_softness = max(1.0, float(output_dark_softness_hu))
    rescued_hu = output_hu
    for _ in range(max(1, int(rescue_passes))):
        output_gap_evidence = torch.sigmoid(
            (
                li_hu
                - rescued_hu
                - float(output_gap_threshold_hu)
            )
            / residual_softness
        )
        residual_evidence = torch.maximum(
            source_residual_evidence, output_gap_evidence
        )
        output_darkness = torch.sigmoid(
            (
                float(output_dark_threshold_hu)
                - rescued_hu
            )
            / dark_softness
        )
        rescue_mask = (
            ring
            * residual_evidence
            * confidence_gate
            * output_darkness
            * body
            * li_context
            * (1.0 - metal)
        ).clamp(0.0, 1.0)
        if int(mask_blur_kernel) > 1:
            rescue_mask = gaussian_blur_mask(
                rescue_mask, int(mask_blur_kernel)
            )
            rescue_mask = rescue_mask * ring * body * (1.0 - metal)

        desired_hu = (li_hu - rescued_hu).clamp_min(0.0)
        correction_hu = (
            gain * rescue_mask * desired_hu
        ).clamp(0.0, max(1.0, float(max_correction_hu)))
        rescued_hu = rescued_hu + correction_hu
    rescued = (
        (rescued_hu - float(hu_min))
        / (float(hu_max) - float(hu_min))
        * 2.0
        - 1.0
    )
    rescued = torch.where(metal > 0.5, source.float(), rescued)
    return rescued.clamp(-1.0, 1.0).to(output.dtype)


@torch.no_grad()
def source_residual_fusion(
    source: torch.Tensor,
    prediction: torch.Tensor,
    artifact_mask: torch.Tensor,
    metal_core: torch.Tensor,
    *,
    hu_min: float,
    hu_max: float,
    max_correction_hu: float = 800.0,
    base_correction_hu: float | None = None,
    artifact_strength: torch.Tensor | None = None,
    confidence_threshold: float = 0.0,
    confidence_gamma: float = 1.0,
    edge_protection_strength: float = 0.0,
    edge_scale_hu: float = 120.0,
    edge_dilate_kernel: int = 3,
    strong_artifact_threshold: float = 0.65,
    transition_blur_kernel: int = 1,
    dark_tail: torch.Tensor | None = None,
    dark_tail_fusion_gain: float = 0.0,
    dark_tail_max_correction_hu: float = 450.0,
    dark_tail_blur_kernel: int = 7,
) -> torch.Tensor:
    if source.shape != prediction.shape:
        raise ValueError("source and prediction must have the same shape")
    if artifact_mask.shape[1] == 1 and source.shape[1] != 1:
        artifact_mask = artifact_mask.expand(-1, source.shape[1], -1, -1)
    if metal_core.shape[1] == 1 and source.shape[1] != 1:
        metal_core = metal_core.expand(-1, source.shape[1], -1, -1)
    floor = min(max(float(confidence_threshold), 0.0), 0.99)
    gamma = max(0.05, float(confidence_gamma))

    def confidence_gate(value: torch.Tensor) -> torch.Tensor:
        normalized = ((value.float() - floor) / (1.0 - floor)).clamp(0.0, 1.0)
        return normalized.pow(gamma)

    base_fusion_gate = confidence_gate(artifact_mask)
    fusion_gate = base_fusion_gate
    hu_per_unit = 0.5 * (float(hu_max) - float(hu_min))
    delta_hu = (prediction.float() - source.float()) * hu_per_unit
    maximum = max(1.0, float(max_correction_hu))
    base = maximum if base_correction_hu is None else min(maximum, max(1.0, float(base_correction_hu)))
    strength = artifact_mask.float() if artifact_strength is None else artifact_strength.float()
    if strength.shape[1] == 1 and source.shape[1] != 1:
        strength = strength.expand(-1, source.shape[1], -1, -1)
    strength = confidence_gate(strength)

    raw_strength = artifact_mask.float() if artifact_strength is None else artifact_strength.float()
    raw_strength = _expand_like(raw_strength, source)
    strong_floor = min(max(float(strong_artifact_threshold), 0.0), 0.99)
    strong_gate = ((raw_strength - strong_floor) / (1.0 - strong_floor)).clamp(0.0, 1.0)

    edge_attenuation = torch.ones_like(fusion_gate)
    protection_strength = min(max(float(edge_protection_strength), 0.0), 1.0)
    if protection_strength > 0.0:
        source_edges = anatomy_edge_map(
            source,
            hu_min=hu_min,
            hu_max=hu_max,
            edge_scale_hu=edge_scale_hu,
            dilate_kernel=edge_dilate_kernel,
        )
        prediction_edges = anatomy_edge_map(
            prediction,
            hu_min=hu_min,
            hu_max=hu_max,
            edge_scale_hu=edge_scale_hu,
            dilate_kernel=edge_dilate_kernel,
        )
        common_edges = torch.minimum(source_edges, prediction_edges)
        edge_attenuation = 1.0 - protection_strength * common_edges * (1.0 - strong_gate)
        fusion_gate = fusion_gate * edge_attenuation.clamp(0.0, 1.0)

    if int(transition_blur_kernel) > 1:
        fusion_gate = gaussian_blur_mask(fusion_gate, int(transition_blur_kernel))
        fusion_gate = torch.maximum(fusion_gate, base_fusion_gate * strong_gate)

    limit = base + (maximum - base) * strength.clamp(0.0, 1.0)
    bounded_delta = limit * torch.tanh(delta_hu / limit)
    correction_hu = fusion_gate * bounded_delta

    tail_gain = min(max(float(dark_tail_fusion_gain), 0.0), 1.0)
    if dark_tail is not None and tail_gain > 0.0:
        tail_gate = _expand_like(dark_tail.float().clamp(0.0, 1.0), source) * tail_gain
        if int(dark_tail_blur_kernel) > 1:
            tail_gate = gaussian_blur_mask(tail_gate, int(dark_tail_blur_kernel))
        tail_gate = tail_gate * edge_attenuation.clamp(0.0, 1.0)
        tail_gate = tail_gate * (1.0 - fusion_gate).clamp(0.0, 1.0)
        tail_limit = max(1.0, float(dark_tail_max_correction_hu))
        tail_delta = tail_limit * torch.tanh(delta_hu / tail_limit)
        correction_hu = correction_hu + tail_gate * tail_delta

    output = source.float() + correction_hu / hu_per_unit
    output = torch.where(metal_core > 0.5, source.float(), output)
    return output.clamp(-1.0, 1.0).to(source.dtype)
