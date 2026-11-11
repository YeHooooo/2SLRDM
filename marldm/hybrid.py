from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from . import CONDITIONING_VERSION, REFINER_VERSION
from .fusion import anatomy_edge_map, source_residual_fusion
from .models import (
    ConditionalUNet,
    GaussianDiffusion,
    PixelResidualRefiner,
    load_frozen_vae,
)


def resize_image(x: torch.Tensor, size) -> torch.Tensor:
    if x.shape[-2:] == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def build_refiner(cfg) -> PixelResidualRefiner:
    return PixelResidualRefiner(
        image_channels=cfg.refiner.image_channels,
        base_channels=cfg.refiner.base_channels,
        channel_mult=cfg.refiner.channel_mult,
        gradient_checkpointing=cfg.refiner.gradient_checkpointing,
    )


def build_coarse_model(cfg, device):
    return ConditionalUNet(
        cfg.vae.latent_channels,
        cfg.diffusion.base_channels,
        cfg.diffusion.time_dim,
    ).to(device)


def _load_coarse_state(model, checkpoint):
    payload = torch.load(
        Path(checkpoint), map_location="cpu", weights_only=False
    )
    if payload.get("conditioning_version") != CONDITIONING_VERSION:
        raise ValueError(
            f"Coarse checkpoint must be {CONDITIONING_VERSION}; "
            f"got {payload.get('conditioning_version')!r}"
        )
    model.load_state_dict(payload.get("ema") or payload.get("model", payload))
    return payload


def build_joint_components(cfg, device, initialize_ldm=True):
    vae = load_frozen_vae(cfg.vae, device)
    model = build_coarse_model(cfg, device)
    if initialize_ldm:
        initialization = str(cfg.train_joint.ldm_initialization).lower()
        if initialization == "model_test5":
            _load_coarse_state(model, cfg.diffusion.coarse_checkpoint)
        elif initialization != "random":
            raise ValueError(
                "train_joint.ldm_initialization must be 'model_test5' or 'random'"
            )
    model.train().requires_grad_(True)
    diffusion = GaussianDiffusion(
        cfg.diffusion.timesteps, cfg.diffusion.beta_schedule
    ).to(device)
    return vae, model, diffusion


def _coarse_conditions(vae, raw, li, support, li_delta, coarse_size):
    size = (int(coarse_size), int(coarse_size))
    raw_small = resize_image(raw, size)
    li_small = resize_image(li, size)
    support_small = F.interpolate(support, size=size, mode="bilinear", align_corners=False)
    delta_small = F.interpolate(li_delta, size=size, mode="bilinear", align_corners=False)
    z_raw = vae.encode(raw_small, sample=False)
    z_li = vae.encode(li_small, sample=False)
    latent_support = F.interpolate(
        support_small, z_raw.shape[-2:], mode="bilinear", align_corners=False
    )
    latent_delta = F.interpolate(
        delta_small, z_raw.shape[-2:], mode="bilinear", align_corners=False
    )
    return raw_small, z_raw, z_li, latent_support, latent_delta


def coarse_from_training_pair(
    vae,
    coarse_model,
    diffusion,
    clean,
    raw,
    li,
    confidence,
    support,
    metal,
    li_delta,
    cfg,
    source_anchor=None,
):
    with torch.no_grad():
        raw_small, z_raw, z_li, latent_support, latent_delta = _coarse_conditions(
            vae, raw, li, support, li_delta, cfg.data.coarse_size
        )
        clean_small = resize_image(
            clean, (int(cfg.data.coarse_size), int(cfg.data.coarse_size))
        )
        z_clean = vae.encode(clean_small, sample=False)
    residual0 = z_raw - z_clean
    max_t = min(
        int(cfg.refiner.coarse_train_timestep_max), diffusion.timesteps - 1
    )
    t = torch.randint(0, max_t + 1, (raw.shape[0],), device=raw.device)
    noise = torch.randn_like(residual0)
    residual_t = diffusion.q_sample(residual0, t, noise)
    pred_noise = coarse_model(
        residual_t, t, z_li, z_raw, latent_support, latent_delta
    )
    pred_residual0 = diffusion.predict_x0(
        residual_t.float(), t, pred_noise.float()
    ).clamp(-4.0, 4.0)
    latent_weight = 1.0 + float(cfg.train_joint.ldm_support_weight) * latent_support
    expanded_weight = latent_weight.expand_as(pred_noise)
    diffusion_noise_loss = (
        (pred_noise.float() - noise.float()).square() * expanded_weight
    ).sum() / expanded_weight.sum().clamp_min(1.0)
    diffusion_x0_loss = (
        (pred_residual0 - residual0.float()).abs() * expanded_weight
    ).sum() / expanded_weight.sum().clamp_min(1.0)

    decoded = vae.decode(torch.cat((z_raw - pred_residual0, z_raw), dim=0))
    corrected_small, raw_decode_small = decoded.chunk(2, dim=0)
    coarse = fuse_coarse_candidate(
        raw if source_anchor is None else source_anchor,
        corrected_small.float(),
        raw_decode_small.float(),
        confidence,
        metal,
        cfg,
    )
    return coarse, diffusion_noise_loss, diffusion_x0_loss


@torch.no_grad()
def sample_coarse(
    vae,
    coarse_model,
    diffusion,
    raw,
    li,
    confidence,
    sampling_support,
    metal,
    li_delta,
    cfg,
    source_anchor=None,
):
    _, z_raw, z_li, latent_support, latent_delta = _coarse_conditions(
        vae, raw, li, sampling_support, li_delta, cfg.data.coarse_size
    )
    residual = diffusion.ddim_sample_residual(
        coarse_model,
        z_li,
        z_raw,
        latent_support,
        latent_delta,
        steps=int(cfg.diffusion.sample_steps),
        eta=float(cfg.diffusion.sample_eta),
        start_timestep=int(cfg.diffusion.start_timestep),
        support_threshold=float(cfg.diffusion.sampling_support_threshold),
    )
    decoded = vae.decode(torch.cat((z_raw - residual, z_raw), dim=0))
    corrected_small, raw_decode_small = decoded.chunk(2, dim=0)
    return fuse_coarse_candidate(
        raw if source_anchor is None else source_anchor,
        corrected_small.float(), raw_decode_small.float(), confidence, metal, cfg
    )


def fuse_coarse_candidate(
    source, corrected_small, raw_decode_small, confidence, metal, cfg
):
    size = source.shape[-2:]
    corrected = resize_image(corrected_small, size)
    raw_decode = resize_image(raw_decode_small, size)
    candidate = (source.float() + corrected.float() - raw_decode.float()).clamp(-1.0, 1.0)
    return source_residual_fusion(
        source,
        candidate,
        confidence,
        metal,
        hu_min=cfg.data.hu_min,
        hu_max=cfg.data.hu_max,
        max_correction_hu=cfg.fusion.max_correction_hu,
        base_correction_hu=cfg.fusion.base_correction_hu,
        artifact_strength=confidence,
        confidence_threshold=cfg.fusion.confidence_threshold,
        confidence_gamma=cfg.fusion.confidence_gamma,
        edge_protection_strength=cfg.fusion.edge_protection_strength,
        edge_scale_hu=cfg.fusion.edge_scale_hu,
        edge_dilate_kernel=cfg.fusion.edge_dilate_kernel,
        strong_artifact_threshold=cfg.fusion.strong_artifact_threshold,
        transition_blur_kernel=cfg.fusion.transition_blur_kernel,
    )


def refiner_inputs(source, cfg):
    edge = anatomy_edge_map(
        source,
        hu_min=cfg.data.hu_min,
        hu_max=cfg.data.hu_max,
        edge_scale_hu=cfg.refiner.edge_input_scale_hu,
        dilate_kernel=cfg.refiner.edge_input_dilate_kernel,
    )
    center = edge.shape[1] // 2
    return edge[:, center : center + 1]


def apply_refiner(
    refiner,
    source,
    coarse,
    li,
    confidence,
    support,
    metal,
    cfg,
):
    edge = refiner_inputs(source, cfg)
    raw_prediction = refiner(source, coarse, li, confidence, support, edge)
    hu_per_unit = 0.5 * (float(cfg.data.hu_max) - float(cfg.data.hu_min))
    maximum = float(cfg.refiner.max_correction_hu) / hu_per_unit
    correction = raw_prediction.float() * maximum
    gate = support.float().clamp(0.0, 1.0) * confidence.float().clamp(
        0.0, 1.0
    ).pow(float(cfg.refiner.gate_confidence_gamma))
    floor = float(cfg.refiner.gate_confidence_threshold)
    gate = ((gate - floor) / max(1e-6, 1.0 - floor)).clamp(0.0, 1.0)
    if gate.shape[1] == 1 and correction.shape[1] != 1:
        gate = gate.expand(-1, correction.shape[1], -1, -1)
    if metal.shape[1] == 1 and correction.shape[1] != 1:
        metal = metal.expand(-1, correction.shape[1], -1, -1)
    applied = correction * gate * (1.0 - (metal > 0.5).to(correction.dtype))
    refined = (coarse.float() - applied).clamp(-1.0, 1.0)
    refined = torch.where(metal > 0.5, source.float(), refined)
    return refined.to(source.dtype), applied, gate[:, :1], edge


def load_joint_checkpoint(coarse_model, refiner, path, device, use_ema=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("refiner_version") != REFINER_VERSION:
        raise ValueError(
            f"Joint checkpoint must be {REFINER_VERSION}; "
            f"got {payload.get('refiner_version')!r}"
        )
    ldm_state = payload.get("ldm_ema") if use_ema else None
    coarse_model.load_state_dict(ldm_state or payload["ldm_model"])
    coarse_model.eval()
    if payload.get("refiner_enabled", True) is False:
        raise ValueError(
            "This checkpoint was trained with refiner_enabled=False, "
            "so it cannot be loaded with the refiner enabled."
        )
    refiner_state = payload.get("refiner_ema") if use_ema else None
    refiner_model = payload.get("refiner_model")
    if refiner_state is None and refiner_model is None:
        raise ValueError("Checkpoint does not contain refiner weights")
    refiner.load_state_dict(refiner_state or refiner_model)
    refiner.eval()
    return payload
