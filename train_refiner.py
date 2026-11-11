from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from marldm import REFINER_VERSION
from marldm.artifacts import (
    expanded_sampling_mask,
    li_delta_hint,
    transfer_real_artifact_residual,
)
from marldm.config import load_config
from marldm.data import CTSliceDataset
from marldm.hybrid import (
    apply_refiner,
    build_coarse_model,
    build_joint_components,
    build_refiner,
    coarse_from_training_pair,
    sample_coarse,
)
from marldm.losses import (
    directional_gradient_l1,
    laplacian_l1,
    projection_consistency_l1,
    separated_transfer_mask,
)
from marldm.masks import (
    metal_fill_mask,
    near_metal_ring_mask,
    residual_artifact_masks,
)
from marldm.projection_li import astra_worker_status, projection_domain_li
from marldm.utils import (
    EMA,
    amp_dtype,
    atomic_torch_save,
    autocast_context,
    cycle,
    cuda_device_index,
    memory_status,
    resolve_device,
    seed_everything,
    weighted_l1,
)


def make_dataset(cfg, seed: int, domain: str, split: str):
    root = cfg.with_metal if domain == "metal" else cfg.without_metal
    return CTSliceDataset(
        root=root,
        split=split,
        context_slices=cfg.context_slices,
        image_size=cfg.image_size,
        stride=cfg.train_stride if split == "train" else cfg.val_stride,
        val_fraction=cfg.val_fraction,
        seed=seed,
        hu_min=cfg.hu_min,
        hu_max=cfg.hu_max,
        with_mask=domain == "metal",
        metal_threshold=cfg.metal_threshold,
        mask_dilate_kernel=cfg.mask_dilate_kernel,
        li_dilate_kernel=cfg.li_dilate_kernel,
        streak_kernel=cfg.streak_kernel,
        streak_quantile=cfg.streak_quantile,
        min_body_hu=cfg.min_body_hu,
        li_root=getattr(cfg, "li_root", None) if domain == "metal" else None,
        li_strict=getattr(cfg, "li_strict", True),
        defer_crop=True,
    )


@torch.no_grad()
def donor_conditions(metal_image, metal_mask, cfg, donor_li=None):
    if donor_li is None:
        donor_li = projection_domain_li(metal_image, metal_mask, cfg.data, cfg.li)
    elif donor_li.shape != metal_image.shape:
        raise ValueError(
            f"Precomputed LI shape {tuple(donor_li.shape)} does not match "
            f"metal image shape {tuple(metal_image.shape)}"
        )
    confidence, support, severity = residual_artifact_masks(
        metal_image,
        donor_li,
        metal_mask,
        hu_min=cfg.data.hu_min,
        hu_max=cfg.data.hu_max,
        low_threshold_hu=cfg.artifact_mask.low_threshold_hu,
        high_threshold_hu=cfg.artifact_mask.high_threshold_hu,
        residual_softness_hu=cfg.artifact_mask.residual_softness_hu,
        core_dilate_kernel=cfg.artifact_mask.core_dilate_kernel,
        confidence_blur_kernel=cfg.artifact_mask.confidence_blur_kernel,
        support_dilate_kernel=cfg.artifact_mask.support_dilate_kernel,
        support_blur_kernel=cfg.artifact_mask.support_blur_kernel,
        hysteresis_steps=cfg.artifact_mask.hysteresis_steps,
        min_body_hu=cfg.artifact_mask.min_body_hu,
        anatomy_edge_scale_hu=cfg.artifact_mask.anatomy_edge_scale_hu,
        direct_threshold_hu=cfg.artifact_mask.direct_threshold_hu,
        dark_threshold_hu=cfg.artifact_mask.dark_threshold_hu,
        dark_softness_hu=cfg.artifact_mask.dark_softness_hu,
        dark_blur_kernel=cfg.artifact_mask.dark_blur_kernel,
        edge_confidence_floor=cfg.artifact_mask.edge_confidence_floor,
        severity_scale_hu=cfg.artifact_mask.severity_scale_hu,
        near_metal_outer_kernel=cfg.artifact_mask.near_metal_outer_kernel,
        near_metal_ring_blur_kernel=cfg.artifact_mask.near_metal_ring_blur_kernel,
        near_metal_local_kernel=cfg.artifact_mask.near_metal_local_kernel,
        near_metal_dark_threshold_hu=cfg.artifact_mask.near_metal_dark_threshold_hu,
        near_metal_dark_softness_hu=cfg.artifact_mask.near_metal_dark_softness_hu,
        near_metal_reference_min_hu=cfg.artifact_mask.near_metal_reference_min_hu,
        near_metal_reference_softness_hu=cfg.artifact_mask.near_metal_reference_softness_hu,
        near_metal_reference_max_hu=cfg.artifact_mask.near_metal_reference_max_hu,
        near_metal_max_confidence=cfg.artifact_mask.near_metal_max_confidence,
    )
    fill = metal_fill_mask(
        metal_mask,
        cfg.near_metal.condition_fill_dilate_kernel,
        cfg.near_metal.condition_fill_blur_kernel,
    )
    return donor_li, confidence, support, fill, severity


def _resize(x, size, nearest=False):
    if x.shape[-2:] == tuple(size):
        return x
    if nearest:
        return F.interpolate(x, size=size, mode="nearest")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


@torch.no_grad()
def build_pseudo_pair(
    clean,
    metal_image,
    metal_mask,
    cfg,
    metal_li=None,
    identity_probability: float = 0.10,
):
    size = (int(cfg.data.image_size), int(cfg.data.image_size))
    clean = _resize(clean, size)
    metal_image = _resize(metal_image, size)
    metal_mask = _resize(metal_mask, size, nearest=True)
    if metal_li is not None:
        metal_li = _resize(metal_li, size)
    donor_li, donor_confidence, donor_support, condition_fill, severity = donor_conditions(
        metal_image, metal_mask, cfg, metal_li
    )

    probability = float(identity_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("identity_probability must be in [0, 1]")
    identity = torch.rand(clean.shape[0], device=clean.device) < probability
    donor_confidence = donor_confidence.clone()
    donor_support = donor_support.clone()
    condition_fill = condition_fill.clone()
    metal_mask = metal_mask.clone()
    severity = severity.clone()
    donor_confidence[identity] = 0.0
    donor_support[identity] = 0.0
    condition_fill[identity] = 0.0
    metal_mask[identity] = 0.0
    severity[identity] = 0.0

    transfer_mask = separated_transfer_mask(
        donor_support,
        donor_confidence,
        gamma=cfg.artifact_transfer.transfer_mask_gamma,
    )
    pseudo_raw, transferred_residual = transfer_real_artifact_residual(
        clean,
        metal_image,
        donor_li,
        transfer_mask,
        metal_mask,
        hu_min=cfg.data.hu_min,
        hu_max=cfg.data.hu_max,
        base_residual_hu=cfg.artifact_transfer.base_residual_hu,
        max_residual_hu=cfg.artifact_transfer.max_residual_hu,
        severity=severity,
        severity_gain=cfg.artifact_transfer.severity_gain,
        dark_residual_gain=cfg.artifact_transfer.dark_residual_gain,
        min_scale=cfg.artifact_transfer.min_scale,
        max_scale=cfg.artifact_transfer.max_scale,
    )

    metal_binary = (metal_mask > 0.5).to(pseudo_raw.dtype)
    pseudo_source = (
        pseudo_raw * (1.0 - metal_binary) + metal_image * metal_binary
    ).clamp(-1.0, 1.0)
    pseudo_li = projection_domain_li(pseudo_source, metal_mask, cfg.data, cfg.li)

    _, confidence, support, _, pseudo_severity = donor_conditions(
        pseudo_source, metal_mask, cfg, pseudo_li
    )
    confidence = confidence.clone()
    support = support.clone()
    pseudo_severity = pseudo_severity.clone()
    confidence[identity] = 0.0
    support[identity] = 0.0
    pseudo_severity[identity] = 0.0

    raw_condition = (
        pseudo_source * (1.0 - condition_fill) + pseudo_li * condition_fill
    )
    li_delta = li_delta_hint(
        raw_condition,
        pseudo_li,
        cfg.artifact_transfer.li_delta_scale_hu,
        cfg.data.hu_min,
        cfg.data.hu_max,
    )

    center = transferred_residual.shape[1] // 2
    hu_per_unit = 0.5 * (float(cfg.data.hu_max) - float(cfg.data.hu_min))
    transferred_center_hu = (
        transferred_residual[:, center : center + 1].float() * hu_per_unit
    )
    dark_softness = max(1.0, float(cfg.near_metal.loss_dark_softness_hu))
    near_dark_target = torch.sigmoid(
        (
            -transferred_center_hu
            - float(cfg.near_metal.loss_dark_threshold_hu)
        )
        / dark_softness
    )
    near_dark_target = near_dark_target * (
        transferred_center_hu < 0.0
    ).to(near_dark_target.dtype)
    near_ring = near_metal_ring_mask(
        metal_mask,
        outer_kernel=cfg.artifact_mask.near_metal_outer_kernel,
        blur_kernel=cfg.artifact_mask.near_metal_ring_blur_kernel,
    )
    near_dark_target = (near_dark_target * near_ring).clamp(0.0, 1.0)
    near_dark_target[identity] = 0.0

    return (
        clean, pseudo_li, raw_condition, confidence, support, transfer_mask,
        metal_mask, li_delta, transferred_residual, pseudo_severity,
        pseudo_source, near_dark_target,
    )


def refiner_loss(
    refiner,
    vae,
    coarse_model,
    diffusion,
    clean,
    metal_image,
    metal_mask,
    cfg,
    micro_step: int,
    metal_li=None,
    force_projection=False,
):
    with torch.no_grad():
        (
            clean, pseudo_li, raw_condition, confidence, support, transfer_mask,
            pseudo_metal, li_delta, _, severity, pseudo_source, near_dark_target,
        ) = build_pseudo_pair(clean, metal_image, metal_mask, cfg, metal_li)
    coarse, diffusion_noise, diffusion_x0 = coarse_from_training_pair(
        vae, coarse_model, diffusion, clean, raw_condition, pseudo_li,
        confidence, support, pseudo_metal, li_delta, cfg,
        source_anchor=pseudo_source,
    )

    refined, applied, refine_gate, _ = apply_refiner(
        refiner,
        pseudo_source,
        coarse,
        pseudo_li,
        confidence,
        support,
        pseudo_metal,
        cfg,
    )
    center = pseudo_metal.shape[1] // 2
    metal_center = pseudo_metal[:, center : center + 1]
    valid = (1.0 - (metal_center > 0.5).float()).clamp(0.0, 1.0)
    image_weight = (
        1.0
        + float(cfg.refiner.confidence_loss_weight) * confidence
        + float(cfg.refiner.transfer_loss_weight) * transfer_mask
        + float(cfg.near_metal.loss_weight) * near_dark_target
    ) * valid
    target_correction = coarse.float() - clean.float()
    residual_match = weighted_l1(applied, target_correction, image_weight)
    image_loss = weighted_l1(refined, clean, image_weight)
    outside = weighted_l1(
        refined, clean, ((1.0 - support) * valid).clamp(0.0, 1.0)
    )
    structure_weight = (
        1.0
        + 2.0 * confidence
        + transfer_mask
        + float(cfg.near_metal.loss_weight) * near_dark_target
    ) * valid
    directional = directional_gradient_l1(refined, clean, structure_weight)
    laplacian = laplacian_l1(refined, clean, structure_weight)
    do_projection = force_projection or (
        int(cfg.refiner.projection_every) > 0
        and micro_step % int(cfg.refiner.projection_every) == 0
    )
    if do_projection:
        projection = projection_consistency_l1(
            refined,
            clean,
            pseudo_metal,
            angles=cfg.refiner.projection_angles,
            image_size=cfg.refiner.projection_size,
        )
    else:
        projection = refined.new_zeros(())
    total = (
        float(cfg.train_joint.diffusion_noise_weight) * diffusion_noise
        + float(cfg.train_joint.diffusion_x0_weight) * diffusion_x0
        + float(cfg.refiner.residual_weight) * residual_match
        + float(cfg.refiner.image_weight) * image_loss
        + float(cfg.refiner.outside_preserve_weight) * outside
        + float(cfg.refiner.directional_weight) * directional
        + float(cfg.refiner.laplacian_weight) * laplacian
        + float(cfg.refiner.projection_weight) * projection
    )
    metrics = {
        "diffusion_noise": diffusion_noise.detach(),
        "diffusion_x0": diffusion_x0.detach(),
        "residual": residual_match.detach(),
        "image": image_loss.detach(),
        "outside": outside.detach(),
        "directional": directional.detach(),
        "laplacian": laplacian.detach(),
        "projection": projection.detach(),
        "gate": refine_gate.mean().detach(),
        "transfer": transfer_mask.mean().detach(),
        "support": support.mean().detach(),
        "severity": severity.mean().detach(),
        "near_dark": near_dark_target.mean().detach(),
    }
    return total, metrics


class FixedValidationRandomness:
    def __init__(self, seed):
        self.seed = int(seed)

    def __enter__(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        self.cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, *_):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_state)


@torch.no_grad()
def validate(
    refiner, vae, coarse_model, diffusion, clean_loader, metal_loader,
    device, dtype, cfg, max_batches=10,
):
    refiner.eval()
    coarse_model.eval()
    losses, sums = [], {}
    with FixedValidationRandomness(cfg.seed + 9001):
        metal_iter = cycle(metal_loader)
        for index, clean_batch in enumerate(clean_loader):
            if index >= max_batches:
                break
            clean = clean_batch["image"].to(device, non_blocking=True)
            metal_batch = next(metal_iter)
            metal_image = metal_batch["image"].to(device, non_blocking=True)
            metal_mask = metal_batch["metal_mask"].to(device, non_blocking=True)
            metal_li = metal_batch.get("li")
            if metal_li is not None:
                metal_li = metal_li.to(device, non_blocking=True)
            with autocast_context(device, dtype):
                loss, metrics = refiner_loss(
                    refiner, vae, coarse_model, diffusion, clean, metal_image,
                    metal_mask, cfg, micro_step=0, metal_li=metal_li,
                    force_projection=True,
                )
            losses.append(float(loss.detach()))
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
    refiner.train()
    coarse_model.train()
    count = max(1, len(losses))
    return sum(losses) / count, {key: value / count for key, value in sums.items()}


def normalized_to_hu(x, cfg):
    return (x + 1.0) * 0.5 * (cfg.data.hu_max - cfg.data.hu_min) + cfg.data.hu_min


@torch.no_grad()
def save_preview(
    refiner, vae, coarse_model, diffusion, metal_batch, cfg, device, dtype,
    epoch, global_step,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refiner.eval()
    count = min(int(cfg.refiner.preview_num_images), metal_batch["image"].shape[0])
    source = metal_batch["image"][:count].to(device)
    metal = metal_batch["metal_mask"][:count].to(device)
    precomputed_li = metal_batch.get("li")
    if precomputed_li is not None:
        precomputed_li = precomputed_li[:count].to(device)
    li, confidence, support, fill, _ = donor_conditions(source, metal, cfg, precomputed_li)
    raw_condition = source * (1.0 - fill) + li * fill
    li_delta = li_delta_hint(
        raw_condition, li, cfg.artifact_transfer.li_delta_scale_hu,
        cfg.data.hu_min, cfg.data.hu_max,
    )
    sampling = expanded_sampling_mask(
        support, cfg.diffusion.sampling_mask_threshold,
        cfg.diffusion.sampling_mask_dilate_kernel,
        cfg.diffusion.sampling_mask_blur_kernel,
    )
    old_steps = cfg.diffusion.sample_steps
    cfg.diffusion.sample_steps = cfg.refiner.preview_sample_steps
    with autocast_context(device, dtype):
        coarse = sample_coarse(
            vae, coarse_model, diffusion, raw_condition, li, confidence,
            sampling, metal, li_delta, cfg, source_anchor=source,
        )
        refined, applied, gate, _ = apply_refiner(
            refiner, source, coarse, li, confidence, support, metal, cfg
        )
    cfg.diffusion.sample_steps = old_steps
    center = source.shape[1] // 2
    arrays = [
        normalized_to_hu(source[:, center].float().cpu().numpy(), cfg),
        normalized_to_hu(coarse[:, center].float().cpu().numpy(), cfg),
        normalized_to_hu(refined[:, center].float().cpu().numpy(), cfg),
        confidence[:, 0].cpu().numpy(),
        gate[:, 0].float().cpu().numpy(),
        applied[:, center].float().cpu().numpy()
        * (0.5 * (cfg.data.hu_max - cfg.data.hu_min)),
    ]
    final_title = "Hybrid final"
    titles = ["Input 512", "Coarse LDM", final_title, "Confidence", "Refiner gate", "Applied correction HU"]
    lo, hi = map(float, cfg.refiner.preview_window)
    fig, axes = plt.subplots(count, 6, figsize=(21, 3.5 * count), squeeze=False)
    for row in range(count):
        for col, title in enumerate(titles):
            image = arrays[col][row]
            if col in {3, 4}:
                axes[row, col].imshow(image, cmap="magma", vmin=0, vmax=1)
            elif col == 5:
                axes[row, col].imshow(image, cmap="coolwarm", vmin=-300, vmax=300)
            else:
                axes[row, col].imshow(image, cmap="gray", vmin=lo, vmax=hi)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.suptitle(f"Model_test10 near-metal dark | epoch {epoch + 1} | step {global_step}")
    fig.tight_layout()
    output_dir = Path(cfg.refiner.preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"hybrid_epoch_{epoch + 1:04d}_step_{global_step:08d}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    refiner.train()
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Train Model_test10 near-metal-aware LDM and 512x512 refiner"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Override vae.checkpoint, e.g. checkpoints/vae512_best.pt",
    )
    parser.add_argument(
        "--coarse-size",
        type=int,
        choices=(256, 512),
        default=None,
        help="Override the LDM/VAE working resolution",
    )
    parser.add_argument(
        "--with-metal-root",
        default=None,
        help="Override data.with_metal (useful for Linux server paths)",
    )
    parser.add_argument(
        "--without-metal-root",
        default=None,
        help="Override data.without_metal (useful for Linux server paths)",
    )
    parser.add_argument(
        "--li-root",
        default=None,
        help="Override data.li_root; omit to keep the config value",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output:
        cfg.train_joint.checkpoint = args.output
    if args.preview_dir:
        cfg.refiner.preview_dir = args.preview_dir
    if args.num_workers is not None:
        cfg.data.num_workers = max(0, int(args.num_workers))
    if args.vae_checkpoint:
        cfg.vae.checkpoint = args.vae_checkpoint
    if args.coarse_size is not None:
        cfg.data.coarse_size = int(args.coarse_size)
    if args.with_metal_root is not None:
        cfg.data.with_metal = args.with_metal_root
    if args.without_metal_root is not None:
        cfg.data.without_metal = args.without_metal_root
    if args.li_root is not None:
        cfg.data.li_root = args.li_root
    if int(cfg.data.image_size) != 512:
        raise ValueError("Model_test10 requires data.image_size=512")

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = amp_dtype(cfg.train_joint.amp) if device.type == "cuda" else None
    print(
        "LI memory policy: "
        f"backend={cfg.li.backend} "
        f"persistent_workspace=True "
        f"isolated_process={getattr(cfg.li, 'isolate_astra_process', True)} "
        f"worker_restart={getattr(cfg.li, 'astra_worker_restart_interval', 128)} "
        f"release_torch_cache={getattr(cfg.li, 'release_torch_cache_before_astra', True)} "
        f"clear_astra={getattr(cfg.li, 'clear_astra_after_batch', True)} "
        f"cleanup_interval={getattr(cfg.li, 'astra_cleanup_interval', 64)} "
        f"trim_host_heap={getattr(cfg.li, 'trim_host_memory_after_astra', True)} "
        f"host_trim_interval={getattr(cfg.li, 'host_trim_interval', 256)}",
        flush=True,
    )
    clean_train = make_dataset(cfg.data, cfg.seed, "clean", "train")
    metal_train = make_dataset(cfg.data, cfg.seed, "metal", "train")
    clean_val = make_dataset(cfg.data, cfg.seed, "clean", "val")
    metal_val = make_dataset(cfg.data, cfg.seed, "metal", "val")
    args_loader = dict(
        batch_size=cfg.train_joint.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.data.num_workers > 0,
        drop_last=True,
    )
    clean_loader = DataLoader(clean_train, shuffle=True, **args_loader)
    metal_loader = DataLoader(metal_train, shuffle=True, **args_loader)
    val_args = dict(batch_size=1, num_workers=max(0, cfg.data.num_workers // 2), drop_last=True)
    clean_val_loader = DataLoader(clean_val, **val_args)
    metal_val_loader = DataLoader(metal_val, **val_args)

    vae, coarse_model, diffusion = build_joint_components(cfg, device)
    refiner = build_refiner(cfg).to(device)
    parameter_groups = [
        {
            "name": "ldm",
            "params": coarse_model.parameters(),
            "lr": float(cfg.train_joint.ldm_learning_rate),
        }
    ]
    parameter_groups.append(
        {
            "name": "refiner",
            "params": refiner.parameters(),
            "lr": float(cfg.train_joint.refiner_learning_rate),
        }
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=cfg.train_joint.weight_decay,
    )
    ldm_ema = EMA(coarse_model, cfg.train_joint.ema_decay)
    refiner_ema = EMA(refiner, cfg.train_joint.ema_decay)
    start_epoch, global_step, micro_step, best = 0, 0, 0, float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("refiner_version") != REFINER_VERSION:
            raise ValueError(
                f"Checkpoint must be {REFINER_VERSION}; "
                f"got {payload.get('refiner_version')!r}"
            )
        coarse_model.load_state_dict(payload["ldm_model"])
        if not payload.get("refiner_enabled", True):
            raise ValueError("Resuming joint training requires a checkpoint with Refiner weights")
        refiner.load_state_dict(payload["refiner_model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("ldm_ema"):
            ldm_ema.shadow = {
                key: value.to(device) for key, value in payload["ldm_ema"].items()
            }
        if payload.get("refiner_ema"):
            refiner_ema.shadow = {
                key: value.to(device) for key, value in payload["refiner_ema"].items()
            }
        start_epoch = payload.get("epoch", -1) + 1
        global_step = payload.get("global_step", 0)
        micro_step = payload.get(
            "micro_step", global_step * int(cfg.train_joint.grad_accum)
        )
        best = payload.get("best", best)
        del payload
        print(
            f"resumed epoch={start_epoch} global_step={global_step} "
            f"{memory_status(device)} {astra_worker_status()}",
            flush=True,
        )

    output = Path(cfg.train_joint.checkpoint)
    last_path = output.with_name(output.name.replace("_best.pt", "_last.pt"))
    grad_accum = int(cfg.train_joint.grad_accum)
    optimizer.zero_grad(set_to_none=True)
    metal_iter = cycle(metal_loader)
    stop = False
    for epoch in range(start_epoch, cfg.train_joint.epochs):
        ldm_trainable = epoch >= int(cfg.train_joint.refiner_warmup_epochs)
        coarse_model.requires_grad_(ldm_trainable)
        coarse_model.train(ldm_trainable)
        refiner.train()
        phase = "joint" if ldm_trainable else "refiner warmup"
        bar = tqdm(
            clean_loader,
            desc=f"Hybrid512 {phase} epoch {epoch + 1}/{cfg.train_joint.epochs}",
        )
        last_metal_batch = None
        for batch_index, clean_batch in enumerate(bar):
            clean = clean_batch["image"].to(device, non_blocking=True)
            metal_batch = next(metal_iter)
            last_metal_batch = metal_batch
            metal_image = metal_batch["image"].to(device, non_blocking=True)
            metal_mask = metal_batch["metal_mask"].to(device, non_blocking=True)
            metal_li = metal_batch.get("li")
            if metal_li is not None:
                metal_li = metal_li.to(device, non_blocking=True)
            with autocast_context(device, dtype):
                loss, metrics = refiner_loss(
                    refiner, vae, coarse_model, diffusion, clean, metal_image,
                    metal_mask, cfg, micro_step, metal_li,
                )
                scaled = loss / grad_accum
            scaled.backward()
            micro_step += 1
            did_step = (batch_index + 1) % grad_accum == 0
            if did_step:
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
                if ldm_trainable:
                    torch.nn.utils.clip_grad_norm_(coarse_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ldm_ema.update(coarse_model)
                refiner_ema.update(refiner)
                global_step += 1
            if global_step % int(cfg.train_joint.log_every) == 0:
                bar.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    noise=f"{float(metrics['diffusion_noise']):.4f}",
                    residual=f"{float(metrics['residual']):.4f}",
                )
            memory_every = int(
                getattr(cfg.train_joint, "memory_log_every_steps", 0)
            )
            if (
                did_step
                and memory_every > 0
                and global_step % memory_every == 0
            ):
                print(
                    f"memory step={global_step} {memory_status(device)} "
                    f"{astra_worker_status()}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(
                        cuda_device_index(device)
                    )
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        ema_ldm = build_coarse_model(cfg, device)
        ema_refiner = build_refiner(cfg).to(device)
        ldm_ema.copy_to(ema_ldm)
        refiner_ema.copy_to(ema_refiner)
        ema_ldm.eval()
        ema_refiner.eval()
        val_batches = 1 if args.max_steps is not None else 10
        val_loss, val_metrics = validate(
            ema_refiner, vae, ema_ldm, diffusion,
            clean_val_loader, metal_val_loader, device, dtype, cfg, val_batches,
        )
        payload = {
            "ldm_model": coarse_model.state_dict(),
            "ldm_ema": ldm_ema.state_dict(),
            "refiner_model": refiner.state_dict(),
            "refiner_ema": refiner_ema.state_dict(),
            "refiner_enabled": True,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "micro_step": micro_step,
            "best": min(best, val_loss),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "refiner_version": REFINER_VERSION,
            "image_size": 512,
            "coarse_size": int(cfg.data.coarse_size),
            "vae_checkpoint": str(cfg.vae.checkpoint),
            "ldm_initialization": str(cfg.train_joint.ldm_initialization),
            "initial_coarse_checkpoint": str(cfg.diffusion.coarse_checkpoint),
            "vae_frozen": True,
            "astra_cleanup": "per_batch_v1",
        }
        atomic_torch_save(payload, last_path)
        if val_loss < best:
            best = val_loss
            payload["best"] = best
            atomic_torch_save(payload, output)
        print(
            f"epoch={epoch + 1} val={val_loss:.6f} best={best:.6f} "
            f"noise={val_metrics.get('diffusion_noise', float('nan')):.6f} "
            f"image={val_metrics.get('image', float('nan')):.6f} "
            f"projection={val_metrics.get('projection', float('nan')):.6f}"
        )
        if (
            last_metal_batch is not None
            and int(cfg.refiner.preview_every_epochs) > 0
            and (epoch + 1) % int(cfg.refiner.preview_every_epochs) == 0
        ):
            path = save_preview(
                ema_refiner, vae, ema_ldm, diffusion, last_metal_batch,
                cfg, device, dtype, epoch, global_step,
            )
            print(f"epoch preview saved: {path}")
        del payload, ema_ldm, ema_refiner
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if stop:
            break


if __name__ == "__main__":
    main()
