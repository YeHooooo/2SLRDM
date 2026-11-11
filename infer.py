from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from marldm.artifacts import expanded_sampling_mask, li_delta_hint
from marldm.config import load_config
from marldm.data import slice_key
from marldm.fusion import li_guided_metal_removal, near_metal_dark_rescue
from marldm.hybrid import (
    apply_refiner,
    build_joint_components,
    build_refiner,
    load_joint_checkpoint,
    sample_coarse,
)
from marldm.masks import metal_fill_mask, residual_artifact_masks
from marldm.projection_li import projection_domain_li
from marldm.utils import amp_dtype, autocast_context, resolve_device, seed_everything


def normalize(x, lo, hi):
    return (np.clip(x, lo, hi) - lo) / (hi - lo) * 2.0 - 1.0


def denormalize(x, lo, hi):
    return (x + 1.0) * 0.5 * (hi - lo) + lo


def window_to_uint8(image_hu, window_lo, window_hi):
    if float(window_hi) <= float(window_lo):
        raise ValueError("PNG window upper bound must be greater than lower bound")
    scaled = (np.asarray(image_hu, np.float32) - window_lo) / (window_hi - window_lo)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_png_comparison(input_hu, coarse_hu, final_hu, output_npy, window_lo, window_hi):
    output_npy = Path(output_npy)
    path = output_npy.with_suffix("").with_name(output_npy.stem + "_comparison.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        window_to_uint8(input_hu, window_lo, window_hi),
        window_to_uint8(coarse_hu, window_lo, window_hi),
        window_to_uint8(final_hu, window_lo, window_hi),
    ]
    if any(panel.shape != panels[0].shape for panel in panels):
        raise ValueError("Input, coarse and final panels must have the same shape")
    height, width = panels[0].shape
    header, gap = 24, 8
    canvas = Image.new("L", (width * 3 + gap * 2, height + header), color=0)
    labels = ("Input 512", "Coarse LDM", "Hybrid final")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * (width + gap)
        canvas.paste(Image.fromarray(panel), (left, header))
        draw.text((left + 5, 5), label, fill=255)
    canvas.save(path)
    return path


def neighbors(files, index, count):
    radius = count // 2
    return [
        files[min(max(value, 0), len(files) - 1)]
        for value in range(index - radius, index + radius + 1)
    ]


def case_matches_patient(case_name, selectors):
    case_name = str(case_name).upper()
    return any(
        case_name == str(selector).upper()
        or case_name.startswith(str(selector).upper() + "_")
        for selector in selectors
    )


@torch.no_grad()
def artifact_conditions(source, metal, cfg):
    li = projection_domain_li(source, metal, cfg.data, cfg.li)
    confidence, support, _ = residual_artifact_masks(
        source,
        li,
        metal,
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
        metal,
        cfg.near_metal.condition_fill_dilate_kernel,
        cfg.near_metal.condition_fill_blur_kernel,
    )
    raw_condition = source * (1.0 - fill) + li * fill
    li_delta = li_delta_hint(
        raw_condition, li, cfg.artifact_transfer.li_delta_scale_hu,
        cfg.data.hu_min, cfg.data.hu_max,
    )
    sampling = expanded_sampling_mask(
        support,
        cfg.diffusion.sampling_mask_threshold,
        cfg.diffusion.sampling_mask_dilate_kernel,
        cfg.diffusion.sampling_mask_blur_kernel,
    )
    return li, confidence, support, raw_condition, li_delta, sampling


@torch.no_grad()
def process_stack(raw, vae, coarse_model, diffusion, refiner, cfg, device, dtype):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != int(cfg.data.context_slices):
        raise ValueError(
            f"Expected {cfg.data.context_slices} slices in CHW layout, got {raw.shape}"
        )
    center_index = raw.shape[0] // 2
    center = raw[center_index]
    if center.shape != (512, 512):
        raise ValueError(f"Model_test10 requires 512x512 inputs, got {center.shape}")
    metal_np = (raw > cfg.data.metal_threshold).astype(np.float32)
    if not metal_np.any():
        zeros = np.zeros_like(center, np.float32)
        return center.copy(), center.copy(), zeros, zeros, center.copy()

    source = torch.from_numpy(
        normalize(raw, cfg.data.hu_min, cfg.data.hu_max)
    )[None].to(device)
    metal = torch.from_numpy(metal_np)[None].to(device)
    li, confidence, support, raw_condition, li_delta, sampling = artifact_conditions(
        source, metal, cfg
    )
    with autocast_context(device, dtype):
        coarse = sample_coarse(
            vae, coarse_model, diffusion, raw_condition, li, confidence,
            sampling, metal, li_delta, cfg, source_anchor=source,
        )
        final, _, gate, _ = apply_refiner(
            refiner, source, coarse, li, confidence, support, metal, cfg
        )
        if bool(getattr(cfg.dark_rescue, "enabled", False)):
            final = near_metal_dark_rescue(
                source,
                li,
                final,
                confidence,
                metal,
                hu_min=cfg.data.hu_min,
                hu_max=cfg.data.hu_max,
                outer_kernel=cfg.dark_rescue.outer_kernel,
                ring_blur_kernel=cfg.dark_rescue.ring_blur_kernel,
                residual_threshold_hu=cfg.dark_rescue.residual_threshold_hu,
                residual_softness_hu=cfg.dark_rescue.residual_softness_hu,
                output_gap_threshold_hu=cfg.dark_rescue.output_gap_threshold_hu,
                confidence_threshold=cfg.dark_rescue.confidence_threshold,
                confidence_transition=cfg.dark_rescue.confidence_transition,
                confidence_gamma=cfg.dark_rescue.confidence_gamma,
                output_dark_threshold_hu=cfg.dark_rescue.output_dark_threshold_hu,
                output_dark_softness_hu=cfg.dark_rescue.output_dark_softness_hu,
                min_body_hu=cfg.dark_rescue.min_body_hu,
                li_target_min_hu=cfg.dark_rescue.li_target_min_hu,
                li_target_softness_hu=cfg.dark_rescue.li_target_softness_hu,
                max_correction_hu=cfg.dark_rescue.max_correction_hu,
                blend_gain=cfg.dark_rescue.blend_gain,
                mask_blur_kernel=cfg.dark_rescue.mask_blur_kernel,
                rescue_passes=cfg.dark_rescue.rescue_passes,
            )
        if bool(getattr(cfg.metal_removal, "enabled", False)):
            final = li_guided_metal_removal(
                final,
                li,
                metal,
                dilate_kernel=cfg.metal_removal.dilate_kernel,
                blur_kernel=cfg.metal_removal.blur_kernel,
                transition_gain=cfg.metal_removal.transition_gain,
            )
    coarse_hu = denormalize(
        coarse[0, center_index].float().cpu().numpy(), cfg.data.hu_min, cfg.data.hu_max
    ).astype(np.float32)
    final_hu = denormalize(
        final[0, center_index].float().cpu().numpy(), cfg.data.hu_min, cfg.data.hu_max
    ).astype(np.float32)
    li_hu = denormalize(
        li[0, center_index].float().cpu().numpy(), cfg.data.hu_min, cfg.data.hu_max
    ).astype(np.float32)
    metal_center = metal_np[center_index] > 0.5
    coarse_hu[metal_center] = center[metal_center]
    if not bool(getattr(cfg.metal_removal, "enabled", False)):
        final_hu[metal_center] = center[metal_center]
    below_normalization_range = center < float(cfg.data.hu_min)
    coarse_hu[below_normalization_range] = center[below_normalization_range]
    final_hu[below_normalization_range] = center[below_normalization_range]
    li_hu[below_normalization_range] = center[below_normalization_range]
    return (
        final_hu,
        coarse_hu,
        confidence[0, 0].cpu().numpy().astype(np.float32),
        gate[0, 0].float().cpu().numpy().astype(np.float32),
        li_hu,
    )


@torch.no_grad()
def process_one(files, index, vae, coarse_model, diffusion, refiner, cfg, device, dtype):
    selected = neighbors(files, index, cfg.data.context_slices)
    raw = np.stack(
        [np.load(path, allow_pickle=False).astype(np.float32) for path in selected]
    )
    return process_stack(
        raw, vae, coarse_model, diffusion, refiner, cfg, device, dtype
    )


def main():
    parser = argparse.ArgumentParser(description="Apply Model_test10 near-metal-aware hybrid MAR")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None, help="Joint LDM/refiner checkpoint")
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
        "--sample-steps",
        type=int,
        default=None,
        help="Override diffusion.sample_steps for inference",
    )
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--save-coarse", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument(
        "--save-li",
        action="store_true",
        help="Save the runtime projection-domain LI center slice as NPY",
    )
    parser.add_argument(
        "--dark-rescue",
        action="store_true",
        help="Enable inference-only one-sided rescue for dark blocks next to metal",
    )
    parser.add_argument(
        "--remove-metal",
        action="store_true",
        help="Inference-only: replace the metal core with softly blended LI",
    )
    parser.add_argument("--png-window", nargs=2, type=float, default=None)
    parser.add_argument(
        "--timing-json",
        default=None,
        help="Write per-slice pipeline timing statistics to this JSON file",
    )
    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=5,
        help="Number of leading slices excluded from steady-state Time/slice",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N slices; 0 prints only the final summary",
    )
    parser.add_argument(
        "--patient",
        action="append",
        default=[],
        help=(
            "Only process this patient prefix or full case name. Repeat the "
            "argument to select multiple patients."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dark_rescue:
        cfg.dark_rescue.enabled = True
    if args.remove_metal:
        cfg.metal_removal.enabled = True
    if args.vae_checkpoint:
        cfg.vae.checkpoint = args.vae_checkpoint
    if args.coarse_size is not None:
        cfg.data.coarse_size = int(args.coarse_size)
    if args.sample_steps is not None:
        if int(args.sample_steps) < 1:
            raise ValueError("--sample-steps must be positive")
        cfg.diffusion.sample_steps = int(args.sample_steps)
    if int(cfg.data.image_size) != 512:
        raise ValueError("Model_test10 requires data.image_size=512")
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = amp_dtype(cfg.train_joint.amp) if device.type == "cuda" else None
    vae, coarse_model, diffusion = build_joint_components(
        cfg, device, initialize_ldm=False
    )
    refiner = build_refiner(cfg).to(device)
    load_joint_checkpoint(
        coarse_model,
        refiner,
        args.checkpoint or cfg.train_joint.checkpoint,
        device,
        use_ema=True,
    )
    coarse_model.eval().requires_grad_(False)
    refiner.eval().requires_grad_(False)
    window_lo, window_hi = map(
        float, args.png_window or cfg.refiner.preview_window
    )

    source = Path(args.input)
    if source.is_dir():
        files = sorted(source.rglob("*.npy"), key=lambda path: (str(path.parent), slice_key(path)))
        if args.patient:
            files = [
                path
                for path in files
                if case_matches_patient(path.parent.name, args.patient)
            ]
            if not files:
                raise FileNotFoundError(
                    f"No NPY files matched --patient {args.patient!r} under {source}"
                )
            selected_cases = sorted({path.parent.name for path in files})
            print(
                f"Selected {len(selected_cases)} cases / {len(files)} slices: "
                + ", ".join(selected_cases)
            )
        case_files = {
            parent: sorted(parent.glob("*.npy"), key=slice_key)
            for parent in sorted({path.parent for path in files}, key=str)
        }
        case_indices = {
            path: case_index
            for siblings in case_files.values()
            for case_index, path in enumerate(siblings)
        }
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        slice_times = []
        for index, path in enumerate(files):
            siblings = case_files[path.parent]
            sibling_index = case_indices[path]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            final, coarse, confidence, gate, li_hu = process_one(
                siblings, sibling_index, vae, coarse_model, diffusion, refiner,
                cfg, device, dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            slice_times.append(time.perf_counter() - started)
            output_npy = output_dir / path.relative_to(source)
            output_npy.parent.mkdir(parents=True, exist_ok=True)
            if args.save_npy or not args.save_png:
                np.save(output_npy, final)
            if args.save_coarse:
                np.save(output_npy.with_name(f"{path.stem}_coarse.npy"), coarse)
            if args.save_mask:
                np.save(output_npy.with_name(f"{path.stem}_confidence.npy"), confidence)
                np.save(output_npy.with_name(f"{path.stem}_refiner_gate.npy"), gate)
            if args.save_li:
                np.save(output_npy.with_name(f"{path.stem}_li.npy"), li_hu)
            if args.save_png:
                save_png_comparison(
                    np.load(path, allow_pickle=False), coarse, final,
                    output_npy, window_lo, window_hi,
                )
            progress_every = int(args.progress_every)
            if progress_every > 0 and (
                (index + 1) % progress_every == 0 or index + 1 == len(files)
            ):
                running = float(np.mean(slice_times))
                print(
                    f"[{index + 1}/{len(files)}] {path.name} "
                    f"pipeline_time={slice_times[-1]:.4f}s "
                    f"mean={running:.4f}s/slice",
                    flush=True,
                )
        warmup = min(max(0, int(args.timing_warmup)), max(0, len(slice_times) - 1))
        steady = slice_times[warmup:] if warmup < len(slice_times) else slice_times
        timing = {
            "num_slices": len(slice_times),
            "ddim_sample_steps": int(cfg.diffusion.sample_steps),
            "diffusion_train_timesteps": int(cfg.diffusion.timesteps),
            "start_timestep": int(cfg.diffusion.start_timestep),
            "sample_eta": float(cfg.diffusion.sample_eta),
            "dark_rescue_enabled": bool(cfg.dark_rescue.enabled),
            "refiner_enabled": True,
            "coarse_size": int(cfg.data.coarse_size),
            "timing_scope": (
                "NPY stack loading + projection-domain LI + condition maps + "
                "DDIM/LDM + Refiner + dark-rescue; excludes PNG/NPY output writing"
            ),
            "warmup_slices_excluded": warmup,
            "total_pipeline_seconds": float(np.sum(slice_times)),
            "mean_seconds_per_slice_all": float(np.mean(slice_times)),
            "mean_seconds_per_slice_steady_state": float(np.mean(steady)),
            "median_seconds_per_slice_steady_state": float(np.median(steady)),
            "p95_seconds_per_slice_steady_state": float(np.percentile(steady, 95)),
            "min_seconds_per_slice_steady_state": float(np.min(steady)),
            "max_seconds_per_slice_steady_state": float(np.max(steady)),
        }
        timing_path = (
            Path(args.timing_json)
            if args.timing_json
            else output_dir / "timing.json"
        )
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "Timing summary: "
            f"all={timing['mean_seconds_per_slice_all']:.6f}s/slice, "
            f"steady={timing['mean_seconds_per_slice_steady_state']:.6f}s/slice, "
            f"median={timing['median_seconds_per_slice_steady_state']:.6f}s/slice"
        )
        print(f"Saved timing: {timing_path}")
    else:
        files = sorted(source.parent.glob("*.npy"), key=slice_key)
        final, coarse, confidence, gate, li_hu = process_one(
            files, files.index(source), vae, coarse_model, diffusion,
            refiner, cfg, device, dtype,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_npy or not args.save_png:
            np.save(output, final)
        if args.save_coarse:
            np.save(output.with_name(output.stem + "_coarse.npy"), coarse)
        if args.save_mask:
            np.save(output.with_name(output.stem + "_confidence.npy"), confidence)
            np.save(output.with_name(output.stem + "_refiner_gate.npy"), gate)
        if args.save_li:
            np.save(output.with_name(output.stem + "_li.npy"), li_hu)
        if args.save_png:
            save_png_comparison(
                np.load(source, allow_pickle=False), coarse, final,
                output, window_lo, window_hi,
            )
        print(output)


if __name__ == "__main__":
    main()
