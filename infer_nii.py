from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from infer import process_stack
from marldm.config import load_config
from marldm.hybrid import (
    build_joint_components,
    build_refiner,
    load_joint_checkpoint,
)
from marldm.utils import amp_dtype, resolve_device, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_slice_axis(shape: tuple[int, ...], requested: int | None) -> int:
    if len(shape) != 3:
        raise ValueError(f"Model_test10 expects a 3D NIfTI volume, got shape {shape}")
    if requested is not None:
        axis = int(requested)
        if axis < 0:
            axis += len(shape)
        if axis not in range(len(shape)):
            raise ValueError(f"Invalid slice axis {requested} for shape {shape}")
    else:
        candidates = [
            axis
            for axis in range(3)
            if tuple(shape[index] for index in range(3) if index != axis) == (512, 512)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely infer a 512x512 slice axis from shape {shape}; "
                "pass --slice-axis explicitly"
            )
        axis = candidates[0]
    in_plane = tuple(shape[index] for index in range(3) if index != axis)
    if in_plane != (512, 512):
        raise ValueError(
            f"Selected axis {axis} gives in-plane shape {in_plane}, expected (512, 512)"
        )
    return axis


def default_output_path(input_path: Path, output_dir: Path) -> Path:
    name = input_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else input_path.stem
    return output_dir / f"{stem}_model_test10_mar.nii.gz"


def neighboring_indices(index: int, count: int, context_slices: int) -> list[int]:
    radius = int(context_slices) // 2
    return [
        min(max(value, 0), count - 1)
        for value in range(index - radius, index + radius + 1)
    ]


def save_nifti_like(source_image, output_volume: np.ndarray, output_path: Path) -> None:
    import nibabel as nib

    header = source_image.header.copy()
    header.set_data_dtype(np.float32)
    output_image = source_image.__class__(
        np.asarray(output_volume, dtype=np.float32),
        source_image.affine.copy(),
        header=header,
    )
    qform, qform_code = source_image.get_qform(coded=True)
    sform, sform_code = source_image.get_sform(coded=True)
    if qform is not None:
        output_image.set_qform(qform, int(qform_code))
    if sform is not None:
        output_image.set_sform(sform, int(sform_code))
    output_image.header["cal_min"] = float(np.nanmin(output_volume))
    output_image.header["cal_max"] = float(np.nanmax(output_volume))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        output_path.name.removesuffix(".nii.gz") + ".tmp.nii.gz"
    )
    nib.save(output_image, temporary)
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Model_test10 to a 3D 512x512 NIfTI CT volume"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to train_joint.checkpoint in the selected configuration",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "val_nii",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--slice-axis", type=int, default=None)
    parser.add_argument(
        "--slice-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Optional zero-based half-open range [START, END); other slices remain unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect NIfTI geometry and output path without loading the model",
    )
    parser.add_argument(
        "--remove-metal",
        action="store_true",
        help="Inference-only: replace the metal core with softly blended LI",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import nibabel as nib
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "NIfTI support requires nibabel. Install it with: pip install nibabel"
        ) from exc

    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input NIfTI does not exist: {args.input}")
    source_image = nib.load(args.input)
    shape = tuple(map(int, source_image.shape))
    slice_axis = resolve_slice_axis(shape, args.slice_axis)
    output_path = args.output or default_output_path(args.input, args.output_dir)
    slice_count = shape[slice_axis]

    start, end = 0, slice_count
    if args.slice_range is not None:
        start, end = map(int, args.slice_range)
        if not (0 <= start < end <= slice_count):
            raise ValueError(
                f"--slice-range must satisfy 0 <= START < END <= {slice_count}"
            )

    print(f"Input       : {args.input.resolve()}")
    print(f"Shape       : {shape}")
    print(f"Voxel size  : {tuple(float(v) for v in source_image.header.get_zooms()[:3])}")
    print(f"Axis codes  : {nib.aff2axcodes(source_image.affine)}")
    print(f"Slice axis  : {slice_axis} ({slice_count} slices)")
    print(f"Process     : [{start}, {end})")
    print(f"Output      : {output_path.resolve()}")
    if args.dry_run:
        return
    if not args.config.is_file():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or Path(cfg.train_joint.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Joint checkpoint does not exist: {checkpoint}")
    if args.remove_metal:
        cfg.metal_removal.enabled = True
    if args.vae_checkpoint is not None:
        cfg.vae.checkpoint = str(args.vae_checkpoint)
    if args.coarse_size is not None:
        cfg.data.coarse_size = int(args.coarse_size)
    if int(cfg.data.image_size) != 512:
        raise ValueError("Model_test10 NIfTI inference requires data.image_size=512")
    if int(cfg.data.context_slices) != 3:
        raise ValueError("Model_test10 NIfTI inference currently requires context_slices=3")

    volume = np.asanyarray(source_image.dataobj).astype(np.float32)
    slices = np.moveaxis(volume, slice_axis, 0)
    output_slices = slices.copy()
    metal_by_slice = np.any(
        slices > float(cfg.data.metal_threshold), axis=(1, 2)
    )
    context_required = np.zeros(slice_count, dtype=bool)
    for index in np.flatnonzero(metal_by_slice):
        for neighbor in neighboring_indices(
            int(index), slice_count, int(cfg.data.context_slices)
        ):
            context_required[neighbor] = True
    requested = np.zeros(slice_count, dtype=bool)
    requested[start:end] = True
    process_indices = np.flatnonzero(context_required & requested)
    print(
        f"Metal slices: {int(metal_by_slice.sum())}; "
        f"2.5D slices to process: {len(process_indices)}"
    )

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = amp_dtype(cfg.train_joint.amp) if device.type == "cuda" else None
    vae, coarse_model, diffusion = build_joint_components(
        cfg, device, initialize_ldm=False
    )
    refiner = build_refiner(cfg).to(device)
    load_joint_checkpoint(
        coarse_model, refiner, checkpoint, device, use_ema=True
    )
    coarse_model.eval().requires_grad_(False)
    refiner.eval().requires_grad_(False)

    for index in tqdm(process_indices, desc="Model_test10 NIfTI"):
        selected = neighboring_indices(
            int(index), slice_count, int(cfg.data.context_slices)
        )
        raw_stack = slices[selected]
        final, _, _, _, _ = process_stack(
            raw_stack,
            vae,
            coarse_model,
            diffusion,
            refiner,
            cfg,
            device,
            dtype,
        )
        output_slices[int(index)] = final

    output_volume = np.moveaxis(output_slices, 0, slice_axis)
    save_nifti_like(source_image, output_volume, output_path)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
