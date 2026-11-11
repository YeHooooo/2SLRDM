from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from marldm.config import load_config, to_plain_dict
from marldm.data import CTSliceDataset
from marldm.models.vae import build_vae
from marldm.utils import (
    amp_dtype,
    atomic_torch_save,
    autocast_context,
    gradient_l1,
    cuda_device_index,
    memory_status,
    resolve_device,
    seed_everything,
)


VAE_TRAINING_VERSION = "model_test10_vae512_v1"


def make_dataset(cfg, seed: int, domain: str, split: str) -> CTSliceDataset:
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
        with_mask=False,
    )


def reconstruction_loss(model, image, cfg, *, sample: bool):
    reconstruction, posterior = model(image, sample=sample)
    pixel = (reconstruction.float() - image.float()).abs().mean()
    edge = gradient_l1(reconstruction.float(), image.float())
    kl = posterior.kl()
    total = (
        float(cfg.train_vae.pixel_weight) * pixel
        + float(cfg.train_vae.edge_weight) * edge
        + float(cfg.vae.kl_weight) * kl
    )
    return total, reconstruction, {
        "pixel": pixel.detach(),
        "edge": edge.detach(),
        "kl": kl.detach(),
    }


@torch.no_grad()
def validate(model, loader, device, dtype, cfg, max_batches: int) -> tuple[float, dict]:
    model.eval()
    total = 0.0
    sums = {"pixel": 0.0, "edge": 0.0, "kl": 0.0}
    count = 0
    preview = None
    for index, batch in enumerate(loader):
        if index >= max(1, int(max_batches)):
            break
        image = batch["image"].to(device, non_blocking=True)
        with autocast_context(device, dtype):
            loss, reconstruction, metrics = reconstruction_loss(
                model, image, cfg, sample=False
            )
        total += float(loss)
        for key, value in metrics.items():
            sums[key] += float(value)
        count += 1
        if preview is None:
            preview = (
                image[0].float().cpu().numpy(),
                reconstruction[0].float().cpu().numpy(),
            )
    model.train()
    count = max(1, count)
    return total / count, {
        **{key: value / count for key, value in sums.items()},
        "preview": preview,
    }


def normalized_to_uint8(image, hu_min, hu_max, window) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image_hu = (image + 1.0) * 0.5 * (float(hu_max) - float(hu_min)) + float(hu_min)
    low, high = map(float, window)
    scaled = (image_hu - low) / (high - low)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_preview(preview, output_dir: Path, epoch: int, global_step: int, cfg) -> Path | None:
    if preview is None:
        return None
    source, reconstruction = preview
    center = source.shape[0] // 2
    source_png = normalized_to_uint8(
        source[center], cfg.data.hu_min, cfg.data.hu_max, cfg.train_vae.preview_window
    )
    reconstruction_png = normalized_to_uint8(
        reconstruction[center],
        cfg.data.hu_min,
        cfg.data.hu_max,
        cfg.train_vae.preview_window,
    )
    height, width = source_png.shape
    header = 24
    gap = 8
    canvas = Image.new("L", (width * 2 + gap, height + header), 0)
    canvas.paste(Image.fromarray(source_png), (0, header))
    canvas.paste(Image.fromarray(reconstruction_png), (width + gap, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), "Input 512", fill=255)
    draw.text((width + gap + 4, 6), "VAE512 reconstruction", fill=255)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"vae512_epoch_{epoch + 1:04d}_step_{global_step:08d}.png"
    canvas.save(path, format="PNG", compress_level=6)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dedicated 512x512 2.5D CT VAE for Model_test10"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--image-size", type=int, default=512)
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--allow-overwrite-legacy-vae",
        action="store_true",
        help="Allow --output to overwrite cfg.vae.checkpoint (unsafe by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.data.image_size = int(args.image_size)
    if cfg.data.image_size != 512:
        raise ValueError("This experiment requires --image-size 512")
    if args.with_metal_root is not None:
        cfg.data.with_metal = args.with_metal_root
    if args.without_metal_root is not None:
        cfg.data.without_metal = args.without_metal_root
    if args.epochs is not None:
        cfg.train_vae.epochs = max(1, int(args.epochs))
    if args.batch_size is not None:
        cfg.train_vae.batch_size = max(1, int(args.batch_size))
    if args.num_workers is not None:
        cfg.data.num_workers = max(0, int(args.num_workers))
    if args.preview_dir is not None:
        cfg.train_vae.preview_dir = args.preview_dir

    output = Path(args.output or cfg.train_vae.checkpoint)
    legacy_vae = Path(cfg.vae.checkpoint)
    if (
        output.resolve() == legacy_vae.resolve()
        and not args.allow_overwrite_legacy_vae
    ):
        raise ValueError(
            f"Refusing to overwrite the current 256 VAE: {legacy_vae}. "
            "Use the dedicated default checkpoints/vae512_best.pt path."
        )
    last_path = output.with_name(
        output.name.replace("_best.pt", "_last.pt")
        if "_best.pt" in output.name
        else output.stem + "_last.pt"
    )

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = amp_dtype(cfg.train_vae.amp) if device.type == "cuda" else None
    print(
        f"VAE training version : {VAE_TRAINING_VERSION}\n"
        f"Image size           : {cfg.data.image_size}x{cfg.data.image_size}\n"
        f"Latent size          : "
        f"{cfg.data.image_size // (2 ** (len(cfg.vae.channel_mult) - 1))}x"
        f"{cfg.data.image_size // (2 ** (len(cfg.vae.channel_mult) - 1))}\n"
        f"Best checkpoint      : {output.resolve()}\n"
        f"Last checkpoint      : {last_path.resolve()}\n"
        f"Device / AMP         : {device} / {cfg.train_vae.amp}",
        flush=True,
    )

    train_dataset = ConcatDataset(
        [
            make_dataset(cfg.data, cfg.seed, "metal", "train"),
            make_dataset(cfg.data, cfg.seed, "clean", "train"),
        ]
    )
    validation_dataset = ConcatDataset(
        [
            make_dataset(cfg.data, cfg.seed, "metal", "val"),
            make_dataset(cfg.data, cfg.seed, "clean", "val"),
        ]
    )
    loader_args = {
        "batch_size": int(cfg.train_vae.batch_size),
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(cfg.data.num_workers) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=int(cfg.train_vae.batch_size),
        num_workers=max(0, int(cfg.data.num_workers) // 2),
        pin_memory=device.type == "cuda",
    )

    model = build_vae(cfg.vae).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train_vae.learning_rate),
        weight_decay=float(cfg.train_vae.weight_decay),
    )
    start_epoch = 0
    global_step = 0
    best = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_size = int(payload.get("image_size", 0))
        if checkpoint_size not in {0, 512}:
            raise ValueError(
                f"Resume checkpoint was trained at {checkpoint_size}, not 512"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        best = float(payload.get("best", best))
        del payload
        print(
            f"Resumed epoch={start_epoch} step={global_step} "
            f"{memory_status(device)}",
            flush=True,
        )

    grad_accum = max(1, int(cfg.train_vae.grad_accum))
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, int(cfg.train_vae.epochs)):
        model.train()
        bar = tqdm(
            train_loader,
            desc=f"VAE512 epoch {epoch + 1}/{cfg.train_vae.epochs}",
        )
        for batch_index, batch in enumerate(bar):
            image = batch["image"].to(device, non_blocking=True)
            with autocast_context(device, dtype):
                loss, _, metrics = reconstruction_loss(
                    model, image, cfg, sample=True
                )
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            did_step = (batch_index + 1) % grad_accum == 0
            if did_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if did_step and global_step % int(cfg.train_vae.log_every) == 0:
                bar.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    pixel=f"{float(metrics['pixel']):.4f}",
                    edge=f"{float(metrics['edge']):.4f}",
                    kl=f"{float(metrics['kl']):.3f}",
                )
            if (
                did_step
                and int(cfg.train_vae.memory_log_every_steps) > 0
                and global_step % int(cfg.train_vae.memory_log_every_steps) == 0
            ):
                print(
                    f"memory step={global_step} {memory_status(device)}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        validation_batches = (
            1 if args.max_steps is not None else int(cfg.train_vae.val_max_batches)
        )
        validation_loss, validation_metrics = validate(
            model,
            validation_loader,
            device,
            dtype,
            cfg,
            validation_batches,
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best": min(best, validation_loss),
            "val_loss": validation_loss,
            "val_metrics": {
                key: value
                for key, value in validation_metrics.items()
                if key != "preview"
            },
            "training_version": VAE_TRAINING_VERSION,
            "image_size": 512,
            "latent_size": 512 // model.downsample_factor,
            "context_slices": int(cfg.data.context_slices),
            "vae_config": to_plain_dict(cfg.vae),
        }
        atomic_torch_save(payload, last_path)
        if validation_loss < best:
            best = validation_loss
            payload["best"] = best
            atomic_torch_save(payload, output)
        preview_path = save_preview(
            validation_metrics.get("preview"),
            Path(cfg.train_vae.preview_dir),
            epoch,
            global_step,
            cfg,
        )
        print(
            f"epoch={epoch + 1} val={validation_loss:.6f} best={best:.6f} "
            f"pixel={validation_metrics['pixel']:.6f} "
            f"edge={validation_metrics['edge']:.6f} "
            f"preview={preview_path}",
            flush=True,
        )
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if stop:
            break


if __name__ == "__main__":
    main()
