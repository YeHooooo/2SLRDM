from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import metal_artifact_masks


DATE_SUFFIX = re.compile(r"_\d{4}-\d{2}-\d{2}$")
SLICE_NUMBER = re.compile(r"_slice_(\d+)\.npy$")


def patient_id(case_name: str) -> str:
    return DATE_SUFFIX.sub("", case_name)


def stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64 - 1)


def split_is_val(case_name: str, val_fraction: float, seed: int) -> bool:
    return stable_fraction(patient_id(case_name), seed) < val_fraction


def slice_key(path: Path) -> tuple[int, str]:
    match = SLICE_NUMBER.search(path.name)
    return (int(match.group(1)) if match else 0, path.name)


@dataclass(frozen=True)
class SliceEntry:
    files: tuple[Path, ...]
    center: Path
    case: str


def discover_entries(
    root: str | Path,
    split: str,
    context_slices: int,
    stride: int,
    val_fraction: float,
    seed: int,
) -> list[SliceEntry]:
    if context_slices < 1 or context_slices % 2 != 1:
        raise ValueError("context_slices must be a positive odd integer")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    entries: list[SliceEntry] = []
    radius = context_slices // 2
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        is_val = split_is_val(case_dir.name, val_fraction, seed)
        if (split == "val") != is_val:
            continue
        files = sorted(case_dir.rglob("*.npy"), key=slice_key)
        for i in range(0, len(files), max(1, stride)):
            neighbors = tuple(files[min(max(j, 0), len(files) - 1)] for j in range(i - radius, i + radius + 1))
            entries.append(SliceEntry(neighbors, files[i], case_dir.name))
    if not entries:
        raise RuntimeError(f"No {split} .npy slices discovered under {root}")
    return entries


class CTSliceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        context_slices: int,
        image_size: int,
        stride: int,
        val_fraction: float,
        seed: int,
        hu_min: float,
        hu_max: float,
        with_mask: bool = False,
        metal_threshold: float = 2500.0,
        mask_dilate_kernel: int = 21,
        li_dilate_kernel: int = 15,
        streak_kernel: int = 31,
        streak_quantile: float = 0.975,
        min_body_hu: float = -850.0,
        li_root: str | Path | None = None,
        li_strict: bool = True,
        defer_crop: bool = False,
    ):
        self.root = Path(root)
        self.entries = discover_entries(self.root, split, context_slices, stride, val_fraction, seed)
        self.split = split
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.with_mask = with_mask
        self.li_root = Path(li_root) if li_root else None
        self.li_strict = bool(li_strict)
        self.defer_crop = bool(defer_crop)
        self.metal_threshold = float(metal_threshold)
        self.mask_args = dict(
            threshold=metal_threshold,
            mask_dilate_kernel=mask_dilate_kernel,
            li_dilate_kernel=li_dilate_kernel,
            streak_kernel=streak_kernel,
            streak_quantile=streak_quantile,
            min_body_hu=min_body_hu,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self.hu_min, self.hu_max)
        return ((x - self.hu_min) / (self.hu_max - self.hu_min) * 2.0 - 1.0).astype(np.float32)

    def _resize(self, stack: np.ndarray, mask: np.ndarray | None):
        from scipy import ndimage

        _, h, w = stack.shape
        size = self.image_size
        if (h, w) != (size, size):
            stack = ndimage.zoom(stack, (1.0, size / h, size / w), order=1, prefilter=False)
            if mask is not None:
                mask = ndimage.zoom(mask, (1.0, size / h, size / w), order=1, prefilter=False)
        return np.ascontiguousarray(stack), None if mask is None else np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        raw = np.stack([np.load(path).astype(np.float32) for path in entry.files])
        center_raw = raw[len(entry.files) // 2]
        mask_pack = None
        if self.with_mask:
            artifact_mask, li_mask = metal_artifact_masks(center_raw, **self.mask_args)
            metal_stack = (raw > self.metal_threshold).astype(np.float32)
            mask_pack = np.concatenate(
                [artifact_mask[None], li_mask[None], metal_stack], axis=0
            )
        normalized = self.normalize(raw)
        li_stack = None
        li_root = getattr(self, "li_root", None)
        if li_root is not None:
            li_paths = [li_root / path.relative_to(self.root) for path in entry.files]
            if all(path.exists() for path in li_paths):
                li_stack = self.normalize(np.stack([np.load(path).astype(np.float32) for path in li_paths]))
                combined = np.concatenate([normalized, li_stack], axis=0)
                if self.defer_crop:
                    normalized, li_stack = np.split(np.ascontiguousarray(combined), 2, axis=0)
                    mask_pack = None if mask_pack is None else np.ascontiguousarray(mask_pack)
                else:
                    combined, mask_pack = self._resize(combined, mask_pack)
                    normalized, li_stack = np.split(combined, 2, axis=0)
            elif getattr(self, "li_strict", True):
                missing = next(path for path in li_paths if not path.exists())
                raise FileNotFoundError(f"Missing paired LI slice: {missing}")
        if li_stack is None:
            if self.defer_crop:
                stack = np.ascontiguousarray(normalized)
                mask_pack = None if mask_pack is None else np.ascontiguousarray(mask_pack)
            else:
                stack, mask_pack = self._resize(normalized, mask_pack)
        else:
            stack = normalized
        item = {
            "image": torch.from_numpy(stack),
            "path": str(entry.center),
            "case": entry.case,
        }
        if mask_pack is not None:
            item["mask"] = torch.from_numpy(mask_pack[0:1])
            item["li_mask"] = torch.from_numpy(mask_pack[1:2])
            item["metal_mask"] = torch.from_numpy(mask_pack[2:])
        if li_stack is not None:
            item["li"] = torch.from_numpy(li_stack)
        return item


def dataset_from_config(cfg, domain: str, split: str, seed: int = 2026) -> CTSliceDataset:
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
    )
