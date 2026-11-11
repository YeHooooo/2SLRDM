from __future__ import annotations

import ctypes
import gc
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def amp_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    dx = (pred_dx - target_dx).abs()
    dy = (pred_dy - target_dy).abs()
    if weight is not None:
        dx = dx * weight[..., :, 1:]
        dy = dy * weight[..., 1:, :]
        denom = weight[..., :, 1:].sum() + weight[..., 1:, :].sum()
        return (dx.sum() + dy.sum()) / denom.clamp_min(1.0)
    return 0.5 * (dx.mean() + dy.mean())


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[1] == 1 and pred.shape[1] != 1:
        weight = weight.expand(-1, pred.shape[1], -1, -1)
    return ((pred - target).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def center_crop_or_pad(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    h, w = x.shape[-2:]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h or pad_w:
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="replicate")
    h, w = x.shape[-2:]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[..., top : top + target_h, left : left + target_w]


def cycle(loader: Iterable):
    while True:
        yield from loader


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        state.update(self.shadow)
        model.load_state_dict(state)


def atomic_torch_save(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _current_process_rss_gib() -> float | None:
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
        except (OSError, ValueError, IndexError):
            pass
    return None


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}")
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


_malloc_trim = None
_malloc_trim_checked = False


def trim_process_heap() -> bool:
    global _malloc_trim, _malloc_trim_checked
    gc.collect()
    if os.name != "posix":
        return False
    if not _malloc_trim_checked:
        _malloc_trim_checked = True
        try:
            function = ctypes.CDLL(None).malloc_trim
            function.argtypes = [ctypes.c_size_t]
            function.restype = ctypes.c_int
            _malloc_trim = function
        except (AttributeError, OSError):
            _malloc_trim = None
    if _malloc_trim is None:
        return False
    return bool(_malloc_trim(0))


def memory_status(device: torch.device) -> str:
    fields = []
    rss = _current_process_rss_gib()
    if rss is not None:
        fields.append(f"rss={rss:.2f}GiB")
    if device.type == "cuda":
        cuda_index = cuda_device_index(device)
        gib = 1024.0 ** 3
        allocated = torch.cuda.memory_allocated(cuda_index) / gib
        reserved = torch.cuda.memory_reserved(cuda_index) / gib
        peak = torch.cuda.max_memory_allocated(cuda_index) / gib
        free, total = torch.cuda.mem_get_info(cuda_index)
        fields.extend(
            (
                f"torch_alloc={allocated:.2f}GiB",
                f"torch_reserved={reserved:.2f}GiB",
                f"torch_peak={peak:.2f}GiB",
                f"cuda_global_used={(total - free) / gib:.2f}GiB",
            )
        )
    return " ".join(fields) or "memory metrics unavailable"


def cosine_warmup(step: int, total: int, warmup: int = 500) -> float:
    if step < warmup:
        return max(1e-4, step / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
