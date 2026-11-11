from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from marldm.config import load_config
from marldm.projection_li import projection_li_slice


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def valid_existing_output(source: Path, output: Path) -> bool:
    if not output.is_file():
        return False
    try:
        source_array = np.load(source, mmap_mode="r", allow_pickle=False)
        output_array = np.load(output, mmap_mode="r", allow_pickle=False)
        return source_array.ndim == 2 and output_array.shape == source_array.shape
    except (OSError, ValueError):
        return False


def compute_li(image_hu: np.ndarray, cfg, backend: str) -> tuple[np.ndarray, bool]:
    if image_hu.ndim != 2:
        raise ValueError(f"Expected a 2D CT slice, got shape {image_hu.shape}")
    image_hu = image_hu.astype(np.float32, copy=False)
    metal_mask = image_hu > float(cfg.data.metal_threshold)
    has_metal = bool(np.any(metal_mask))
    if not has_metal:
        return np.ascontiguousarray(image_hu), False

    result = projection_li_slice(
        image_hu,
        metal_mask,
        backend=backend,
        num_angles=int(cfg.li.num_angles),
        detector_spacing=float(cfg.li.detector_spacing),
        detector_count=int(cfg.li.detector_count),
        trace_threshold=float(cfg.li.trace_threshold),
        mu_water=float(cfg.li.mu_water),
        minimum_hu=float(cfg.li.minimum_hu),
    )
    return np.ascontiguousarray(result.image_hu, dtype=np.float32), True


def default_output_root(cfg, input_root: Path) -> Path:
    configured = getattr(cfg.data, "li_root", None)
    if configured:
        return Path(configured)
    return input_root.with_name(input_root.name + "_li")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute projection-domain LI for every with-metal CT slice"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-root", default=None, help="Defaults to data.with_metal")
    parser.add_argument("--output-root", default=None, help="Defaults to data.li_root")
    parser.add_argument("--backend", default=None, help="Override li.backend")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test file limit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_root = Path(args.input_root) if args.input_root else Path(cfg.data.with_metal)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else default_output_root(cfg, input_root)
    )
    backend = str(args.backend or cfg.li.backend)

    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Input and output roots must be different")

    sources = sorted(input_root.rglob("*.npy"))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        sources = sources[: args.limit]
    if not sources:
        raise RuntimeError(f"No .npy files found under {input_root}")

    counters = {"computed": 0, "skipped": 0, "metal": 0, "no_metal": 0}
    failures: list[dict[str, str]] = []
    progress = tqdm(sources, desc="Precomputing LI", unit="slice", dynamic_ncols=True)
    for source in progress:
        relative = source.relative_to(input_root)
        output = output_root / relative
        if not args.overwrite and valid_existing_output(source, output):
            counters["skipped"] += 1
            progress.set_postfix(**counters)
            continue
        try:
            image_hu = np.load(source, allow_pickle=False)
            li_hu, has_metal = compute_li(image_hu, cfg, backend)
            atomic_save_npy(output, li_hu)
            counters["computed"] += 1
            counters["metal" if has_metal else "no_metal"] += 1
        except Exception as exc:
            failures.append({"path": str(source), "error": repr(exc)})
            tqdm.write(f"FAILED: {source}: {exc}")
        progress.set_postfix(**counters)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "backend": backend,
        "num_angles": int(cfg.li.num_angles),
        "metal_threshold": float(cfg.data.metal_threshold),
        "total_selected": len(sources),
        **counters,
        "failed": len(failures),
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "li_precompute_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2))
    print(f"summary: {summary_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
