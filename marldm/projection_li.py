from __future__ import annotations

import atexit
from collections import OrderedDict
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import shared_memory
import traceback
import warnings

import numpy as np
import torch

from .utils import cuda_device_index, trim_process_heap


@dataclass
class ProjectionLIResult:

    image_hu: np.ndarray
    sinogram: np.ndarray
    metal_trace: np.ndarray
    interpolated_sinogram: np.ndarray


def interpolate_projection(projection: np.ndarray, metal_trace: np.ndarray) -> np.ndarray:
    projection = np.asarray(projection, dtype=np.float32)
    trace = np.asarray(metal_trace, dtype=bool)
    if projection.shape != trace.shape:
        raise ValueError(f"projection/trace shape mismatch: {projection.shape} vs {trace.shape}")

    corrected = projection.copy()
    detector_positions = np.arange(corrected.shape[1])
    for view in range(corrected.shape[0]):
        metal = trace[view]
        if not metal.any():
            continue
        nonmetal = ~metal
        if int(nonmetal.sum()) < 2:
            continue
        corrected[view, metal] = np.interp(
            detector_positions[metal],
            detector_positions[nonmetal],
            corrected[view, nonmetal],
        )
    return corrected


class _AstraWorkspace:

    def __init__(
        self,
        height: int,
        width: int,
        num_angles: int,
        detector_spacing: float,
        detector_count: int,
        backend: str,
    ) -> None:
        import astra

        if height != width:
            raise ValueError(
                "The current ASTRA LI implementation expects square CT slices"
            )
        if detector_count <= 0:
            detector_count = int(np.ceil(np.sqrt(2.0) * max(height, width)))

        self.astra = astra
        self.backend = backend
        self.projector_id = None
        self.data_ids: list[int] = []
        self.algorithm_ids: list[int] = []
        self.closed = False

        angles = np.linspace(
            0.0, np.pi, int(num_angles), endpoint=False, dtype=np.float32
        )
        self.volume_geometry = astra.create_vol_geom(height, width)
        self.projection_geometry = astra.create_proj_geom(
            "parallel",
            float(detector_spacing),
            int(detector_count),
            angles,
        )

        try:
            if backend == "astra_cpu":
                self.projector_id = astra.create_projector(
                    "linear", self.projection_geometry, self.volume_geometry
                )

            self.image_volume_id = astra.data2d.create(
                "-vol", self.volume_geometry
            )
            self.metal_volume_id = astra.data2d.create(
                "-vol", self.volume_geometry
            )
            self.image_sinogram_id = astra.data2d.create(
                "-sino", self.projection_geometry
            )
            self.metal_sinogram_id = astra.data2d.create(
                "-sino", self.projection_geometry
            )
            self.interpolated_sinogram_id = astra.data2d.create(
                "-sino", self.projection_geometry
            )
            self.reconstruction_id = astra.data2d.create(
                "-vol", self.volume_geometry
            )
            self.data_ids.extend(
                [
                    self.image_volume_id,
                    self.metal_volume_id,
                    self.image_sinogram_id,
                    self.metal_sinogram_id,
                    self.interpolated_sinogram_id,
                    self.reconstruction_id,
                ]
            )

            fp_name = "FP_CUDA" if backend == "astra_cuda" else "FP"
            image_fp_config = astra.astra_dict(fp_name)
            image_fp_config["VolumeDataId"] = self.image_volume_id
            image_fp_config["ProjectionDataId"] = self.image_sinogram_id
            metal_fp_config = astra.astra_dict(fp_name)
            metal_fp_config["VolumeDataId"] = self.metal_volume_id
            metal_fp_config["ProjectionDataId"] = self.metal_sinogram_id
            if self.projector_id is not None:
                image_fp_config["ProjectorId"] = self.projector_id
                metal_fp_config["ProjectorId"] = self.projector_id
            self.image_fp_id = astra.algorithm.create(image_fp_config)
            self.metal_fp_id = astra.algorithm.create(metal_fp_config)
            self.algorithm_ids.extend([self.image_fp_id, self.metal_fp_id])

            fbp_name = "FBP_CUDA" if backend == "astra_cuda" else "FBP"
            fbp_config = astra.astra_dict(fbp_name)
            fbp_config["ProjectionDataId"] = self.interpolated_sinogram_id
            fbp_config["ReconstructionDataId"] = self.reconstruction_id
            if self.projector_id is not None:
                fbp_config["ProjectorId"] = self.projector_id
            fbp_config["option"] = {"FilterType": "ram-lak"}
            self.fbp_id = astra.algorithm.create(fbp_config)
            self.algorithm_ids.append(self.fbp_id)
        except Exception:
            self.close()
            raise

    def run(
        self,
        image: np.ndarray,
        metal_mask: np.ndarray,
        trace_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.closed:
            raise RuntimeError("Attempted to use a closed ASTRA workspace")

        astra = self.astra
        astra.data2d.store(
            self.image_volume_id,
            np.ascontiguousarray(image, dtype=np.float32),
        )
        astra.data2d.store(
            self.metal_volume_id,
            np.ascontiguousarray(metal_mask, dtype=np.float32),
        )
        astra.algorithm.run(self.image_fp_id)
        astra.algorithm.run(self.metal_fp_id)
        sinogram = astra.data2d.get(self.image_sinogram_id).astype(
            np.float32, copy=False
        )
        projected_metal = astra.data2d.get(self.metal_sinogram_id)
        trace = projected_metal > float(trace_threshold)
        interpolated = interpolate_projection(sinogram, trace)
        astra.data2d.store(
            self.interpolated_sinogram_id,
            np.ascontiguousarray(interpolated, dtype=np.float32),
        )
        astra.algorithm.run(self.fbp_id)
        reconstructed = astra.data2d.get(self.reconstruction_id).astype(
            np.float32, copy=False
        )
        return sinogram, trace, interpolated, reconstructed

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.algorithm_ids:
            try:
                self.astra.algorithm.delete(self.algorithm_ids)
            except Exception:
                pass
            self.algorithm_ids.clear()
        if self.data_ids:
            try:
                self.astra.data2d.delete(self.data_ids)
            except Exception:
                pass
            self.data_ids.clear()
        if self.projector_id is not None:
            try:
                self.astra.projector.delete(self.projector_id)
            except Exception:
                pass
            self.projector_id = None


_ASTRA_WORKSPACES: OrderedDict[tuple, _AstraWorkspace] = OrderedDict()
_MAX_ASTRA_WORKSPACES = 4


def _astra_workspace(
    height: int,
    width: int,
    num_angles: int,
    detector_spacing: float,
    detector_count: int,
    backend: str,
) -> _AstraWorkspace:
    key = (
        int(height),
        int(width),
        int(num_angles),
        float(detector_spacing),
        int(detector_count),
        str(backend),
    )
    workspace = _ASTRA_WORKSPACES.get(key)
    if workspace is not None:
        _ASTRA_WORKSPACES.move_to_end(key)
        return workspace

    if len(_ASTRA_WORKSPACES) >= _MAX_ASTRA_WORKSPACES:
        _, oldest = _ASTRA_WORKSPACES.popitem(last=False)
        oldest.close()
    workspace = _AstraWorkspace(*key)
    _ASTRA_WORKSPACES[key] = workspace
    return workspace


def _astra_forward_and_fbp(
    image: np.ndarray,
    metal_mask: np.ndarray,
    *,
    backend: str,
    num_angles: int,
    detector_spacing: float,
    detector_count: int,
    trace_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    workspace = _astra_workspace(
        height,
        width,
        int(num_angles),
        float(detector_spacing),
        int(detector_count),
        backend,
    )
    return workspace.run(image, metal_mask, trace_threshold)


_astra_batches_since_cleanup = 0
_li_calls_since_host_trim = 0


def _astra_cleanup_due(interval: int) -> bool:
    global _astra_batches_since_cleanup
    _astra_batches_since_cleanup += 1
    return _astra_batches_since_cleanup >= max(1, int(interval))


def _host_trim_due(interval: int) -> bool:
    global _li_calls_since_host_trim
    _li_calls_since_host_trim += 1
    if _li_calls_since_host_trim < max(1, int(interval)):
        return False
    _li_calls_since_host_trim = 0
    return True


def clear_astra_memory(*, warn: bool = True) -> None:
    global _astra_batches_since_cleanup, _li_calls_since_host_trim
    try:
        import astra
    except (ImportError, ModuleNotFoundError):
        return
    try:
        for workspace in list(_ASTRA_WORKSPACES.values()):
            workspace.close()
        _ASTRA_WORKSPACES.clear()
        astra.clear()
        _astra_batches_since_cleanup = 0
        _li_calls_since_host_trim = 0
    except Exception as exc:
        if warn:
            warnings.warn(
                f"ASTRA global memory cleanup failed: {exc}", RuntimeWarning
            )


def _clear_astra_at_exit() -> None:
    clear_astra_memory(warn=False)


atexit.register(_clear_astra_at_exit)


def _skimage_forward_and_fbp(
    image: np.ndarray,
    metal_mask: np.ndarray,
    *,
    num_angles: int,
    trace_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from skimage.transform import iradon, radon

    theta = np.linspace(0.0, 180.0, int(num_angles), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False, preserve_range=True).T.astype(np.float32)
    projected_metal = radon(
        metal_mask.astype(np.float32), theta=theta, circle=False, preserve_range=True
    ).T
    trace = projected_metal > float(trace_threshold)
    interpolated = interpolate_projection(sinogram, trace)
    reconstructed = iradon(
        interpolated.T,
        theta=theta,
        output_size=image.shape[0],
        filter_name="ramp",
        circle=False,
        preserve_range=True,
    ).astype(np.float32)
    return sinogram, trace, interpolated, reconstructed


def projection_li_slice(
    image_hu: np.ndarray,
    metal_mask: np.ndarray,
    *,
    backend: str = "astra_cuda",
    num_angles: int = 720,
    detector_spacing: float = 1.0,
    detector_count: int = 0,
    trace_threshold: float = 0.0,
    mu_water: float = 0.192,
    minimum_hu: float = -1000.0,
    allow_fallback: bool = True,
) -> ProjectionLIResult:
    image_hu = np.asarray(image_hu, dtype=np.float32)
    metal_mask = np.asarray(metal_mask, dtype=np.float32)
    if image_hu.ndim != 2 or metal_mask.shape != image_hu.shape:
        raise ValueError("image_hu and metal_mask must be same-shaped 2D arrays")
    if not np.any(metal_mask > 0.5):
        empty = np.zeros((int(num_angles), 1), dtype=np.float32)
        return ProjectionLIResult(image_hu.copy(), empty, empty.astype(bool), empty.copy())

    attenuation = (np.maximum(image_hu, float(minimum_hu)) / 1000.0 + 1.0) * float(mu_water)
    attenuation = np.maximum(attenuation, 0.0).astype(np.float32)
    selected_backend = str(backend).lower()
    if selected_backend in {"astra", "cuda"}:
        selected_backend = "astra_cuda"
    if selected_backend == "cpu":
        selected_backend = "astra_cpu"

    try:
        if selected_backend in {"astra_cuda", "astra_cpu"}:
            sinogram, trace, interpolated, reconstructed = _astra_forward_and_fbp(
                attenuation,
                metal_mask > 0.5,
                backend=selected_backend,
                num_angles=num_angles,
                detector_spacing=detector_spacing,
                detector_count=detector_count,
                trace_threshold=trace_threshold,
            )
        elif selected_backend == "skimage":
            sinogram, trace, interpolated, reconstructed = _skimage_forward_and_fbp(
                attenuation,
                metal_mask > 0.5,
                num_angles=num_angles,
                trace_threshold=trace_threshold,
            )
        else:
            raise ValueError(f"Unknown LI backend: {backend}")
    except (ImportError, RuntimeError) as exc:
        if selected_backend == "skimage" or not allow_fallback:
            raise
        warnings.warn(f"ASTRA LI failed ({exc}); falling back to scikit-image", RuntimeWarning)
        sinogram, trace, interpolated, reconstructed = _skimage_forward_and_fbp(
            attenuation,
            metal_mask > 0.5,
            num_angles=num_angles,
            trace_threshold=trace_threshold,
        )

    reconstructed_hu = (reconstructed / float(mu_water) - 1.0) * 1000.0
    return ProjectionLIResult(
        reconstructed_hu.astype(np.float32),
        sinogram,
        trace.astype(bool),
        interpolated,
    )


def _astra_worker_main(
    connection,
    image_shm_name: str,
    mask_shm_name: str,
    output_shm_name: str,
    shape: tuple[int, ...],
    projection_kwargs: dict,
) -> None:
    image_shm = mask_shm = output_shm = None
    try:
        image_shm = shared_memory.SharedMemory(name=image_shm_name)
        mask_shm = shared_memory.SharedMemory(name=mask_shm_name)
        output_shm = shared_memory.SharedMemory(name=output_shm_name)
        images_hu = np.ndarray(shape, dtype=np.float32, buffer=image_shm.buf)
        masks = np.ndarray(shape, dtype=np.float32, buffer=mask_shm.buf)
        output_hu = np.ndarray(shape, dtype=np.float32, buffer=output_shm.buf)

        import astra

        if str(projection_kwargs["backend"]).lower() in {
            "astra",
            "cuda",
            "astra_cuda",
        }:
            astra.set_gpu_index(0)
        connection.send(("ready", None))

        while True:
            command = connection.recv()
            if command == "stop":
                connection.send(("stopped", None))
                break
            if command != "run":
                raise RuntimeError(f"Unknown ASTRA worker command: {command!r}")

            output_hu[...] = images_hu
            for batch_index in range(shape[0]):
                for channel_index in range(shape[1]):
                    current_mask = masks[batch_index, channel_index]
                    if not np.any(current_mask > 0.5):
                        continue
                    result = projection_li_slice(
                        images_hu[batch_index, channel_index],
                        current_mask,
                        allow_fallback=False,
                        **projection_kwargs,
                    )
                    output_hu[batch_index, channel_index] = result.image_hu
                    del result
            connection.send(("ok", None))
    except EOFError:
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        clear_astra_memory(warn=False)
        for segment in (image_shm, mask_shm, output_shm):
            if segment is not None:
                try:
                    segment.close()
                except OSError:
                    pass
        try:
            connection.close()
        except OSError:
            pass


class _AstraProcessService:

    def __init__(
        self,
        shape: tuple[int, ...],
        projection_kwargs: dict,
        restart_interval: int,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.projection_kwargs = dict(projection_kwargs)
        self.restart_interval = max(1, int(restart_interval))
        self.signature = (
            self.shape,
            tuple(sorted(self.projection_kwargs.items())),
            self.restart_interval,
        )
        self.context = mp.get_context("spawn")
        self.connection = None
        self.process = None
        self.calls_since_restart = 0
        self.generation = 0
        self.closed = False

        size_bytes = int(np.prod(self.shape, dtype=np.int64)) * np.dtype(
            np.float32
        ).itemsize
        self.image_shm = shared_memory.SharedMemory(
            create=True, size=size_bytes
        )
        self.mask_shm = shared_memory.SharedMemory(
            create=True, size=size_bytes
        )
        self.output_shm = shared_memory.SharedMemory(
            create=True, size=size_bytes
        )
        self.images_hu = np.ndarray(
            self.shape, dtype=np.float32, buffer=self.image_shm.buf
        )
        self.masks = np.ndarray(
            self.shape, dtype=np.float32, buffer=self.mask_shm.buf
        )
        self.output_hu = np.ndarray(
            self.shape, dtype=np.float32, buffer=self.output_shm.buf
        )
        try:
            self._start_worker()
        except Exception:
            self.close()
            raise

    def _start_worker(self) -> None:
        parent_connection, child_connection = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=_astra_worker_main,
            args=(
                child_connection,
                self.image_shm.name,
                self.mask_shm.name,
                self.output_shm.name,
                self.shape,
                self.projection_kwargs,
            ),
            name="model_test10_astra",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self.connection = parent_connection
        self.process = process
        if not parent_connection.poll(60.0):
            self._stop_worker()
            raise RuntimeError("Timed out while starting the ASTRA worker")
        status, payload = parent_connection.recv()
        if status != "ready":
            self._stop_worker()
            raise RuntimeError(
                f"ASTRA worker failed during startup:\n{payload or status}"
            )
        self.calls_since_restart = 0
        self.generation += 1

    def _stop_worker(self) -> None:
        connection, process = self.connection, self.process
        self.connection = None
        self.process = None
        if connection is not None:
            if process is not None and process.is_alive():
                try:
                    connection.send("stop")
                    if connection.poll(5.0):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def _restart_worker(self) -> None:
        self._stop_worker()
        self._start_worker()

    def _run_once(
        self, images_hu: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        if self.connection is None or self.process is None:
            raise RuntimeError("ASTRA worker is not running")
        np.copyto(self.images_hu, images_hu, casting="unsafe")
        np.copyto(self.masks, masks, casting="unsafe")
        self.connection.send("run")
        if not self.connection.poll(120.0):
            raise RuntimeError("Timed out while waiting for the ASTRA worker")
        status, payload = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"ASTRA worker failed:\n{payload or status}")
        return self.output_hu.copy()

    def run(
        self, images_hu: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        if self.closed:
            raise RuntimeError("Attempted to use a closed ASTRA process service")
        if images_hu.shape != self.shape or masks.shape != self.shape:
            raise ValueError(
                f"ASTRA shared-memory shape mismatch: "
                f"{images_hu.shape}, {masks.shape}, expected {self.shape}"
            )
        if self.calls_since_restart >= self.restart_interval:
            self._restart_worker()

        try:
            result = self._run_once(images_hu, masks)
        except Exception:
            self._restart_worker()
            result = self._run_once(images_hu, masks)
        self.calls_since_restart += 1
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stop_worker()
        for segment in (self.image_shm, self.mask_shm, self.output_shm):
            try:
                segment.close()
            except OSError:
                pass
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


_ASTRA_PROCESS_SERVICE: _AstraProcessService | None = None


def _isolated_astra_projection(
    images_hu: np.ndarray,
    masks: np.ndarray,
    li_config,
) -> np.ndarray:
    global _ASTRA_PROCESS_SERVICE
    projection_kwargs = {
        "backend": str(li_config.backend),
        "num_angles": int(li_config.num_angles),
        "detector_spacing": float(li_config.detector_spacing),
        "detector_count": int(li_config.detector_count),
        "trace_threshold": float(li_config.trace_threshold),
        "mu_water": float(li_config.mu_water),
        "minimum_hu": float(li_config.minimum_hu),
    }
    restart_interval = max(
        1, int(getattr(li_config, "astra_worker_restart_interval", 128))
    )
    signature = (
        tuple(images_hu.shape),
        tuple(sorted(projection_kwargs.items())),
        restart_interval,
    )
    if (
        _ASTRA_PROCESS_SERVICE is None
        or _ASTRA_PROCESS_SERVICE.signature != signature
    ):
        if _ASTRA_PROCESS_SERVICE is not None:
            _ASTRA_PROCESS_SERVICE.close()
        _ASTRA_PROCESS_SERVICE = _AstraProcessService(
            tuple(images_hu.shape),
            projection_kwargs,
            restart_interval,
        )
    return _ASTRA_PROCESS_SERVICE.run(images_hu, masks)


def _shutdown_astra_process_service() -> None:
    global _ASTRA_PROCESS_SERVICE
    if _ASTRA_PROCESS_SERVICE is not None:
        _ASTRA_PROCESS_SERVICE.close()
        _ASTRA_PROCESS_SERVICE = None


def astra_worker_status() -> str:
    service = _ASTRA_PROCESS_SERVICE
    if service is None or service.process is None:
        return "astra_worker=inactive"
    return (
        f"astra_worker_pid={service.process.pid} "
        f"astra_generation={service.generation} "
        f"astra_calls={service.calls_since_restart}/{service.restart_interval}"
    )


atexit.register(_shutdown_astra_process_service)


@torch.no_grad()
def projection_domain_li(
    normalized_images: torch.Tensor,
    metal_masks: torch.Tensor,
    data_config,
    li_config,
) -> torch.Tensor:
    if normalized_images.ndim != 4 or metal_masks.ndim != 4:
        raise ValueError("normalized_images and metal_masks must be BCHW tensors")
    batch, channels, height, width = normalized_images.shape
    if metal_masks.shape[0] != batch or metal_masks.shape[-2:] != (height, width):
        raise ValueError("image/mask batch or spatial shape mismatch")
    if metal_masks.shape[1] == 1:
        metal_masks = metal_masks.expand(-1, channels, -1, -1)
    elif metal_masks.shape[1] != channels:
        raise ValueError("metal_masks must have either 1 or C channels")

    device = normalized_images.device
    output_dtype = normalized_images.dtype
    lo, hi = float(data_config.hu_min), float(data_config.hu_max)
    backend = str(li_config.backend).lower()
    uses_astra = backend in {
        "astra",
        "cuda",
        "cpu",
        "astra_cuda",
        "astra_cpu",
    }
    uses_astra_cuda = backend in {"astra", "cuda", "astra_cuda"}
    isolate_astra = uses_astra_cuda and bool(
        getattr(li_config, "isolate_astra_process", True)
    )
    synchronize_cuda = (
        uses_astra_cuda and not isolate_astra and device.type == "cuda"
    )
    cuda_index = cuda_device_index(device) if synchronize_cuda else None
    cleanup_enabled = uses_astra and not isolate_astra and bool(
        getattr(li_config, "clear_astra_after_batch", True)
    )
    cleanup_interval = max(
        1, int(getattr(li_config, "astra_cleanup_interval", 64))
    )
    cleanup_due = cleanup_enabled and _astra_cleanup_due(cleanup_interval)
    trim_host = bool(getattr(li_config, "trim_host_memory_after_astra", True))
    host_trim_interval = max(
        1, int(getattr(li_config, "host_trim_interval", 256))
    )
    host_trim_due = trim_host and _host_trim_due(host_trim_interval)

    if cleanup_due:
        clear_astra_memory()
    if synchronize_cuda and cleanup_due:
        torch.cuda.synchronize(cuda_index)
        if bool(getattr(li_config, "release_torch_cache_before_astra", True)):
            torch.cuda.empty_cache()

    completed = False
    try:
        images = normalized_images.detach().float().cpu().numpy()
        masks = metal_masks.detach().float().cpu().numpy()
        images_hu = (images + 1.0) * 0.5 * (hi - lo) + lo
        if isolate_astra:
            output_hu = _isolated_astra_projection(
                images_hu, masks, li_config
            )
        else:
            output_hu = images_hu.copy()
            for b in range(batch):
                for c in range(channels):
                    if not np.any(masks[b, c] > 0.5):
                        continue
                    result = projection_li_slice(
                        images_hu[b, c],
                        masks[b, c],
                        backend=li_config.backend,
                        num_angles=int(li_config.num_angles),
                        detector_spacing=float(li_config.detector_spacing),
                        detector_count=int(li_config.detector_count),
                        trace_threshold=float(li_config.trace_threshold),
                        mu_water=float(li_config.mu_water),
                        minimum_hu=float(li_config.minimum_hu),
                    )
                    output_hu[b, c] = result.image_hu
                    del result
        completed = True
    finally:
        if cleanup_enabled and not completed:
            clear_astra_memory()
        if isolate_astra and not completed:
            _shutdown_astra_process_service()
        if synchronize_cuda and not completed:
            torch.cuda.synchronize(cuda_index)

    np.clip(output_hu, lo, hi, out=output_hu)
    normalized = (output_hu - lo) / (hi - lo) * 2.0 - 1.0
    output = torch.from_numpy(
        normalized.astype(np.float32, copy=False)
    ).to(device=device, dtype=output_dtype)
    del normalized, output_hu, images_hu, images, masks
    if host_trim_due:
        trim_process_heap()
    return output
