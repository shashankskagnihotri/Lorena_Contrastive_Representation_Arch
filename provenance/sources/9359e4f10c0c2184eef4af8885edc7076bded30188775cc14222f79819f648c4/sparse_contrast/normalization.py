"""Train-only population moments and immutable native-channel normalization."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.profiler import record_function

NORMALIZATION_MIN_STD = 1e-6


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PopulationMoments:
    """CPU float64 chunked Chan aggregation, counting natural zeros.

    Each call forms one chunk. var_mean uses correction=0 over the contiguous
    channel-first flattened chunk; chunks are merged in caller order.
    """

    def __init__(self, channels: int):
        self.channels = channels
        self.pixels = 0
        self.images = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, dtype=torch.float64)
        self.minimum = torch.full((channels,), float("inf"), dtype=torch.float64)
        self.maximum = torch.full((channels,), float("-inf"), dtype=torch.float64)
        self.zero_count = torch.zeros(channels, dtype=torch.int64)

    def update(self, native: torch.Tensor) -> None:
        if native.ndim != 4 or native.shape[1] != self.channels or native.shape[0] == 0:
            raise ValueError("Moments require nonempty BCHW with matching channels")
        values = native.detach().to(device="cpu", dtype=torch.float64).permute(1, 0, 2, 3).contiguous().flatten(1)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite coefficient encountered while fitting normalization")
        count = values.shape[1]
        variance, mean = torch.var_mean(values, dim=1, correction=0)
        if (variance < 0).any():
            raise ValueError("Negative population variance")
        delta = mean - self.mean
        total = self.pixels + count
        self.m2 += variance * count + delta.square() * (self.pixels * count / total)
        self.mean += delta * (count / total)
        self.pixels = total
        self.images += native.shape[0]
        self.minimum = torch.minimum(self.minimum, values.amin(1))
        self.maximum = torch.maximum(self.maximum, values.amax(1))
        self.zero_count += (values == 0).sum(1)

    def finalize(self, expected_images: int | None = None, expected_pixels: int | None = None) -> dict:
        if self.pixels == 0:
            raise ValueError("No images visited")
        if expected_images is not None and self.images != expected_images:
            raise ValueError(f"Image coverage mismatch: {self.images} != {expected_images}")
        if expected_pixels is not None and self.pixels != expected_pixels:
            raise ValueError(f"Pixel coverage mismatch: {self.pixels} != {expected_pixels}")
        variance = self.m2 / self.pixels
        if not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError("Invalid population variance")
        sigma = variance.sqrt()
        guarded = sigma < NORMALIZATION_MIN_STD
        effective = torch.where(guarded, torch.ones_like(sigma), sigma)
        return {
            "mean": self.mean.tolist(), "sigma": sigma.tolist(),
            "effective_std": effective.tolist(), "guarded_channels": guarded.tolist(),
            "normalization_min_std": NORMALIZATION_MIN_STD,
            "image_count": self.images,
            "pixel_counts": [self.pixels] * self.channels,
            "min": self.minimum.tolist(), "max": self.maximum.tolist(),
            "natural_zero_counts": self.zero_count.tolist(),
            "natural_zero_fraction": (self.zero_count.double() / self.pixels).tolist(),
            "accumulation_dtype": "float64",
            "accumulation_method": "ordered_cpu_Chan_chunks_torch_var_mean_correction_0",
            "variance_correction": 0,
        }


class FrozenNormalization(nn.Module):
    def __init__(self, mean: Iterable[float], effective_std: Iterable[float], artifact_hash: str = ""):
        super().__init__()
        means = torch.as_tensor(mean, dtype=torch.float32).flatten().clone()
        stds = torch.as_tensor(effective_std, dtype=torch.float32).flatten().clone()
        if means.numel() == 0 or means.shape != stds.shape:
            raise ValueError("Mean and effective std must be matching nonempty vectors")
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError("Normalization requires finite means and positive finite stds")
        self.register_buffer("mean", means)
        self.register_buffer("effective_std", stds)
        self.artifact_hash = artifact_hash

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 4 or coefficients.shape[1] != self.mean.numel():
            raise ValueError("Normalization requires matching native BCHW channels")
        if coefficients.dtype != torch.float32:
            raise TypeError("Native standardization requires float32")
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            with record_function("normalization/frozen_native_mean_std"):
                return (coefficients - self.mean[None, :, None, None]) / self.effective_std[None, :, None, None]

    def normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Normalize (..., C*P*P), preserving zero-coefficient affine offsets."""
        if patches.ndim < 2 or patches.shape[-1] % self.mean.numel():
            raise ValueError("Patch width must be divisible by native channel count")
        if patches.dtype != torch.float32:
            raise TypeError("Native patch standardization requires float32")
        with torch.autocast(device_type=patches.device.type, enabled=False):
            with record_function("normalization/frozen_patch_mean_std"):
                shaped = patches.reshape(*patches.shape[:-1], self.mean.numel(), patches.shape[-1] // self.mean.numel())
                return ((shaped - self.mean[:, None]) / self.effective_std[:, None]).reshape_as(patches)


@contextmanager
def artifact_lock(path: str | Path):
    """Advisory flock on the shared CephFS artifact; all fitters use this lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_bytes(path: Path, contents: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_artifact(path: str | Path, payload: dict) -> dict:
    """Write versioned JSON/NPZ atomically, then update the discoverable alias.

    Caller holds artifact_lock across cache check, fitting, and writing. The
    artifact SHA hashes canonical JSON excluding its own artifact_sha256 key;
    NPZ bytes have a separate actual file hash. Full JSON file hash can be
    recorded independently by checkpoint provenance without circular hashes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    artifact["schema_version"] = 1
    version = canonical_hash(artifact)[:16]
    npz_path = path.with_name(f"{path.stem}.{version}.npz")
    fd, temporary = tempfile.mkstemp(prefix=npz_path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream,
                     mean=np.asarray(artifact["mean"], dtype=np.float64),
                     sigma=np.asarray(artifact["sigma"], dtype=np.float64),
                     effective_std=np.asarray(artifact["effective_std"], dtype=np.float64),
                     pixel_counts=np.asarray(artifact["pixel_counts"], dtype=np.int64),
                     guarded_channels=np.asarray(artifact["guarded_channels"], dtype=bool))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, npz_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    artifact["npz_file"] = npz_path.name
    artifact["npz_sha256"] = file_sha256(npz_path)
    artifact["artifact_sha256"] = canonical_hash(artifact)
    encoded = (json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _atomic_bytes(path.with_name(f"{path.stem}.{version}.json"), encoded)
    _atomic_bytes(path, encoded)
    return artifact


def load_normalization(path: str | Path, expected: dict | None = None) -> tuple[FrozenNormalization, dict]:
    path = Path(path)
    with open(path, encoding="utf-8") as stream:
        artifact = json.load(stream)
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact.get("artifact_sha256") != canonical_hash(payload):
        raise ValueError(f"Normalization artifact payload hash mismatch: {path}")
    npz_path = path.parent / artifact["npz_file"]
    if file_sha256(npz_path) != artifact["npz_sha256"]:
        raise ValueError(f"Normalization NPZ file hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as npz:
        for key in ("mean", "sigma", "effective_std", "pixel_counts", "guarded_channels"):
            if not np.array_equal(npz[key], np.asarray(artifact[key])):
                raise ValueError(f"Normalization JSON/NPZ disagreement: {key}")
    if expected:
        for key, value in expected.items():
            if artifact.get(key) != value:
                raise ValueError(f"Normalization identity mismatch for {key}")
    return FrozenNormalization(artifact["mean"], artifact["effective_std"], artifact["artifact_sha256"]), artifact


def validate_sample_ids(actual: list[str], expected: list[str]) -> None:
    if len(set(actual)) != len(actual):
        raise ValueError("Duplicate sample IDs in normalization fit")
    if actual != expected:
        raise ValueError("Normalization IDs/order differ from the permitted clean training partition")


def check_standardized_moments(moments: dict, guarded: list[bool], tolerance: float = 2e-5) -> None:
    for c, is_guarded in enumerate(guarded):
        if not is_guarded and (abs(moments["mean"][c]) > tolerance or abs(moments["sigma"][c] - 1) > tolerance):
            raise ValueError(f"Standardized dense train moments failed for native channel {c}: {moments}")
