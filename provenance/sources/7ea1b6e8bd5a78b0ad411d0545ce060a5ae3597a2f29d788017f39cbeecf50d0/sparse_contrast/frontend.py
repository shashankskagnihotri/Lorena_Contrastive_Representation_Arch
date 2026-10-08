"""Fixed native contrast port; provenance and source caveats: PREPROCESSING.md.

Derived from Shashank Agnihotri's semseg/utils/blur_preprocessing.py,
commit 73deca8c15b8d91fee29bac71b6f73c013a47d22. No upstream license was
present in that checkout; this port does not assert a new upstream license.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

SOURCE_COMMIT = "73deca8c15b8d91fee29bac71b6f73c013a47d22"
SOURCE_SHA256 = "f7c4420a3ab2bf4036323b9b4842e66d46225a88ce6f4440101c7a190861dff8"
REPRESENTATIONS = ("single_color", "grayscale", "color_opponency")


def _native_weights(representation: str, blur_depth: int) -> np.ndarray:
    """Preserve source float64 weight construction followed by float32 cast."""
    channels = 1 if representation == "grayscale" else 3
    depth = blur_depth + 1
    weight = np.ones((channels, depth * 3, 1, 1), dtype=np.float64)
    weight[:, :3] *= 1 / 3
    weight[:, 3:] *= -(1 / (depth * 3 - 3))
    if representation == "single_color":
        for c in range(channels):
            weight[c, :, 0, 0] = 0
            weight[c, c, 0, 0] = 1
            for i in range(1, depth):
                weight[c, i * 3 + c, 0, 0] = -1 / (depth - 1)
    elif representation == "color_opponency":
        intensity = weight[0].copy()
        weight = np.zeros_like(weight)
        weight[0] = intensity
        center_rg = [0.5, -0.5, 0]
        surround_rg = [-1 / ((depth - 1) * 2), 1 / ((depth - 1) * 2), 0]
        center_by = [-0.5 / 3, -0.5 / 3, 1 / 3]
        surround_by = [0.5 / (depth * 3 - 3), 0.5 / (depth * 3 - 3), -1 / (depth * 3 - 3)]
        for i in range(depth * 3):
            if i < 3:
                weight[1, i, 0, 0] = center_rg[i]
                weight[2, i, 0, 0] = center_by[i]
            else:
                weight[1, i, 0, 0] = surround_rg[(i - 3) % 3]
                weight[2, i, 0, 0] = surround_by[(i - 3) % 3]
    return weight


class Frontend(nn.Module):
    """Float32 BCHW raw [0,1] -> signed native contrast, with no learned state.

    The study uses blur_depth=1, the inspected application CLI default. Depth
    zero is intentionally rejected because it is a color map, not contrast.
    Grayscale source inputs are replicated only BEFORE the original RGB
    transform. The legacy grayscale-to-backbone adapter is not part of native
    coefficients. Input range/finite checks belong to dataset validation to
    avoid hidden CPU synchronization in online measurements.
    """

    def __init__(self, representation: str, blur_depth: int = 1):
        super().__init__()
        representation = "grayscale" if representation == "bw" else representation
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        if not isinstance(blur_depth, int) or blur_depth < 1:
            raise ValueError("A contrast frontend requires integer blur_depth >= 1")
        self.representation = representation
        self.profiling = False
        self.blur_depth = blur_depth
        self.native_channels = 1 if representation == "grayscale" else 3
        self.channel_names = {
            "single_color": ("red_contrast", "green_contrast", "blue_contrast"),
            "grayscale": ("mean_rgb_contrast",),
            "color_opponency": ("mean_rgb_contrast", "red_green_contrast", "blue_yellow_contrast"),
        }[representation]
        blur = np.full((3, 1, 3, 3), 1 / 9, dtype=np.float64)
        self.register_buffer("blur_weight", torch.from_numpy(blur).float())
        self.register_buffer("mix_weight", torch.from_numpy(_native_weights(representation, blur_depth)).float())
        self.config = {
            "version": 1,
            "representation": representation,
            "source_mode": "bw/black_white" if representation == "grayscale" else representation,
            "native_channels": self.native_channels,
            "channel_names": list(self.channel_names),
            "blur_depth": blur_depth,
            "blur_kernel": "3x3_uniform_1/9",
            "padding": "zero_1_each_iteration",
            "subtraction": "center_minus_mean_of_all_blur_iterations",
            "weight_arithmetic": "numpy_float64_construction_then_float32",
            "execution_dtype": "float32",
            "input_range": [0.0, 1.0],
            "grayscale_input": "replicate_to_RGB_before_transform",
            "native_output": "before_legacy_channel_adapter_threshold_and_normalization",
            "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
        }

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1] not in (1, 3):
            raise ValueError("Expected BCHW raw input with 1 or 3 channels")
        if raw.dtype != torch.float32:
            raise TypeError("Frontend input must be float32 raw [0,1]")
        with torch.autocast(device_type=raw.device.type, enabled=False):
            with record_function("frontend/input_channel_replication") if self.profiling else nullcontext():
                x = raw.expand(-1, 3, -1, -1) if raw.shape[1] == 1 else raw
            with record_function("frontend/blur_cascade") if self.profiling else nullcontext():
                levels = [x]
                for _ in range(self.blur_depth):
                    x = F.conv2d(x, self.blur_weight, padding=1, groups=3)
                    levels.append(x)
                concatenated = torch.cat(levels, dim=1)
            # The source implements color mapping and subtraction in ONE conv.
            with record_function("frontend/fused_color_mapping_signed_subtraction") if self.profiling else nullcontext():
                return F.conv2d(concatenated, self.mix_weight)


def sparsify(coefficients: torch.Tensor, sparsity_percent: float) -> torch.Tensor:
    """Exact per-image rank removal; ties use native C,H,W order."""
    if coefficients.ndim != 4 or coefficients.dtype != torch.float32:
        raise TypeError("Sparsification requires BCHW float32 native coefficients")
    if not math.isfinite(sparsity_percent) or not 0 <= sparsity_percent <= 100:
        raise ValueError("sparsity_percent must be in [0,100]")
    if sparsity_percent == 0:
        return coefficients
    with torch.autocast(device_type=coefficients.device.type, enabled=False):
        flat = coefficients.flatten(1)
        n_remove = math.floor(sparsity_percent * flat.shape[1] / 100)
        if n_remove == 0:
            return coefficients
        indices = torch.argsort(flat.abs(), dim=1, stable=True)
        out = flat.clone()
        out.scatter_(1, indices[:, :n_remove], 0.0)
        return out.reshape_as(coefficients)


def patchify(coefficients: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """B,C,H,W -> B,N,C*P*P. N=raster patch; final order=(C,dy,dx).

    Requires divisible native dimensions. Artificial model padding belongs in
    feature space and is never counted as a native coefficient here.
    """
    if coefficients.ndim != 4 or patch_size < 1:
        raise ValueError("Expected BCHW coefficients and positive patch size")
    b, c, h, w = coefficients.shape
    if h % patch_size or w % patch_size:
        raise ValueError("Native dimensions must be divisible by patch_size")
    return (coefficients.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
            .permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_size) * (w // patch_size), c * patch_size**2))


def patch_support(coefficients: torch.Tensor, patch_size: int = 2, epsilon: float = 0.0) -> torch.Tensor:
    """Native signed coefficients determine support BEFORE affine centering."""
    if coefficients.dtype != torch.float32:
        raise TypeError("Support computation requires float32 native coefficients")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    with torch.autocast(device_type=coefficients.device.type, enabled=False):
        return patchify(coefficients, patch_size).abs().amax(dim=-1) > epsilon
