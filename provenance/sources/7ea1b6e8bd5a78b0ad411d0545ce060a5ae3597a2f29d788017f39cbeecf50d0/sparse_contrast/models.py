"""Native-grid ViT-Small/Swin-Tiny with physically compact sparse execution.

Patch vectors use (channel, local row, local column) order. The support passed
here was measured in native contrast units BEFORE affine normalization.
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F


EXECUTIONS = ("compact", "dense_masked", "dense")


def _scope(enabled, name):
    return torch.profiler.record_function(name) if enabled else nullcontext()


def _init(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _image_drop(x, image_ids, batch_size, probability, training):
    """One Bernoulli per ORIGINAL image, including images with no active rows."""
    if not training or probability == 0:
        return x
    mask = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    mask.bernoulli_(1 - probability).div_(1 - probability)
    return x * mask[image_ids].reshape((-1,) + (1,) * (x.ndim - 1))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, valid):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)
        self.drop_path = drop_path
        self.profiling = False

    def forward(self, x, valid):
        batch = x.shape[0]
        ids = torch.arange(batch, device=x.device)
        with _scope(self.profiling, "vit/attention"):
            residual = self.attn(self.norm1(x), valid)
        with _scope(self.profiling, "vit/attention_residual"):
            x = (x + _image_drop(residual, ids, batch, self.drop_path, self.training)) * valid[..., None]
        with _scope(self.profiling, "vit/mlp_norm_residual"):
            x = (x + _image_drop(self.mlp(self.norm2(x)), ids, batch, self.drop_path, self.training)) * valid[..., None]
        return x


class Backbone(nn.Module):
    def set_profiling(self, enabled=True):
        for module in self.modules():
            if hasattr(module, "profiling"):
                module.profiling = bool(enabled)

    def _validate(self, patches, support, execution):
        if execution not in EXECUTIONS:
            raise ValueError(f"unknown execution {execution}")
        expected = (self.grid * self.grid, self.in_channels * 4)
        if patches.ndim != 3 or tuple(patches.shape[1:]) != expected:
            raise ValueError(f"expected patches [B,{expected[0]},{expected[1]}], got {patches.shape}")
        if support is None:
            if execution != "dense":
                raise ValueError("sparse execution requires saved pre-normalization support")
            support = torch.ones(patches.shape[:2], device=patches.device, dtype=torch.bool)
        if support.shape != patches.shape[:2] or support.dtype != torch.bool or support.device != patches.device:
            raise ValueError("support must be bool [B,N] on the patch device")
        return support if execution != "dense" else torch.ones_like(support)

    def forward(self, images, support=None, execution=None):
        patches = F.unfold(images, kernel_size=2, stride=2).transpose(1, 2)
        return self.forward_patches(patches, support, execution)


class CompactViT(Backbone):
    def __init__(self, image_size, in_channels, drop_path=0.1, execution="compact"):
        super().__init__()
        self.image_size, self.grid, self.in_channels = image_size, image_size // 2, in_channels
        self.execution, self.profiling = execution, False
        self.patch_embed = nn.Linear(4 * in_channels, 384)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, 384))
        self.blocks = nn.ModuleList([ViTBlock(384, 6, drop_path * i / 11) for i in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Linear(384, 10)
        self.apply(_init)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        self.latest_telemetry = {}

    def forward_patches(self, patches, support=None, execution=None):
        execution = execution or self.execution
        support = self._validate(patches, support, execution)
        batch, count, _ = patches.shape
        lengths = support.sum(1)
        with _scope(self.profiling, "vit/patch_gather_pack"):
            if execution == "compact":
                # Dynamic maximum and nonzero extraction are charged to forward.
                length = int(lengths.max())
                image_ids, positions = support.nonzero(as_tuple=True)
                ranks = support.long().cumsum(1)[image_ids, positions] - 1
                retained = patches[image_ids, positions]
                valid_image = torch.arange(length, device=patches.device)[None, :] < lengths[:, None]
            else:
                length = count
                retained = patches
                valid_image = support
        with _scope(self.profiling, "vit/patch_projection_positions_class"):
            projected = self.patch_embed(retained)
            if execution == "compact":
                projected = projected + self.pos_embed[0, positions + 1]
                flat_slots = image_ids * length + ranks
                x = projected.new_zeros((batch * length, 384)).index_copy(0, flat_slots, projected)
                x = x.reshape(batch, length, 384)
            else:
                x = (projected + self.pos_embed[:, 1:]) * valid_image[..., None]
            cls = (self.cls_token + self.pos_embed[:, :1]).expand(batch, -1, -1)
            x = torch.cat((cls, x), dim=1)
            valid = torch.cat((torch.ones((batch, 1), device=patches.device, dtype=torch.bool), valid_image), 1)
        for index, block in enumerate(self.blocks):
            with _scope(self.profiling, f"vit/block_{index}"):
                x = block(x, valid)
        with _scope(self.profiling, "vit/final_norm_head"):
            logits = self.head(self.norm(x)[:, 0])
        self.latest_telemetry = {
            "execution": execution, "retained_per_image": lengths.detach(),
            "executed_sequence_length": length + 1,
            "executed_image_tokens": batch * length,
            "class_tokens": batch, "padding_tokens": batch * length - lengths.sum(),
            "projected_patch_rows": retained.numel() // (self.in_channels * 4),
        }
        return logits


def _window_geometry(grid, window, shift):
    """Map ORIGINAL raster positions to rolled windows and official shift regions."""
    if grid % window:
        raise ValueError("configured stage grids must be divisible by window size")
    shift = 0 if grid <= window else shift
    row, col = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    rr, cc = (row - shift) % grid, (col - shift) % grid
    windows = (rr // window) * (grid // window) + cc // window
    local = (rr % window) * window + cc % window
    regions = torch.zeros((grid, grid), dtype=torch.long)
    if shift:
        slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
        for a, hs in enumerate(slices):
            for b, ws in enumerate(slices):
                regions[hs, ws] = 3 * a + b
    return windows.flatten(), local.flatten(), regions[rr, cc].flatten(), shift


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads, self.window = heads, window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0) + window - 1
        relative[..., 0] *= 2 * window - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

    def attend(self, qkv, local, regions, valid=None):
        groups, length, _, dim = qkv.shape
        q, k, v = qkv.reshape(groups, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        rel = self.relative_position_index[local[:, :, None], local[:, None, :]]
        bias = self.relative_position_bias_table[rel].permute(0, 3, 1, 2)
        scores = scores + bias
        # Official Swin uses finite -100 for cyclic boundary restrictions.
        # Missing keys use -inf, with a dummy attention-only diagonal for
        # otherwise all-masked inactive queries. Their states are discarded.
        scores = scores + (regions[:, :, None] != regions[:, None, :])[:, None] * -100.0
        if valid is not None:
            allowed = valid[:, None, :].expand(-1, length, -1)
            fallback = (~valid.any(1))[:, None, None] & torch.eye(length, device=qkv.device, dtype=torch.bool)[None]
            scores = scores.masked_fill(~(allowed | fallback)[:, None], -torch.inf)
        return (scores.softmax(-1) @ v).transpose(1, 2).reshape(groups, length, dim)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, grid, window, shift, drop_path):
        super().__init__()
        self.grid, self.window, self.drop_path = grid, window, drop_path
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = WindowAttention(dim, heads, window)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(dim)
        win, local, regions, self.shift = _window_geometry(grid, window, shift)
        self.register_buffer("window_ids", win, persistent=False)
        self.register_buffer("local_positions", local, persistent=False)
        self.register_buffer("shift_regions", regions, persistent=False)
        self.profiling = False
        self.latest_telemetry = {}

    def compact(self, x, flat_ids, batch):
        image_ids = flat_ids // (self.grid ** 2)
        positions = flat_ids % (self.grid ** 2)
        windows_per_image = (self.grid // self.window) ** 2
        with _scope(self.profiling, "swin/window_group_shift_position"):
            window_ids = image_ids * windows_per_image + self.window_ids[positions]
            order = torch.argsort(window_ids, stable=True)
            counts = torch.bincount(window_ids, minlength=batch * windows_per_image)
            offsets = counts.cumsum(0) - counts
            # Exact retained-length groups: no QKV, MLP or attention padding.
            lengths = torch.unique(counts[counts > 0]).tolist()
        with _scope(self.profiling, "swin/attention"):
            qkv = self.attn.qkv(self.norm1(x)).reshape(x.shape[0], 3, x.shape[-1])
            attention = x.new_zeros(x.shape)
            for length in lengths:
                chosen = (counts == length).nonzero(as_tuple=True)[0]
                indices = order[offsets[chosen, None] + torch.arange(length, device=x.device)[None]]
                local = self.local_positions[positions[indices]]
                regions = self.shift_regions[positions[indices]]
                values = self.attn.attend(qkv[indices], local, regions)
                attention = attention.index_copy(0, indices.flatten(), values.flatten(0, 1))
            if not lengths:
                # Empty tensors retain zero parameter gradients without spatial tokens.
                attention = qkv[:, 0] + self.attn.relative_position_bias_table.sum() * 0
            residual = self.attn.proj(attention)
        with _scope(self.profiling, "swin/attention_residual"):
            x = x + _image_drop(residual, image_ids, batch, self.drop_path, self.training)
        with _scope(self.profiling, "swin/mlp_norm_residual"):
            x = x + _image_drop(self.mlp(self.norm2(x)), image_ids, batch, self.drop_path, self.training)
        self.latest_telemetry = {
            "active_per_image": torch.bincount(image_ids, minlength=batch).detach(),
            "window_lengths": counts.detach(), "executed_rows": x.shape[0],
            "active_windows": (counts > 0).sum(), "total_windows": counts.numel(),
            "window_length_groups": len(lengths),
            "executed_attention_pairs": counts.square().sum(), "padding_rows": 0,
        }
        return x

    def dense(self, x, support):
        batch, count, dim = x.shape
        window_count, area = (self.grid // self.window) ** 2, self.window ** 2
        # Permutation to exact rolled-window, row-major sequence order.
        with _scope(self.profiling, "swin/window_group_shift_position"):
            order = torch.argsort(self.window_ids * area + self.local_positions)
            local = self.local_positions[order].reshape(window_count, area).repeat(batch, 1)
            regions = self.shift_regions[order].reshape(window_count, area).repeat(batch, 1)
            valid = support[:, order].reshape(batch * window_count, area)
        with _scope(self.profiling, "swin/attention"):
            qkv = self.attn.qkv(self.norm1(x)).reshape(batch, count, 3, dim)
            qkv = qkv[:, order].reshape(batch * window_count, area, 3, dim)
            values = self.attn.attend(qkv, local, regions, valid)
            values = values.reshape(batch, count, dim)
            residual = self.attn.proj(values[:, torch.argsort(order)])
        ids = torch.arange(batch, device=x.device)
        with _scope(self.profiling, "swin/attention_residual"):
            x = (x + _image_drop(residual, ids, batch, self.drop_path, self.training)) * support[..., None]
        with _scope(self.profiling, "swin/mlp_norm_residual"):
            x = (x + _image_drop(self.mlp(self.norm2(x)), ids, batch, self.drop_path, self.training)) * support[..., None]
        counts = valid.sum(1)
        self.latest_telemetry = {
            "active_per_image": support.sum(1).detach(), "window_lengths": counts.detach(),
            "active_windows": (counts > 0).sum(), "total_windows": counts.numel(),
            "executed_rows": batch * count,
            "executed_attention_pairs": batch * window_count * area * area,
            "padding_rows": batch * count - support.sum(),
        }
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, grid):
        super().__init__()
        self.grid, self.output_grid = grid, (grid + 1) // 2
        self.norm = nn.LayerNorm(4 * dim, eps=1e-5)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def compact(self, x, flat_ids, batch):
        positions = flat_ids % self.grid ** 2
        image_ids = flat_ids // self.grid ** 2
        row, col = positions // self.grid, positions % self.grid
        parent = image_ids * self.output_grid ** 2 + (row // 2) * self.output_grid + col // 2
        parents, inverse = torch.unique(parent, sorted=True, return_inverse=True)
        # torchvision reference: [top-left, bottom-left, top-right, bottom-right].
        slot = row % 2 + 2 * (col % 2)
        children = x.new_zeros((parents.numel() * 4, x.shape[-1]))
        children = children.index_copy(0, inverse * 4 + slot, x)
        merged = children.reshape(parents.numel(), 4 * x.shape[-1])
        return self.reduction(self.norm(merged)), parents

    def dense(self, x, support):
        batch, _, dim = x.shape
        x = x.reshape(batch, self.grid, self.grid, dim)
        mask = support.reshape(batch, self.grid, self.grid)
        pad = self.grid % 2
        x = F.pad(x, (0, 0, 0, pad, 0, pad))
        mask = F.pad(mask, (0, pad, 0, pad))
        merged = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), -1)
        support = mask[:, 0::2, 0::2] | mask[:, 1::2, 0::2] | mask[:, 0::2, 1::2] | mask[:, 1::2, 1::2]
        x = self.reduction(self.norm(merged)) * support[..., None]
        return x.reshape(batch, -1, 2 * dim), support.reshape(batch, -1)


class CompactSwin(Backbone):
    def __init__(self, image_size, in_channels, drop_path=0.1, execution="compact"):
        super().__init__()
        self.image_size, self.grid, self.in_channels = image_size, image_size // 2, in_channels
        self.execution, self.profiling = execution, False
        self.depths, self.heads = (2, 2, 6, 2), (3, 6, 12, 24)
        self.windows = (7, 7, 4, 2) if image_size == 28 else (4, 4, 4, 2)
        self.patch_embed = nn.Linear(4 * in_channels, 96)
        self.patch_norm = nn.LayerNorm(96, eps=1e-5)
        self.stages, self.merges = nn.ModuleList(), nn.ModuleList()
        grid, index = self.grid, 0
        self.grids = []
        for stage, depth in enumerate(self.depths):
            self.grids.append(grid)
            dim, window = 96 * 2 ** stage, self.windows[stage]
            blocks = []
            for block in range(depth):
                blocks.append(SwinBlock(dim, self.heads[stage], grid, window, window // 2 if block % 2 else 0, drop_path * index / 11))
                index += 1
            self.stages.append(nn.ModuleList(blocks))
            if stage < 3:
                self.merges.append(PatchMerging(dim, grid))
                grid = (grid + 1) // 2
        self.norm = nn.LayerNorm(768, eps=1e-5)
        self.head = nn.Linear(768, 10)
        self.apply(_init)
        for stage in self.stages:
            for block in stage:
                nn.init.trunc_normal_(block.attn.relative_position_bias_table, std=0.02)
        self.latest_telemetry = {}

    def forward_patches(self, patches, support=None, execution=None):
        execution = execution or self.execution
        support = self._validate(patches, support, execution)
        batch = patches.shape[0]
        with _scope(self.profiling, "swin/patch_gather"):
            if execution == "compact":
                flat_ids = support.flatten().nonzero(as_tuple=True)[0]
                retained = patches.flatten(0, 1)[flat_ids]
            else:
                retained = patches
        with _scope(self.profiling, "swin/patch_projection_norm"):
            x = self.patch_norm(self.patch_embed(retained))
            if execution != "compact":
                x = x * support[..., None]
        telemetry = {"execution": execution, "initial_retained_per_image": support.sum(1).detach(), "stages": []}
        for stage_index, blocks in enumerate(self.stages):
            stage_info = {"grid": self.grids[stage_index], "blocks": []}
            for block_index, block in enumerate(blocks):
                with _scope(self.profiling, f"swin/stage_{stage_index}/block_{block_index}"):
                    x = block.compact(x, flat_ids, batch) if execution == "compact" else block.dense(x, support)
                    stage_info["blocks"].append(block.latest_telemetry)
            telemetry["stages"].append(stage_info)
            if stage_index < 3:
                with _scope(self.profiling, f"swin/merge_{stage_index}"):
                    if execution == "compact":
                        x, flat_ids = self.merges[stage_index].compact(x, flat_ids, batch)
                    else:
                        x, support = self.merges[stage_index].dense(x, support)
        with _scope(self.profiling, "swin/final_norm_pool_head"):
            x = self.norm(x)
            if execution == "compact":
                image_ids = flat_ids // self.grids[-1] ** 2
                count = torch.bincount(image_ids, minlength=batch)
                # Final 2x2 scatter avoids atomic duplicate-index summation.
                dense = x.new_zeros((batch * self.grids[-1] ** 2, 768)).index_copy(0, flat_ids, x)
                pooled = dense.reshape(batch, self.grids[-1] ** 2, 768).sum(1)
            else:
                count = support.sum(1)
                pooled = (x * support[..., None]).sum(1)
            pooled = pooled / count.clamp_min(1)[:, None]
            logits = self.head(pooled)
        telemetry["final_retained_per_image"] = count.detach()
        self.latest_telemetry = telemetry
        return logits


def build_backbone(architecture, image_size, in_channels, drop_path=0.1, execution="compact"):
    if image_size not in (28, 32) or not isinstance(in_channels, int) or in_channels < 1:
        raise ValueError("native image_size must be 28 or 32 and native in_channels positive")
    if execution not in EXECUTIONS or not 0 <= drop_path < 1:
        raise ValueError("invalid execution or stochastic-depth probability")
    names = {"vit_small": CompactViT, "vit": CompactViT, "swin_tiny": CompactSwin, "swin": CompactSwin}
    if architecture not in names:
        raise ValueError(f"unknown architecture {architecture}")
    return names[architecture](image_size, in_channels, drop_path, execution)


def dense_reference(model):
    """Build the pinned third-party dense architecture and copy EVERY parameter.

    The reference is used only for tests, never as compact execution. No
    non-strict checkpoint load or unmatched/random reference parameters.
    """
    if isinstance(model, CompactViT):
        from timm.models.vision_transformer import VisionTransformer
        ref = VisionTransformer(img_size=model.image_size, patch_size=2, in_chans=model.in_channels,
                                num_classes=10, embed_dim=384, depth=12, num_heads=6,
                                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_path_rate=model.blocks[-1].drop_path)
        for block in ref.blocks:
            block.attn.fused_attn = False
        state = model.state_dict()
        state["patch_embed.proj.weight"] = state.pop("patch_embed.weight").reshape(384, model.in_channels, 2, 2)
        state["patch_embed.proj.bias"] = state.pop("patch_embed.bias")
        ref.load_state_dict(state, strict=True)
    elif isinstance(model, CompactSwin):
        from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlock

        def block_factory(dim, num_heads, window_size, shift_size, **kwargs):
            stage = int(math.log2(dim // 96))
            window, grid = model.windows[stage], model.grids[stage]
            shift = window // 2 if shift_size[0] and grid > window else 0
            return SwinTransformerBlock(dim, num_heads, [window, window], [shift, shift], **kwargs)

        ref = SwinTransformer(patch_size=[2, 2], embed_dim=96, depths=list(model.depths),
                              num_heads=list(model.heads), window_size=[7, 7], num_classes=10,
                              stochastic_depth_prob=model.stages[-1][-1].drop_path, block=block_factory)
        mapped = {}
        mapped["features.0.0.weight"] = model.patch_embed.weight.reshape(96, model.in_channels, 2, 2)
        mapped["features.0.0.bias"] = model.patch_embed.bias
        # torchvision constructor fixes stem input channels to 3; replace stem
        # with the identical native-channel Conv2d before strict loading.
        ref.features[0][0] = nn.Conv2d(model.in_channels, 96, 2, stride=2)
        for key, value in model.patch_norm.state_dict().items():
            mapped[f"features.0.2.{key}"] = value
        for stage_index, blocks in enumerate(model.stages):
            for block_index, block in enumerate(blocks):
                prefix = f"features.{stage_index * 2 + 1}.{block_index}"
                for key, value in block.state_dict().items():
                    key = key.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.3.")
                    mapped[f"{prefix}.{key}"] = value
                mapped[f"{prefix}.attn.relative_position_index"] = block.attn.relative_position_index.flatten()
            if stage_index < 3:
                for key, value in model.merges[stage_index].state_dict().items():
                    mapped[f"features.{stage_index * 2 + 2}.{key}"] = value
        for module_name in ("norm", "head"):
            for key, value in getattr(model, module_name).state_dict().items():
                mapped[f"{module_name}.{key}"] = value
        ref.load_state_dict(mapped, strict=True)
    else:
        raise TypeError(type(model))
    return ref.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
