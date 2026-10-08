"""Requested deterministic updates and exact epoch-boundary continuation checks.

These use complete native models, short synthetic optimizer traces, and the
production optimizer/scheduler/RNG helpers. They are software correctness tests,
not benchmark accuracy estimates. Real-input overfit checks are separate.
"""
import gc
import os

import pytest
import torch

from sparse_contrast.common import (atomic_torch, initialize_matched, parameter_hash,
                                    restore_rng, rng_state, seed_all)
from sparse_contrast.models import build_backbone
from sparse_contrast.train import learning_rate, make_optimizer


RECIPE = dict(lr=3e-4, min_lr=1e-6, warmup_epochs=1, epochs=3,
              weight_decay=0.05, betas=[0.9, 0.999], grad_clip=1.)


def tree_equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert torch.equal(first.cpu(), second.cpu())
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            tree_equal(a, b)
    else:
        assert first == second


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def training_trace(architecture, stop, checkpoint=None, resume=None):
    device = torch.device(os.environ.get("SPARSE_BENCH_TEST_DEVICE", "cpu"))
    seed_all(17)
    model = build_backbone(architecture, 28, 1, drop_path=0.1).to(device)
    # Production initializes on CPU; do the same, preserving the caller RNG.
    model.cpu()
    initialize_matched(model, 17)
    model.to(device).train()
    optimizer = make_optimizer(model, RECIPE)
    generator = torch.Generator().manual_seed(551)
    patches = torch.randn(2, 196, 4, generator=generator).to(device)
    support = torch.zeros(2, 196, dtype=torch.bool, device=device)
    support[0, [3, 42, 101, 195]] = True
    support[1, [0, 9, 13, 99, 107, 157, 181]] = True
    targets = torch.tensor([1, 7], device=device)
    start, losses = 0, []
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start, losses = state["next_epoch"], state["losses"]
        # RNG byte states belong on CPU even when model tensors map to CUDA.
        state["rng"]["cpu"] = state["rng"]["cpu"].cpu()
        if state["rng"]["cuda"] is not None:
            state["rng"]["cuda"] = [item.cpu() for item in state["rng"]["cuda"]]
        restore_rng(state["rng"])
    for epoch in range(start, stop):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(RECIPE, epoch, 1)
        logits = model.forward_patches(patches, support)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert norm > 0
        optimizer.step()
        losses.append(float(loss.detach()))
    state = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                 next_epoch=stop, scheduler=dict(step=stop, recipe=RECIPE), rng=rng_state(), losses=losses)
    if checkpoint is not None:
        atomic_torch(checkpoint, state)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
def test_same_seed_and_serialized_epoch_resume_identical(architecture, tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    expected = training_trace(architecture, 3)
    repeated = training_trace(architecture, 3)
    for field in ("model", "optimizer", "scheduler", "losses"):
        tree_equal(expected[field], repeated[field])
    del repeated
    checkpoint = tmp_path / "epoch_boundary.pt"
    intermediate = training_trace(architecture, 1, checkpoint=checkpoint)
    del intermediate
    resumed = training_trace(architecture, 3, resume=checkpoint)
    for field in ("model", "optimizer", "scheduler", "losses"):
        tree_equal(expected[field], resumed[field])


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
def test_matched_initialization_independent_of_input_channels(architecture):
    torch.set_num_threads(2)
    first = build_backbone(architecture, 28, 1)
    second = build_backbone(architecture, 28, 3)
    initialize_matched(first, 0)
    initialize_matched(second, 0)
    other = dict(second.named_parameters())
    for name, parameter in first.named_parameters():
        if parameter.shape == other[name].shape:
            assert torch.equal(parameter, other[name]), name
    first_hash = parameter_hash(first)
    initialize_matched(first, 1)
    assert parameter_hash(first) != first_hash
