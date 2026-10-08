#!/usr/bin/env python3
"""Fit and verify all native-channel moments on the exact clean train split."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from sparse_contrast.frontend import Frontend, REPRESENTATIONS
from sparse_contrast.common import ROOT
from sparse_contrast.scope import read_scope, representations
from sparse_contrast.normalization import (
    PopulationMoments, FrozenNormalization, artifact_lock,
    check_standardized_moments, load_normalization, validate_sample_ids, write_artifact,
)


def selected_representations(dataset_name: str, requested: str, scope=None) -> tuple[str, ...]:
    scope = read_scope(ROOT) if scope is None else scope
    allowed = representations(dataset_name, scope)
    if requested == "all":
        return allowed
    if requested not in allowed:
        raise ValueError(f"Representation {requested!r} is outside the active {dataset_name} scope; allowed: {', '.join(allowed)}")
    return (requested,)


def fit_dataset(dataset_name: str, args: argparse.Namespace, scope=None) -> list[dict]:
    from sparse_contrast.data import load_clean

    modes = selected_representations(dataset_name, args.representation, scope)
    dataset = load_clean(dataset_name, split="train", data_root=args.data_root, protocol=args.protocol)
    if args.protocol == "heldout_val":
        expected_count = {"mnist": 55000, "cifar10": 45000}[dataset_name]
    else:
        expected_count = {"mnist": 60000, "cifar10": 50000}[dataset_name]
    if len(dataset) != expected_count:
        raise ValueError(f"Wrong {dataset_name}/{args.protocol} training size: {len(dataset)}")
    expected_ids = list(dataset.sample_ids)
    if len(set(expected_ids)) != expected_count:
        raise ValueError("Duplicate/missing IDs in training partition")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    frontends = {mode: Frontend(mode).to(args.device).eval() for mode in modes}
    paths = {mode: output_dir / f"{dataset_name}_{mode}_{args.protocol}.json" for mode in modes}
    identities = {mode: {
        "dataset": dataset_name, "representation": mode, "protocol": args.protocol,
        "split_hash": dataset.split_hash, "frontend_config": frontends[mode].config,
        "partition": "clean_train", "sparsity_percent_for_fit": 0,
        "input_range": [0.0, 1.0], "augmentation": "none",
    } for mode in modes}
    artifacts = {}
    with ExitStack() as stack:
        # Stable order prevents lock inversion when concurrent fitters share modes.
        for mode in sorted(modes):
            stack.enter_context(artifact_lock(paths[mode]))
        for mode in modes:
            if paths[mode].exists():
                # Refuse corrupted or incompatible aliases, preserving all evidence.
                # A changed frontend/split uses a separately chosen output directory.
                _, artifacts[mode] = load_normalization(paths[mode], expected=identities[mode])
                print(f"Verified cached {paths[mode]} {artifacts[mode]['artifact_sha256']}", flush=True)
        missing = [mode for mode in modes if mode not in artifacts]
        if not missing:
            return [artifacts[mode] for mode in modes]
        moments = {mode: PopulationMoments(frontends[mode].native_channels) for mode in missing}
        visited = []
        start = time.monotonic()
        image_shape = None
        with torch.inference_mode():
            for batch_number, (images, _labels, ids) in enumerate(loader):
                visited.extend(ids)
                if images.dtype != torch.uint8:
                    raise TypeError("Clean loader must yield decoded uint8 CHW inputs")
                if image_shape is None:
                    image_shape = tuple(images.shape[-2:])
                elif tuple(images.shape[-2:]) != image_shape:
                    raise ValueError("Unexpected changing native image dimensions")
                raw = images.to(args.device, dtype=torch.float32, non_blocking=False).div_(255)
                for mode in missing:
                    moments[mode].update(frontends[mode](raw))
                if batch_number % 20 == 0:
                    print(f"Fitting {dataset_name}: {len(visited)}/{expected_count} images", flush=True)
        validate_sample_ids(visited, expected_ids)
        pixels = expected_count * image_shape[0] * image_shape[1]
        fitted = {mode: moments[mode].finalize(expected_count, pixels) for mode in missing}
        norms = {mode: FrozenNormalization(fitted[mode]["mean"], fitted[mode]["effective_std"]).to(args.device)
                 for mode in missing}
        verification = {mode: PopulationMoments(frontends[mode].native_channels) for mode in missing}
        verified_ids = []
        with torch.inference_mode():
            for batch_number, (images, _labels, ids) in enumerate(loader):
                verified_ids.extend(ids)
                raw = images.to(args.device, dtype=torch.float32, non_blocking=False).div_(255)
                for mode in missing:
                    verification[mode].update(norms[mode](frontends[mode](raw)))
                if batch_number % 20 == 0:
                    print(f"Verifying standardized dense {dataset_name}: {len(verified_ids)}/{expected_count}", flush=True)
        validate_sample_ids(verified_ids, expected_ids)
        for mode in missing:
            standardized = verification[mode].finalize(expected_count, pixels)
            check_standardized_moments(standardized, fitted[mode]["guarded_channels"])
            payload = {
                **identities[mode], **fitted[mode],
                "sample_ids": expected_ids,
                "source_commit": frontends[mode].config["source_commit"],
                "source_file_sha256": frontends[mode].config["source_sha256"],
                "native_image_shape": list(image_shape),
                "channel_names": list(frontends[mode].channel_names),
                "reduction_order": {
                    "image_order": "dataset.sample_ids",
                    "batch_size": args.batch_size, "workers": args.workers,
                    "chunk_pixel_order": "native_channel_then_image_then_row_column",
                    "cpu_threads": torch.get_num_threads(),
                },
                "standardized_dense_train_validation": standardized,
                "setup_elapsed_seconds_shared_all_modes": time.monotonic() - start,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "torch_version": str(torch.__version__),
                "frontend_device": args.device,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            }
            artifacts[mode] = write_artifact(paths[mode], payload)
            load_normalization(paths[mode], expected=identities[mode])
            print(f"Saved {paths[mode]} guards={fitted[mode]['guarded_channels']} "
                  f"hash={artifacts[mode]['artifact_sha256']}", flush=True)
    return [artifacts[mode] for mode in modes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/ceph/sagnihot/datasets")
    parser.add_argument("--dataset", choices=("all", "mnist", "cifar10"), default="all")
    parser.add_argument("--representation", choices=("all", *REPRESENTATIONS), default="all")
    parser.add_argument("--protocol", choices=("heldout_val", "full_refit"), default="heldout_val")
    parser.add_argument("--output-dir", default="normalization")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    scope = read_scope(ROOT)
    datasets = ("mnist", "cifar10") if args.dataset == "all" else (args.dataset,)
    try:
        for dataset in datasets:
            selected_representations(dataset, args.representation, scope)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for dataset in datasets:
        fit_dataset(dataset, args, scope)


if __name__ == "__main__":
    main()
