#!/usr/bin/env python
"""Requested real-image, augmentation-free tiny-subset overfit diagnostic.

This is explicitly separate from production pilots/main runs. It never reads
test/corruption arrays and never changes a production recipe. Run on allocated
compute; the project launcher supplies the Slurm allocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sparse_contrast.common import (ROOT, atomic_json, atomic_torch, directory_lock,
                                    digest, hardware, initialize_matched,
                                    restore_rng, rng_state, seed_all, stop_requested)
from sparse_contrast.data import load_clean
from sparse_contrast.pipeline import Classifier
from sparse_contrast.train import make_optimizer


def run_check(dataset, architecture, device, output, max_epochs, samples_per_class):
    config = dict(dataset=dataset, architecture=architecture, representation="raw", execution="dense",
                  sparsity_percent=None, seed=0, protocol="heldout_val", precision="fp32",
                  diagnostic="augmentation_free_tiny_subset_overfit", samples_per_class=samples_per_class,
                  max_epochs=max_epochs, batch_size=8, loss_target=0.10, accuracy_target=1.0,
                  recipe=dict(stochastic_depth=0.0, lr=3e-4, weight_decay=0.0,
                              betas=[0.9, 0.999], grad_clip=1.0))
    config["config_hash"] = digest(config)
    run_dir = output / dataset / architecture / config["config_hash"]
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "completed.json"
    if completed.exists():
        return json.loads(completed.read_text())
    with directory_lock(run_dir / "writer.lock"):
        seed_all(config["seed"])
        data = load_clean(dataset, "train", protocol="heldout_val")
        selected = []
        for label in range(10):
            choices = np.flatnonzero(np.asarray(data.labels) == label)[:samples_per_class]
            if len(choices) != samples_per_class:
                raise ValueError("Not enough clean training images for the fixed diagnostic panel")
            selected.extend(int(index) for index in choices)
        records = [data[index] for index in selected]
        raw = torch.stack([row[0] for row in records]).to(device)
        labels = torch.tensor([row[1] for row in records], device=device)
        identities = [row[2] for row in records]
        atomic_json(run_dir / "config.json", dict(config, sample_ids=identities, split_hash=data.split_hash,
                                                   hardware=hardware(), invocation=sys.argv))
        model = Classifier(config)
        initialize_matched(model.backbone, config["seed"])
        model.to(device)
        optimizer = make_optimizer(model, config["recipe"])
        checkpoint = run_dir / "latest.pt"
        epoch_start, step, history = 0, 0, []
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["config_hash"] != config["config_hash"] or state["sample_ids"] != identities:
                raise ValueError("Overfit continuation config/sample identity mismatch")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            epoch_start, step, history = state["next_epoch"], state["step"], state["history"]
            restore_rng(state["rng"])
        writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=step + 1 if step else None, flush_secs=30)
        writer.add_text("provenance/config", json.dumps(config, indent=2), step)
        writer.add_images("training_inputs/raw_unaugmented", raw.float().cpu() / 255., step)
        start_time = time.perf_counter()
        try:
            for epoch in range(epoch_start, max_epochs):
                if stop_requested():
                    raise InterruptedError("Project STOP marker")
                model.train()
                order = torch.randperm(len(selected), generator=torch.Generator().manual_seed(10000 + epoch)).to(device)
                loss_sum, correct = 0., 0
                for begin in range(0, len(order), 8):
                    ids = order[begin:begin + 8]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(raw[ids])
                    loss = torch.nn.functional.cross_entropy(logits, labels[ids])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite overfit loss at epoch {epoch}, step {step}")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if norm <= 0:
                        raise FloatingPointError("Overfit backward has no gradient flow")
                    optimizer.step()
                    step += 1
                    loss_sum += float(loss.detach()) * len(ids)
                    correct += int((logits.detach().argmax(1) == labels[ids]).sum())
                    writer.add_scalar("train_step/loss", float(loss.detach()), step)
                    writer.add_scalar("train_step/gradient_norm", float(norm), step)
                model.eval()
                monitor_loss, monitor_correct = 0., 0
                with torch.inference_mode():
                    for begin in range(0, len(selected), 8):
                        logits = model(raw[begin:begin + 8])
                        target = labels[begin:begin + 8]
                        monitor_loss += float(torch.nn.functional.cross_entropy(logits, target, reduction="sum"))
                        monitor_correct += int((logits.argmax(1) == target).sum())
                row = dict(epoch=epoch + 1, step=step, train_online_loss=loss_sum / len(selected),
                           train_online_accuracy=correct / len(selected), train_monitor_loss=monitor_loss / len(selected),
                           train_monitor_accuracy=monitor_correct / len(selected), train_count=len(selected))
                history.append(row)
                for name in ("train_online_loss", "train_online_accuracy", "train_monitor_loss", "train_monitor_accuracy"):
                    writer.add_scalar("overfit/" + name, row[name], step)
                passed = row["train_monitor_accuracy"] == 1.0 and row["train_monitor_loss"] < config["loss_target"]
                if passed or (epoch + 1) % 10 == 0 or epoch + 1 == max_epochs:
                    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
                                 next_epoch=epoch + 1, step=step, history=history,
                                 config_hash=config["config_hash"], sample_ids=identities)
                    atomic_torch(checkpoint, state)
                    atomic_json(run_dir / "history.json", history)
                    writer.flush()
                    print(json.dumps(dict(dataset=dataset, architecture=architecture, **row)), flush=True)
                if passed:
                    for name, parameter in model.named_parameters():
                        writer.add_histogram("parameters/" + name, parameter.detach().cpu(), step)
                    result = dict(state="verified", diagnostic=config["diagnostic"], dataset=dataset,
                                  architecture=architecture, sample_ids=identities, seed=0, config_hash=config["config_hash"],
                                  final=row, loss_reduced=row["train_monitor_loss"] < history[0]["train_monitor_loss"],
                                  elapsed_seconds=time.perf_counter() - start_time, hardware=hardware(),
                                  artifact_directory=str(run_dir), benchmark_result=False)
                    writer.flush()
                    events = EventAccumulator(str(run_dir / "tensorboard"), size_guidance={"scalars": 0})
                    events.Reload()
                    for tag in ("train_step/loss", "train_step/gradient_norm", "overfit/train_monitor_accuracy"):
                        if tag not in events.Tags()["scalars"]:
                            raise RuntimeError(f"Missing overfit TensorBoard metric: {tag}")
                    atomic_json(completed, result)
                    return result
            raise RuntimeError(f"{dataset}/{architecture} failed full-subset overfit criteria after {max_epochs} epochs; inspect saved history")
        except BaseException as error:
            atomic_json(run_dir / f"failure-{time.time_ns()}.json", dict(error=repr(error), traceback=traceback.format_exc(),
                                                                        epoch=locals().get("epoch"), step=step))
            raise
        finally:
            writer.flush()
            writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "all"], default="all")
    parser.add_argument("--architecture", choices=["vit_small", "swin_tiny", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/checks/overfit")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--samples-per-class", type=int, default=2)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This cluster diagnostic requires a scheduler GPU allocation")
    if args.max_epochs < 1 or args.samples_per_class < 1:
        raise ValueError("Diagnostic sizes must be positive")
    datasets = ["mnist", "cifar10"] if args.dataset == "all" else [args.dataset]
    architectures = ["vit_small", "swin_tiny"] if args.architecture == "all" else [args.architecture]
    for dataset in datasets:
        for architecture in architectures:
            result = run_check(dataset, architecture, torch.device(args.device), args.output,
                               args.max_epochs, args.samples_per_class)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
