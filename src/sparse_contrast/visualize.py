"""Fixed, signed diagnostic panels. Rendered displays never enter model inputs."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import time
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="talk", style="whitegrid", font='serif', font_scale=1.0)
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .common import ROOT, atomic_json, digest, run_dir
from .data import load_clean, CorruptionDataset, StatelessAugment, fixed_panel_indices, corruption_cells
from .frontend import Frontend, REPRESENTATIONS, sparsify, patch_support, patchify
from .normalization import load_normalization


@contextmanager
def preserve_diagnostic_state(model=None):
    """Preserve Python/NumPy/torch RNGs, every module mode, and telemetry state."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    flags = {}
    if model is not None:
        for module in model.modules():
            for name in ('profiling', 'collect_diagnostics', 'latest_diagnostics', 'latest_telemetry'):
                if hasattr(module, name):
                    flags[(module, name)] = copy.deepcopy(getattr(module, name))
        model.eval()
        if hasattr(model, 'collect_diagnostics'):
            model.collect_diagnostics = False
    try:
        yield
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        for module, mode in modes:
            module.training = mode
        for (module, name), value in flags.items():
            setattr(module, name, value)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def panel_tensors(dataset):
    samples = [dataset[i] for i in fixed_panel_indices(dataset)]
    return torch.stack([value[0] for value in samples]), np.asarray([value[1] for value in samples]), np.asarray([value[2] for value in samples])


def signed_display_scale(c0, mean, std):
    """One scale per channel fitted on fixed CLEAN TRAIN diagnostics only.

    Include affine-standardized zero so the scale remains valid after rank
    removal. Display scales are symmetric; they do not alter stored tensors.
    """
    z0 = (c0 - mean) / std
    native = c0.abs().amax((0, 2, 3)).clamp_min(1e-12)
    normalized = torch.maximum(z0.abs().amax((0, 2, 3)), (mean / std).flatten().abs()).clamp_min(1e-12)
    return native.cpu().numpy(), normalized.cpu().numpy()


def preview_arrays(raw, frontend, mean, std, percent):
    with torch.inference_mode(), torch.autocast(device_type=raw.device.type, enabled=False):
        x = raw.float() / 255.0 if raw.dtype == torch.uint8 else raw.float()
        c0 = frontend(x)
        cs = sparsify(c0, percent)
        support = patch_support(cs, 2, 0.0)
        z0, zs = (c0 - mean) / std, (cs - mean) / std
    result = {'raw': raw.cpu().numpy(), 'native_0': c0.cpu().numpy(), 'native_sparse': cs.cpu().numpy(),
              'normalized_0': z0.cpu().numpy(), 'normalized_sparse': zs.cpu().numpy(),
              'coefficient_support': (cs != 0).cpu().numpy(), 'patch_support': support.cpu().numpy()}
    return result


def signed_panel_figure(arrays, channel, scales, labels, ids, percent, channel_name, first=0):
    """Up to five columns; all signed rows use matched fixed symmetric ranges."""
    stop = min(first + 5, len(labels))
    count = stop - first
    fig, axes = plt.subplots(7, count, figsize=(2.1 * count + 1.5, 12.8), squeeze=False, layout='constrained')
    native_scale, normalized_scale = float(scales[0][channel]), float(scales[1][channel])
    row_labels = ['Raw', 'Native 0%', f'Native {percent:g}%', 'Normalized 0%', f'Normalized {percent:g}%', 'Coefficient support', 'Patch support']
    images = {}
    for column, sample in enumerate(range(first, stop)):
        raw = arrays['raw'][sample]
        axes[0, column].imshow(raw[0] if raw.shape[0] == 1 else raw.transpose(1, 2, 0), cmap='gray' if raw.shape[0] == 1 else None, vmin=0, vmax=255)
        axes[0, column].set_title(f'y={int(labels[sample])}; {str(ids[sample]).split("/")[-1]}', fontsize=11)
        for row, key, scale in [(1, 'native_0', native_scale), (2, 'native_sparse', native_scale), (3, 'normalized_0', normalized_scale), (4, 'normalized_sparse', normalized_scale)]:
            images[row] = axes[row, column].imshow(arrays[key][sample, channel], cmap='RdBu_r', vmin=-scale, vmax=scale)
        axes[5, column].imshow(arrays['coefficient_support'][sample, channel], cmap='gray', vmin=0, vmax=1)
        h, w = raw.shape[-2:]
        support = arrays['patch_support'][sample].reshape(h // 2, w // 2)
        absent = ~np.repeat(np.repeat(support, 2, axis=0), 2, axis=1)
        # Gray crosshatching distinguishes absent patches from signed numerical zero.
        axes[4, column].contourf(absent, levels=[0.5, 1.5], colors='none', hatches=['xxxx']) if absent.any() else None
        axes[6, column].imshow(support, cmap='gray', vmin=0, vmax=1)
        for row in range(7):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            if column == 0:
                axes[row, column].set_ylabel(row_labels[row], fontsize=10)
    for row in (1, 2, 3, 4):
        fig.colorbar(images[row], ax=list(axes[row]), fraction=0.025, pad=0.01).ax.tick_params(labelsize=9)
    fig.suptitle(f'{channel_name}; hatched = absent patch', fontsize=13)
    return fig


def raw_prediction_figure(raw, augmented, labels, predictions, ids):
    fig, axes = plt.subplots(2, len(labels), figsize=(2.0 * len(labels), 4.7), squeeze=False, layout='constrained')
    for i in range(len(labels)):
        for row, values in enumerate((raw, augmented)):
            image = values[i].detach().cpu().numpy() if torch.is_tensor(values) else values[i]
            axes[row, i].imshow(image[0] if image.shape[0] == 1 else image.transpose(1, 2, 0), cmap='gray' if image.shape[0] == 1 else None, vmin=0, vmax=255)
            axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
        axes[0, i].set_title(f'{str(ids[i]).split("/")[-1]}\ny={int(labels[i])}', fontsize=10)
        axes[1, i].set_title(f'Prediction {int(predictions[i])}', fontsize=10)
    axes[0, 0].set_ylabel('Raw', fontsize=11)
    axes[1, 0].set_ylabel('Model input', fontsize=11)
    return fig


def confusion_figure(matrix, title):
    values = np.asarray(matrix)
    if values.shape != (10, 10) or not np.issubdtype(values.dtype, np.number) or np.any(values < 0):
        raise ValueError('Expected nonnegative 10x10 confusion counts')
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    sns.heatmap(values, ax=ax, cmap='Blues', square=True, cbar_kws={'label': 'Images'}, annot=False)
    ax.set(xlabel='Predicted class', ylabel='True class', title=title)
    return fig


def _shared_path(config):
    identity = {'dataset': config['dataset'], 'representation': config['representation'], 'protocol': config['protocol'], 'normalization_hash': config['normalization_hash']}
    return ROOT / 'outputs' / 'previews' / config['protocol'] / config['dataset'] / config['representation'] / digest(identity)



def recover_preview_lock(lock):
    """Recover only a recorded Slurm owner with positive terminal-state proof.

    A legacy lock without a job ID, an unknown job, or a live remote job remains
    untouched. campaign.recover_dead_lock rechecks the owner and scheduler state
    immediately before archiving its lease; generated preview artifacts remain.
    """
    owner_path = Path(lock) / 'owner.json'
    if not owner_path.exists():
        return False
    owner = json.loads(owner_path.read_text())
    job = owner.get('job_id')
    if not job:
        return False
    from .campaign import job_states, recover_dead_lock, TERMINAL_STATES
    state = job_states([str(job)]).get(str(job), {}).get('state')
    if state not in TERMINAL_STATES:
        return False
    recover_dead_lock(lock, expected_job=job)
    return True

def ensure_shared_previews(config, frontend, mean, std, device):
    """Create all five fixed panels once, under exclusive cross-node ownership."""
    directory = _shared_path(config)
    done, lock = directory / 'complete.json', directory.with_name(directory.name + '.lock')
    directory.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_recovery_check = float('-inf')
    while not done.exists():
        try:
            lock.mkdir()
        except FileExistsError:
            now = time.monotonic()
            if now - last_recovery_check >= 60:
                last_recovery_check = now
                if recover_preview_lock(lock):
                    continue
            if now - started > 1800:
                raise TimeoutError(f'Another preview writer did not complete: {lock}')
            time.sleep(1)
            continue
        atomic_json(lock / 'owner.json', {'pid': os.getpid(), 'host': os.uname().nodename, 'job_id': os.environ.get('SLURM_JOB_ID')})
        try:
            if not done.exists():
                _write_shared_previews(config, frontend, mean, std, device, directory)
        finally:
            (lock / 'owner.json').unlink(); lock.rmdir()
    return directory, json.loads(done.read_text())


def _write_shared_previews(config, frontend, mean, std, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    train = load_clean(config['dataset'], 'train', protocol=config['protocol'])
    train_raw, _, _ = panel_tensors(train)
    with torch.inference_mode():
        c0 = frontend(train_raw.to(device).float() / 255.0)
        scales = signed_display_scale(c0, mean, std)
    metadata = {'version': 1, 'dataset': config['dataset'], 'representation': config['representation'], 'normalization_hash': config['normalization_hash'],
                'display_only': True, 'display_scale_source': 'max abs over fixed unaugmented training panel, with affine zero offset included for normalized scale',
                'native_scales': scales[0].tolist(), 'normalized_scales': scales[1].tolist(), 'channel_names': list(frontend.channel_names),
                'train_panel_split_hash': train.split_hash, 'figures': {}, 'tensor_artifacts': {}, 'mean': mean.flatten().cpu().tolist(), 'std': std.flatten().cpu().tolist()}
    summary_dir = ROOT / 'outputs' / 'tensorboard' / 'study_summary' / config['protocol'] / config['dataset'] / 'preprocessing' / config['representation'] / config['normalization_hash']
    with SummaryWriter(str(summary_dir)) as writer:
        writer.add_text('provenance/display_policy', json.dumps(metadata, indent=2), 0)
        splits = ('train', 'val') if config['protocol'] == 'heldout_val' else ('train',)
        for split in splits:
            dataset = train if split == 'train' else load_clean(config['dataset'], split, protocol=config['protocol'])
            raw, labels, identities = panel_tensors(dataset)
            for percent in (0, 20, 40, 60, 80):
                arrays = preview_arrays(raw.to(device), frontend, mean, std, percent)
                tag = f'{split}/p{percent}'
                artifact = directory / f'{split}_p{percent}.npz'
                atomic_npz(artifact, **arrays, sample_ids=identities, labels=labels, native_scales=scales[0], normalized_scales=scales[1], mean=mean.cpu().numpy(), std=std.cpu().numpy())
                metadata['tensor_artifacts'][tag] = str(artifact)
                metadata['figures'][tag] = []
                for channel, channel_name in enumerate(frontend.channel_names):
                    for first in (0, 5):
                        fig = signed_panel_figure(arrays, channel, scales, labels, identities, percent, channel_name, first)
                        stem = f'{split}_p{percent}_c{channel}_g{first // 5}'
                        png = directory / f'{stem}.png'
                        fig.savefig(png, dpi=300, bbox_inches='tight')
                        fig.savefig(directory / f'{stem}.pdf', bbox_inches='tight')
                        writer.add_figure(f'{tag}/{channel_name}/group{first // 5}', fig, 0, close=True)
                        metadata['figures'][tag].append(str(png))
                for channel, channel_name in enumerate(frontend.channel_names):
                    for key in ('native_0', 'native_sparse', 'normalized_0', 'normalized_sparse'):
                        writer.add_histogram(f'{split}/{channel_name}/{key}/p{percent}', arrays[key][:, channel], 0)
                patches = patchify(torch.from_numpy(arrays['normalized_sparse']), 2)
                retained = patches[torch.from_numpy(arrays['patch_support'])]
                if retained.numel():
                    writer.add_histogram(f'{tag}/retained_normalized_patches', retained, 0)
        writer.flush()
    atomic_json(directory / 'complete.json', metadata)


def _log_cached_images(writer, metadata, split, percent, step):
    for index, path in enumerate(metadata['figures'][f'{split}/p{percent}']):
        # matplotlib already rendered signed tensors with explicit color scales.
        display = plt.imread(path)[..., :3]
        writer.add_image(f'fixed_preprocessing/{split}/p{percent}/figure{index}', display, step, dataformats='HWC')


def log_training_panel(model, config, writer, step, device, epoch=0):
    """Bounded independent panels, with exact model/RNG state restoration."""
    with preserve_diagnostic_state(model), torch.inference_mode():
        device = torch.device(device)
        output = run_dir(config) / 'diagnostics'
        output.mkdir(parents=True, exist_ok=True)
        summary = None
        if model.frontend is not None:
            _, summary = ensure_shared_previews(config, model.frontend, model.mean, model.std, device)
        splits = ('train', 'val') if config['protocol'] == 'heldout_val' else ('train',)
        for split in splits:
            dataset = load_clean(config['dataset'], split, protocol=config['protocol'])
            raw, labels, identities = panel_tensors(dataset)
            augmented = raw
            if split == 'train' and config['dataset'] == 'cifar10' and not config.get('disable_augmentation', False):
                wrapped = StatelessAugment(dataset, config['seed'])
                wrapped.set_epoch(epoch)
                samples = [wrapped[i][0] for i in fixed_panel_indices(dataset)]
                augmented = torch.stack(samples)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.get('precision') == 'bf16' and device.type == 'cuda'):
                logits = model(augmented.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite fixed-panel prediction')
            predictions = logits.argmax(1).cpu().numpy()
            atomic_npz(output / f'{split}_epoch{epoch}_step{step}.npz', raw=raw.numpy(), augmented=augmented.numpy(), labels=labels, predictions=predictions, logits=logits.float().cpu().numpy(), sample_ids=identities)
            writer.add_histogram(f'diagnostics/{split}/logits', logits.float().cpu(), step)
            writer.add_text(f'diagnostics/{split}/sample_ids', '\n'.join(identities.tolist()), step)
            for first in (0, 5):
                fig = raw_prediction_figure(raw[first:first + 5], augmented[first:first + 5], labels[first:first + 5], predictions[first:first + 5], identities[first:first + 5])
                fig.savefig(output / f'{split}_epoch{epoch}_group{first // 5}.png', dpi=300, bbox_inches='tight')
                writer.add_figure(f'diagnostics/{split}/actual_input_predictions/group{first // 5}', fig, step, close=True)
            if summary is not None:
                _log_cached_images(writer, summary, split, int(config['sparsity_percent']), step)
                arrays = preview_arrays(augmented.to(device), model.frontend, model.mean, model.std, config['sparsity_percent'])
                atomic_npz(output / f'{split}_actual_frontend_epoch{epoch}_step{step}.npz', **arrays, labels=labels, sample_ids=identities, native_scales=np.asarray(summary['native_scales']), normalized_scales=np.asarray(summary['normalized_scales']))
                for key in ('native_0', 'native_sparse', 'normalized_0', 'normalized_sparse'):
                    writer.add_histogram(f'diagnostics/{split}/actual_input/{key}', arrays[key], step)
        writer.flush()


def log_corruption_panel(model, config, writer, dataset, step, device):
    """One predetermined released-corruption panel, never chosen by prediction."""
    with preserve_diagnostic_state(model), torch.inference_mode():
        raw, labels, identities = panel_tensors(dataset)
        device = torch.device(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.get('precision') == 'bf16' and device.type == 'cuda'):
            logits = model(raw.to(device))
        predictions = logits.argmax(1).cpu().numpy()
        tag = f'corrupted_panels/{dataset.corruption}/{dataset.severity}'
        output = run_dir(config) / 'diagnostics' / 'corruptions'
        output.mkdir(parents=True, exist_ok=True)
        stem = f'{dataset.corruption}_{dataset.severity}'
        arrays = {'raw': raw.numpy()}
        if model.frontend is not None:
            _, metadata = ensure_shared_previews(config, model.frontend, model.mean, model.std, device)
            arrays = preview_arrays(raw.to(device), model.frontend, model.mean, model.std, config['sparsity_percent'])
            scales = (np.asarray(metadata['native_scales']), np.asarray(metadata['normalized_scales']))
            # One balanced 10-image panel remains fully stored. Render groups of five.
            for channel, channel_name in enumerate(model.frontend.channel_names):
                for first in (0, 5):
                    fig = signed_panel_figure(arrays, channel, scales, labels, identities, config['sparsity_percent'], channel_name, first)
                    writer.add_figure(f'{tag}/{channel_name}/group{first // 5}', fig, step, close=True)
            arrays.update(native_scales=scales[0], normalized_scales=scales[1])
        else:
            for first in (0, 5):
                fig = raw_prediction_figure(raw[first:first + 5], raw[first:first + 5], labels[first:first + 5], predictions[first:first + 5], identities[first:first + 5])
                writer.add_figure(f'{tag}/group{first // 5}', fig, step, close=True)
        atomic_npz(output / f'{stem}.npz', **arrays, labels=labels, sample_ids=identities, predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description='Prepare shared all-sparsity signed panels after verified normalization fitting')
    parser.add_argument('--dataset', choices=('mnist', 'cifar10', 'all'), default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--protocol', default='heldout_val')
    args = parser.parse_args()
    names = ('mnist', 'cifar10') if args.dataset == 'all' else (args.dataset,)
    with preserve_diagnostic_state(), torch.inference_mode():
        for name in names:
            for representation in REPRESENTATIONS:
                _, stats = load_normalization(ROOT / 'normalization' / f'{name}_{representation}_{args.protocol}.json')
                frontend = Frontend(representation).to(args.device)
                mean = torch.tensor(stats['mean'], device=args.device).view(1, -1, 1, 1)
                std = torch.tensor(stats['effective_std'], device=args.device).view(1, -1, 1, 1)
                config = {'dataset': name, 'representation': representation, 'protocol': args.protocol, 'normalization_hash': stats['artifact_sha256']}
                path, _ = ensure_shared_previews(config, frontend, mean, std, args.device)
                print(path, flush=True)


if __name__ == '__main__':
    main()
