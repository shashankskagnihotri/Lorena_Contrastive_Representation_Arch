"""Validated official data, immutable split identities, and stateless augmentation.

All images are uint8 CHW at this boundary. Only prepare_data.py discovers files;
workers consume the saved manifest and never walk the dataset root.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

PROJECT_ROOT = Path(os.environ.get('SPARSE_CONTRAST_ROOT', Path(__file__).resolve().parents[1]))
DEFAULT_DATA_ROOT = Path('/ceph/sagnihot/datasets')
CIFAR10_CORRUPTIONS = (
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
)
MNIST_CORRUPTIONS = (
    'shot_noise', 'impulse_noise', 'glass_blur', 'motion_blur', 'shear',
    'scale', 'rotate', 'brightness', 'translate', 'stripe', 'fog',
    'spatter', 'dotted_line', 'zigzag', 'canny_edges',
)
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def canonical_dataset(name):
    key = str(name).lower().replace('-', '').replace('_', '')
    if key not in ('mnist', 'cifar10'):
        raise ValueError(f'Unknown dataset: {name}')
    return key


def ids_hash(sample_ids):
    """SHA256 of UTF-8 IDs in visit order, each including one final newline."""
    return hashlib.sha256(''.join(f'{x}\n' for x in sample_ids).encode()).hexdigest()


def load_manifest(path=None, data_root=None):
    path = Path(path) if path is not None else PROJECT_ROOT / 'datasets_manifest.json'
    with path.open() as handle:
        result = json.load(handle)
    if result.get('schema_version') != 1:
        raise ValueError(f'Unsupported dataset manifest: {path}')
    if data_root is not None and Path(data_root).resolve() != Path(result['data_root']).resolve():
        raise ValueError('data_root differs from the prepared manifest; rerun prepare_data.py')
    return result


def _manifest(value=None, data_root=None):
    if isinstance(value, dict):
        if data_root is not None and Path(data_root).resolve() != Path(value['data_root']).resolve():
            raise ValueError('Manifest/data_root mismatch')
        return value
    return load_manifest(value, data_root)


def read_idx(path):
    """Read official unsigned-byte IDX, validating header and exact byte length."""
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            content = handle.read()
    else:
        content = path.read_bytes()
    if len(content) < 4 or content[:2] != b'\0\0' or content[2] != 8:
        raise ValueError(f'Invalid uint8 IDX header: {path}')
    ndim = content[3]
    if ndim not in (1, 3) or len(content) < 4 + 4 * ndim:
        raise ValueError(f'Unexpected IDX dimensions: {path}')
    shape = struct.unpack('>' + 'I' * ndim, content[4:4 + 4 * ndim])
    expected = 4 + 4 * ndim + int(np.prod(shape, dtype=np.int64))
    if len(content) != expected:
        raise ValueError(f'IDX size mismatch: {path}: {len(content)} != {expected}')
    return np.frombuffer(content, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def read_clean_arrays(name, paths, training):
    """Read only explicitly resolved official paths. CIFAR pickles must be verified."""
    name = canonical_dataset(name)
    if name == 'mnist':
        stem = 'train' if training else 'test'
        images = read_idx(paths[f'{stem}_images'])[:, None]
        labels = read_idx(paths[f'{stem}_labels'])
        return images, labels
    names = [f'data_batch_{n}' for n in range(1, 6)] if training else ['test_batch']
    batches, labels = [], []
    for batch_name in names:
        # prepare_data verifies publisher checksums before loading these official pickles.
        with (Path(paths['directory']) / batch_name).open('rb') as handle:
            content = pickle.load(handle, encoding='latin1')
        batches.append(np.asarray(content['data'], dtype=np.uint8).reshape(-1, 3, 32, 32))
        labels.extend(content['labels'])
    return np.concatenate(batches, axis=0), np.asarray(labels, dtype=np.int64)


class CleanDataset(Dataset):
    def __init__(self, name, split='train', data_root=None, protocol='heldout_val', manifest=None):
        self.name = canonical_dataset(name)
        protocol = 'full_refit' if protocol == 'full_train_refit' else protocol
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {split}')
        if protocol not in ('heldout_val', 'full_refit'):
            raise ValueError(f'Unknown protocol: {protocol}')
        if protocol == 'full_refit' and split == 'val':
            raise ValueError('full_refit has no held-out validation partition')
        self.split, self.protocol = split, protocol
        self.manifest = _manifest(manifest, data_root)
        entry = self.manifest['datasets'][self.name]
        if entry.get('status') != 'validated':
            raise RuntimeError(f'{self.name} data is not validated')
        self.images, self.all_labels = read_clean_arrays(self.name, entry['paths'], split != 'test')
        split_path = Path(entry['splits']['json'])
        with split_path.open() as handle:
            self.split_manifest = json.load(handle)
        key = 'full_train' if protocol == 'full_refit' and split == 'train' else split
        section = self.split_manifest['partitions'][key]
        self.indices = np.asarray(section['official_indices'], dtype=np.int64)
        official_split = 'test' if split == 'test' else 'train'
        self.sample_ids = [f'{self.name}/{official_split}/{int(i):05d}' for i in self.indices]
        self.split_hash = ids_hash(self.sample_ids)
        if self.split_hash != section['ids_sha256']:
            raise ValueError(f'Split identity hash mismatch: {split_path}:{key}')
        if len(np.unique(self.indices)) != len(self.indices) or np.any(self.indices < 0) or np.any(self.indices >= len(self.images)):
            raise ValueError('Invalid or duplicate sample IDs in split')
        self.labels = self.all_labels[self.indices]
        self.targets = self.labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        # Copy protects immutable dataset storage from downstream in-place augmentation.
        return torch.from_numpy(self.images[source_index].copy()), int(self.all_labels[source_index]), self.sample_ids[index]


def load_clean(name, split='train', data_root=None, protocol='heldout_val', manifest=None):
    return CleanDataset(name, split, data_root, protocol, manifest)


def get_dataset(name, split='train', **kwargs):
    return load_clean(name, split=split, **kwargs)


def corruption_cells(name):
    name = canonical_dataset(name)
    if name == 'mnist':
        return [(corruption, 'fixed') for corruption in MNIST_CORRUPTIONS]
    return [(corruption, severity) for corruption in CIFAR10_CORRUPTIONS for severity in range(1, 6)]


class CorruptionDataset(Dataset):
    """One official 10,000-image cell, mmap-backed, with no clean-image access."""
    def __init__(self, name, corruption, severity, data_root=None, manifest=None):
        self.name = canonical_dataset(name)
        if (corruption, severity) not in corruption_cells(self.name):
            raise ValueError(f'Nonprimary/invalid corruption cell: {self.name}/{corruption}/{severity}')
        self.corruption, self.severity = corruption, severity
        self.manifest = _manifest(manifest, data_root)
        entry = self.manifest['datasets'][self.name + '_c']
        if entry.get('status') != 'validated':
            raise RuntimeError(f'{self.name}-C has not been validated: {entry.get("status")}')
        self.directory = Path(entry['directory'])
        if self.name == 'mnist':
            self.images = np.load(self.directory / corruption / 'test_images.npy', mmap_mode='r', allow_pickle=False)
            self.labels = np.load(self.directory / corruption / 'test_labels.npy', mmap_mode='r', allow_pickle=False)
            self.offset = 0
            if self.images.shape != (10000, 28, 28, 1) or self.labels.shape != (10000,):
                raise ValueError('MNIST-C cell no longer matches validated shape')
        else:
            self.images = np.load(self.directory / f'{corruption}.npy', mmap_mode='r', allow_pickle=False)
            full_labels = np.load(self.directory / 'labels.npy', mmap_mode='r', allow_pickle=False)
            self.offset = (int(severity) - 1) * 10000
            layout = entry['label_layout']
            if layout == '50000_repeated_clean_test':
                self.labels = full_labels[self.offset:self.offset + 10000]
            elif layout == '10000_shared_clean_test':
                self.labels = full_labels
            else:
                raise ValueError(f'Unverified CIFAR-10-C label layout: {layout}')
            if self.images.shape != (50000, 32, 32, 3) or self.labels.shape != (10000,):
                raise ValueError('CIFAR-10-C cell no longer matches validated shape')
        if self.images.dtype != np.uint8:
            raise ValueError('Corruption image dtype changed from validated uint8')
        self.sample_ids = [f'{self.name}_c/{corruption}/{severity}/{i:05d}' for i in range(10000)]
        self.split_hash = ids_hash(self.sample_ids)
        self.targets = self.labels

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        if not 0 <= index < 10000:
            raise IndexError(index)
        image = self.images[self.offset + index]
        return torch.from_numpy(np.asarray(image).transpose(2, 0, 1).copy()), int(self.labels[index]), self.sample_ids[index]


def keyed_seed(namespace, seed, epoch, identity):
    key = f'sparse_contrast_v1|{namespace}|{int(seed)}|{int(epoch)}|{identity}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')


class StatelessAugment(Dataset):
    """CIFAR zero-pad 4, uniform crop, horizontal flip, keyed by sample/epoch.

    Set epoch before creating each epoch's DataLoader iterator. Use nonpersistent
    workers so their copies receive the epoch. MNIST passes through unchanged.
    """
    def __init__(self, dataset, seed):
        if dataset.split != 'train':
            raise ValueError('Training augmentation can only wrap the train partition')
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0
        self.name = dataset.name
        self.sample_ids, self.split_hash = dataset.sample_ids, dataset.split_hash
        self.indices, self.labels, self.targets = dataset.indices, dataset.labels, dataset.labels

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, identity = self.dataset[index]
        if self.name == 'cifar10':
            rng = np.random.Generator(np.random.PCG64(keyed_seed('augmentation', self.seed, self.epoch, identity)))
            top, left = rng.integers(0, 9, size=2)
            padded = torch.nn.functional.pad(image, (4, 4, 4, 4), mode='constant', value=0)
            image = padded[:, int(top):int(top) + 32, int(left):int(left) + 32]
            if rng.integers(0, 2):
                image = image.flip(-1)
        return image.contiguous(), target, identity


class EpochSampler(Sampler):
    """Sample order independent of model and augmentation RNG; epoch-boundary resume."""
    def __init__(self, dataset, seed, shuffle=True):
        self.dataset, self.seed, self.shuffle, self.epoch = dataset, int(seed), bool(shuffle), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.dataset)))
        rng = np.random.Generator(np.random.PCG64(keyed_seed('sample_order', self.seed, self.epoch, len(self.dataset))))
        return iter(rng.permutation(len(self.dataset)).tolist())

    def __len__(self):
        return len(self.dataset)


def fixed_panel_indices(dataset, panel='diagnostic'):
    """Return positions in dataset for predetermined class-balanced saved panels."""
    if isinstance(dataset, StatelessAugment):
        dataset = dataset.dataset
    if isinstance(dataset, CorruptionDataset):
        manifest = load_manifest()
        with Path(manifest['datasets'][dataset.name]['splits']['json']).open() as handle:
            split = json.load(handle)
        return split['panels']['test'][panel]['official_indices']
    section = dataset.split_manifest['panels'][dataset.split][panel]
    lookup = {int(identity): index for index, identity in enumerate(dataset.indices)}
    return [lookup[int(identity)] for identity in section['official_indices']]
