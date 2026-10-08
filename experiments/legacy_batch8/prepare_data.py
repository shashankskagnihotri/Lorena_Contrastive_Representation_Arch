#!/usr/bin/env python3
"""Resolve, verify, and prepare immutable official data and held-out splits.

No training worker performs discovery. Originals are never changed. Archive
extraction is checksummed, path-safe, and installed through a single rename.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time
import uuid
import zipfile

import numpy as np
from sparse_contrast.data import (
    PROJECT_ROOT, DEFAULT_DATA_ROOT, MNIST_CORRUPTIONS, CIFAR10_CORRUPTIONS,
    read_clean_arrays, read_idx, ids_hash,
)

ARCHIVES = {
    'mnist_train_images': ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873', 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz'),
    'mnist_train_labels': ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432', 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz'),
    'mnist_test_images': ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3', 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz'),
    'mnist_test_labels': ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c', 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'),
    'cifar10': ('cifar-10-python.tar.gz', 'c58f30108f718f92721af3b95e74349a', 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'),
    'mnist_c': ('mnist_c.zip', '4b34b33045869ee6d424616cd3a65da3', 'https://zenodo.org/records/3239543/files/mnist_c.zip?download=1'),
    'cifar10_c': ('CIFAR-10-C.tar', '56bf5dcef84df0e2308c6dcbcbbd8499', 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1'),
}
CIFAR_FILES = {
    'data_batch_1': 'c99cafc152244af753f735de768cd75f',
    'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
    'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
    'data_batch_4': '634d18415352ddfa80567beed471001a',
    'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
    'test_batch': '40351d587109b95175f43aff81a1287e',
    'batches.meta': '5ff9c542aee3614f3951f8cda6e48888',
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with temporary.open('w') as handle:
        json.dump(content, handle, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def directory_lock(path):
    """Atomic mkdir on Ceph; never steal an existing cross-node lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f'Preparation lock already held: {path}; inspect owner.json before manual recovery') from exc
    atomic_json(path / 'owner.json', {'pid': os.getpid(), 'host': os.uname().nodename, 'started': now()})
    try:
        yield
    finally:
        (path / 'owner.json').unlink()
        path.rmdir()


def file_hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            md5.update(chunk)
            sha.update(chunk)
    return {'path': str(Path(path).resolve()), 'bytes': Path(path).stat().st_size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}


def verify_archive(path, expected):
    info = file_hashes(path)
    info['expected_md5'] = expected
    info['publisher_md5_verified'] = info['md5'] == expected
    if not info['publisher_md5_verified']:
        raise ValueError(f'Publisher MD5 mismatch, original preserved: {path}; got {info["md5"]}, expected {expected}')
    return info


def download_archive(key, destination):
    filename, checksum, url = ARCHIVES[key]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / filename
    with directory_lock(destination / f'.{filename}.download.lock'):
        if final.exists():
            return final, verify_archive(final, checksum)
        if shutil.disk_usage(destination).free < 8 * 1024**3:
            raise RuntimeError(f'Less than 8 GiB free for download/extraction at {destination}')
        state_path = PROJECT_ROOT / 'provenance' / f'datasets_download_{key}.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'url': url, 'destination': str(final), 'attempts': []}
        if len(state['attempts']) >= 4:
            raise RuntimeError(f'Automatic retry limit exhausted for {key}; inspect {state_path}')
        for attempt in range(len(state['attempts']) + 1, 5):
            partial = destination / f'.{filename}.attempt-{attempt}.{uuid.uuid4().hex}.partial'
            record = {'number': attempt, 'started': now(), 'partial': str(partial)}
            state['attempts'].append(record)
            atomic_json(state_path, state)
            result = subprocess.run(['curl', '--location', '--fail', '--show-error', '--connect-timeout', '30', '--max-time', '1800', '--output', str(partial), url], check=False)
            record['ended'], record['returncode'] = now(), result.returncode
            if result.returncode == 0:
                info = verify_archive(partial, checksum)
                os.rename(partial, final)
                info['path'] = str(final.resolve())
                record['status'] = 'verified'
                state['verified'] = info
                atomic_json(state_path, state)
                return final, info
            record['status'] = 'download_failed'
            atomic_json(state_path, state)
            # curl transient DNS/connect/partial/timeout/receive errors only; HTTP
            # status failure is recorded and needs explicit root-cause inspection.
            if result.returncode not in (5, 6, 7, 18, 28, 52, 55, 56) or attempt == 4:
                raise RuntimeError(f'Download failed ({result.returncode}): {url}; evidence in {state_path}')
            time.sleep(min(30, 2**attempt))
    raise RuntimeError('Download unexpectedly ended without verified archive')


def safe_extract(archive, destination, top_directory):
    destination = Path(destination)
    target = destination / top_directory
    with directory_lock(destination / f'.{top_directory}.extract.lock'):
        if target.exists():
            return target
        temporary = destination / f'.{top_directory}.extract.{uuid.uuid4().hex}'
        temporary.mkdir()
        def check_member(name, is_link=False):
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or is_link or not relative.parts or relative.parts[0] != top_directory:
                raise ValueError(f'Unsafe archive member: {name}')
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                members = handle.infolist()
                for member in members:
                    check_member(member.filename, stat.S_ISLNK(member.external_attr >> 16))
                required = sum(member.file_size for member in members)
                if shutil.disk_usage(destination).free < required + 1024**3:
                    raise RuntimeError('Insufficient space for complete extraction')
                handle.extractall(temporary)
        else:
            with tarfile.open(archive, 'r:*') as handle:
                members = handle.getmembers()
                for member in members:
                    check_member(member.name, not (member.isfile() or member.isdir()))
                required = sum(member.size for member in members)
                if shutil.disk_usage(destination).free < required + 1024**3:
                    raise RuntimeError('Insufficient space for complete extraction')
                handle.extractall(temporary, members=members)
        os.rename(temporary / top_directory, target)
        temporary.rmdir()
    return target


def discover(root, refresh=False):
    """Inspect root and only relevant common layouts, once per saved inventory."""
    inventory_path = PROJECT_ROOT / 'provenance' / 'datasets_inventory.json'
    if inventory_path.exists() and not refresh:
        result = json.loads(inventory_path.read_text())
        if result['data_root'] != str(root.resolve()):
            raise ValueError('Inventory belongs to another root; use --refresh-inventory')
        return result
    if not root.is_dir():
        raise FileNotFoundError(root)
    # Unrelated datasets are not traversed. These layouts cover torchvision,
    # unpacked originals, and common lowercase benchmark directories.
    directories = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(term in child.name.lower() for term in ('mnist', 'cifar-10', 'cifar10')) and '100' not in child.name:
            directories.append(child)
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and any(term in sub.name.lower() for term in ('raw', 'mnist', 'cifar', 'download')):
                    directories.append(sub)
                    for leaf in sorted(sub.iterdir()):
                        if leaf.is_dir() and leaf.name.lower() == 'raw':
                            directories.append(leaf)
    paths = []
    for directory in dict.fromkeys(directories):
        for path in sorted(directory.iterdir()):
            if path.is_file() and (path.suffix in ('.gz', '.zip', '.tar', '.npy') or path.name.startswith(('train-', 't10k-', 'data_batch', 'test_batch', 'batches.meta'))):
                if directory == root and not any(term in path.name.lower() for term in ('mnist', 'cifar-10', 'cifar10', 'idx')):
                    continue
                paths.append({'path': str(path.resolve()), 'bytes': path.stat().st_size})
    result = {'data_root': str(root.resolve()), 'created': now(), 'directories': [str(x.resolve()) for x in dict.fromkeys(directories)], 'files': paths, 'policy': 'One targeted inventory; do not traverse unrelated datasets; workers read manifest only.'}
    atomic_json(inventory_path, result)
    return result


def first_file(inventory, name):
    matches = [Path(item['path']) for item in inventory['files'] if Path(item['path']).name == name]
    return min(matches, key=lambda x: (len(x.parts), str(x))) if matches else None


def first_directory(inventory, required_filename):
    path = first_file(inventory, required_filename)
    return path.parent if path else None


def validate_images(images, labels, count, chw):
    if images.shape != (count, *chw) or images.dtype != np.uint8:
        raise ValueError(f'Invalid images: shape={images.shape}, dtype={images.dtype}, expected {(count, *chw)} uint8')
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'Invalid labels: {labels.shape}, {labels.dtype}')
    counts = np.bincount(labels.astype(np.int64), minlength=10)
    if labels.min() != 0 or labels.max() != 9 or len(counts) != 10 or np.any(counts == 0):
        raise ValueError('Labels do not cover exactly classes 0..9')
    return {'count': count, 'shape': list(images.shape), 'layout': 'NCHW', 'dtype': str(images.dtype), 'pixel_min': int(images.min()), 'pixel_max': int(images.max()), 'class_counts': counts.tolist()}


def make_splits(name, labels, test_labels, output):
    """Proportional stratification with largest remainders, ties by class ID."""
    rng = np.random.Generator(np.random.PCG64(2026))
    counts = np.bincount(labels.astype(np.int64), minlength=10)
    ideals = counts * (5000 / len(labels))
    allocations = np.floor(ideals).astype(np.int64)
    order = np.lexsort((np.arange(10), -(ideals - allocations)))
    allocations[order[:5000 - int(allocations.sum())]] += 1
    val = []
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        val.extend(rng.permutation(candidates)[:int(allocations[label])].tolist())
    val = np.asarray(sorted(val), dtype=np.int64)
    train = np.setdiff1d(np.arange(len(labels), dtype=np.int64), val, assume_unique=True)
    if len(val) != 5000 or np.intersect1d(train, val).size or len(train) + len(val) != len(labels):
        raise AssertionError('Split coverage/isolation failed')
    partitions = {}
    arrays = {'train': train, 'val': val, 'full_train': np.arange(len(labels)), 'test': np.arange(len(test_labels))}
    for split, indices in arrays.items():
        official = 'test' if split == 'test' else 'train'
        selected_labels = test_labels if split == 'test' else labels
        identities = [f'{name}/{official}/{i:05d}' for i in indices]
        partitions[split] = {'count': len(indices), 'official_indices': indices.tolist(), 'ids_sha256': ids_hash(identities), 'class_counts': np.bincount(selected_labels[indices].astype(np.int64), minlength=10).tolist()}
    # Independent fixed panels choose no examples based on output or timing.
    panels = {}
    for split in ('train', 'val', 'test'):
        panel_rng = np.random.Generator(np.random.PCG64(9421 + {'train': 0, 'val': 1, 'test': 2}[split]))
        original = arrays[split]
        selected_labels = test_labels if split == 'test' else labels
        by_class = [panel_rng.permutation(original[selected_labels[original] == label]) for label in range(10)]
        panels[split] = {}
        for panel, per_class in [('diagnostic', 1), ('timing', 160)]:
            chosen = np.asarray([int(by_class[label][number]) for number in range(min(per_class, min(map(len, by_class)))) for label in range(10)], dtype=np.int64)
            official = 'test' if split == 'test' else 'train'
            identities = [f'{name}/{official}/{i:05d}' for i in chosen]
            panels[split][panel] = {'official_indices': chosen.tolist(), 'sample_ids': identities, 'ids_sha256': ids_hash(identities), 'selection': 'class interleaved, PCG64 seed 9421+split_index, no model information', 'per_class': len(chosen) // 10}
    result = {'schema_version': 1, 'dataset': name, 'protocol': 'heldout_val', 'split_seed': 2026, 'algorithm': 'PCG64(seed=2026); class ascending; per-class permutation; proportional largest-remainder allocation to exactly 5000 validation images; output indices sorted', 'partitions': partitions, 'panels': panels}
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f'{name}_seed2026.json'
    if json_path.exists() and json.loads(json_path.read_text()) != result:
        raise ValueError(f'Refusing to change frozen split artifact: {json_path}')
    atomic_json(json_path, result)
    npz_path = output / f'{name}_seed2026.npz'
    temporary = npz_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, npz_path)
    return {'json': str(json_path.resolve()), 'npz': str(npz_path.resolve()), 'json_sha256': file_hashes(json_path)['sha256'], 'train_ids_sha256': partitions['train']['ids_sha256'], 'val_ids_sha256': partitions['val']['ids_sha256'], 'test_ids_sha256': partitions['test']['ids_sha256']}


def array_info(path, shape):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != shape or values.dtype != np.uint8:
        raise ValueError(f'Unexpected corruption array {path}: {values.shape}/{values.dtype}, expected {shape}/uint8')
    info = file_hashes(path)
    info.update({'shape': list(shape), 'dtype': str(values.dtype), 'pixel_min': int(values.min()), 'pixel_max': int(values.max()), 'layout': 'NHWC'})
    return values, info


def prepare(root, download=False, clean_only=False, refresh_inventory=False):
    inventory = discover(root, refresh_inventory)
    manifest_path = PROJECT_ROOT / 'datasets_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {'schema_version': 1, 'data_root': str(root.resolve()), 'prepared_at': now(), 'inventory': str(PROJECT_ROOT / 'provenance' / 'datasets_inventory.json'), 'datasets': {}, 'archive_checksums': {key: {'filename': value[0], 'md5': value[1], 'source': value[2]} for key, value in ARCHIVES.items()}}
    clean = {}
    mnist_paths, mnist_files = {}, {}
    for split in ('train', 'test'):
        for kind in ('images', 'labels'):
            key = f'mnist_{split}_{kind}'
            filename, checksum, _ = ARCHIVES[key]
            compressed = first_file(inventory, filename)
            original = first_file(inventory, filename[:-3])
            if compressed is None and original is None:
                if not download:
                    raise FileNotFoundError(f'Missing {filename}; rerun with --download')
                compressed, _ = download_archive(key, root / 'mnist' / 'MNIST' / 'raw')
            if compressed:
                verified = verify_archive(compressed, checksum)
                if original:
                    if not np.array_equal(read_idx(original), read_idx(compressed)):
                        raise ValueError(f'Extracted IDX differs from verified archive: {original}')
                    verified['extracted_bytes_verified_equal'] = True
                    verified['extracted'] = file_hashes(original)
                mnist_files[key] = verified
            else:
                mnist_files[key] = {**file_hashes(original), 'publisher_md5_verified': False, 'integrity_limitation': 'No original gzip found; structure and label checks only'}
            mnist_paths[f'{split}_{kind}'] = str((original or compressed).resolve())
    train_x, train_y = read_clean_arrays('mnist', mnist_paths, True)
    test_x, test_y = read_clean_arrays('mnist', mnist_paths, False)
    clean['mnist'] = (train_y, test_y)
    manifest['datasets']['mnist'] = {'status': 'validated', 'paths': mnist_paths, 'files': mnist_files, 'train': validate_images(train_x, train_y, 60000, (1, 28, 28)), 'test': validate_images(test_x, test_y, 10000, (1, 28, 28)), 'splits': make_splits('mnist', train_y, test_y, PROJECT_ROOT / 'outputs' / 'splits')}
    del train_x, test_x
    cifar_dir = first_directory(inventory, 'data_batch_1')
    cifar_archive = first_file(inventory, ARCHIVES['cifar10'][0])
    if cifar_dir is None:
        if cifar_archive is None:
            if not download:
                raise FileNotFoundError('Missing CIFAR-10; rerun with --download')
            cifar_archive, _ = download_archive('cifar10', root)
        verify_archive(cifar_archive, ARCHIVES['cifar10'][1])
        cifar_dir = safe_extract(cifar_archive, root, 'cifar-10-batches-py')
    cifar_files = {name: verify_archive(cifar_dir / name, md5) for name, md5 in CIFAR_FILES.items()}
    archive_info = verify_archive(cifar_archive, ARCHIVES['cifar10'][1]) if cifar_archive else {'publisher_md5_verified': False, 'integrity_limitation': 'Archive absent, all extracted official batch checksums verified'}
    paths = {'directory': str(cifar_dir.resolve())}
    train_x, train_y = read_clean_arrays('cifar10', paths, True)
    test_x, test_y = read_clean_arrays('cifar10', paths, False)
    clean['cifar10'] = (train_y, test_y)
    manifest['datasets']['cifar10'] = {'status': 'validated', 'paths': paths, 'files': cifar_files, 'archive': archive_info, 'train': validate_images(train_x, train_y, 50000, (3, 32, 32)), 'test': validate_images(test_x, test_y, 10000, (3, 32, 32)), 'splits': make_splits('cifar10', train_y, test_y, PROJECT_ROOT / 'outputs' / 'splits')}
    del train_x, test_x
    # Persist clean availability before a potentially slow corruption download.
    for key in ('mnist_c', 'cifar10_c'):
        manifest['datasets'][key] = previous.get('datasets', {}).get(key, {'status': 'not_prepared'})
    atomic_json(manifest_path, manifest)
    if clean_only:
        return manifest
    for name, dirname, corruptions in [('mnist', 'mnist_c', MNIST_CORRUPTIONS), ('cifar10', 'CIFAR-10-C', CIFAR10_CORRUPTIONS)]:
        key = name + '_c'
        candidates = [Path(value) for value in inventory['directories'] if Path(value).name.lower() == dirname.lower()]
        directory = min(candidates, key=lambda x: len(x.parts)) if candidates else (root / dirname if (root / dirname).is_dir() else None)
        archive = first_file(inventory, ARCHIVES[key][0])
        if archive is None and (root / ARCHIVES[key][0]).exists():
            archive = root / ARCHIVES[key][0]
        if directory is None:
            if archive is None:
                if not download:
                    manifest['datasets'][key] = {'status': 'missing', 'reason': 'Official dataset not found; rerun prepare_data.py --download'}
                    atomic_json(manifest_path, manifest)
                    continue
                archive, _ = download_archive(key, root)
            verified_archive = verify_archive(archive, ARCHIVES[key][1])
            directory = safe_extract(archive, root, dirname)
        else:
            verified_archive = verify_archive(archive, ARCHIVES[key][1]) if archive else {'publisher_md5_verified': False, 'integrity_limitation': 'Extracted arrays only; archive absent'}
        entry = {'status': 'validating', 'directory': str(directory.resolve()), 'archive': verified_archive, 'primary_corruptions': list(corruptions), 'cells': {}, 'expected_predictions_per_model': 150000 if name == 'mnist' else 750000}
        manifest['datasets'][key] = entry
        atomic_json(manifest_path, manifest)
        expected_labels = clean[name][1]
        if name == 'cifar10':
            labels_path = directory / 'labels.npy'
            labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
            if labels.shape == (50000,):
                for severity in range(1, 6):
                    if not np.array_equal(labels[(severity - 1) * 10000:severity * 10000], expected_labels):
                        raise ValueError(f'CIFAR-10-C labels mismatch clean test ordering at severity {severity}')
                entry['label_layout'] = '50000_repeated_clean_test'
            elif labels.shape == (10000,) and np.array_equal(labels, expected_labels):
                entry['label_layout'] = '10000_shared_clean_test'
            else:
                raise ValueError(f'Invalid CIFAR-10-C labels/order: {labels.shape}')
            entry['labels'] = {**file_hashes(labels_path), 'shape': list(labels.shape), 'dtype': str(labels.dtype), 'all_severity_labels_equal_clean_test': True}
        for corruption in corruptions:
            if name == 'mnist':
                images_path, labels_path = directory / corruption / 'test_images.npy', directory / corruption / 'test_labels.npy'
                images, info = array_info(images_path, (10000, 28, 28, 1))
                labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
                if labels.shape != (10000,) or not np.issubdtype(labels.dtype, np.integer) or not np.array_equal(labels, expected_labels):
                    raise ValueError(f'MNIST-C labels mismatch clean test ordering: {corruption}')
                info.update({'labels': file_hashes(labels_path), 'severity': 'fixed', 'count': 10000, 'labels_equal_clean_test': True})
                entry['cells'][f'{corruption}/fixed'] = info
            else:
                images, info = array_info(directory / f'{corruption}.npy', (50000, 32, 32, 3))
                for severity in range(1, 6):
                    entry['cells'][f'{corruption}/{severity}'] = {**info, 'severity': severity, 'start': (severity - 1) * 10000, 'stop': severity * 10000, 'count': 10000}
            del images
        expected_cells = 15 if name == 'mnist' else 75
        if len(entry['cells']) != expected_cells or sum(cell['count'] for cell in entry['cells'].values()) != entry['expected_predictions_per_model']:
            raise AssertionError('Incomplete official corruption coverage')
        entry['status'] = 'validated'
        entry['validated_at'] = now()
        entry['excluded_extra_corruptions'] = sorted({p.stem if p.is_file() else p.name for p in directory.iterdir() if (p.is_dir() or p.suffix == '.npy') and p.stem != 'labels'} - set(corruptions))
        atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--refresh-inventory', action='store_true')
    args = parser.parse_args()
    with directory_lock(PROJECT_ROOT / 'provenance' / 'datasets_preparation.lock'):
        result = prepare(args.data_root, args.download, args.clean_only, args.refresh_inventory)
    print(json.dumps({key: value['status'] for key, value in result['datasets'].items()}, sort_keys=True))
    if not args.clean_only and any(value['status'] != 'validated' for value in result['datasets'].values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
