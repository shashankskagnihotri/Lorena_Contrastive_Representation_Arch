"""Protocol and real official-data integrity checks required by the study."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sparse_contrast.data import (
    CIFAR10_CORRUPTIONS, MNIST_CORRUPTIONS, CorruptionDataset, EpochSampler,
    StatelessAugment, corruption_cells, fixed_panel_indices, ids_hash,
    keyed_seed, load_clean, load_manifest, read_idx,
)
from prepare_data import make_splits, safe_extract


def test_official_corruption_registry():
    assert CIFAR10_CORRUPTIONS == ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')
    assert MNIST_CORRUPTIONS == ('shot_noise', 'impulse_noise', 'glass_blur', 'motion_blur', 'shear', 'scale', 'rotate', 'brightness', 'translate', 'stripe', 'fog', 'spatter', 'dotted_line', 'zigzag', 'canny_edges')
    assert len(corruption_cells('mnist')) == 15
    assert len(corruption_cells('cifar10')) == 75
    assert all(severity == 'fixed' for _, severity in corruption_cells('mnist'))
    with pytest.raises(ValueError):
        CorruptionDataset('mnist', 'shot_noise', 1)
    with pytest.raises(ValueError):
        CorruptionDataset('cifar10', 'saturate', 1)


def test_idx_rejects_truncated_and_invalid(tmp_path):
    path = tmp_path / 'bad-idx'
    path.write_bytes(b'\0\0\x08\x01\0\0\0\x02\x01')
    with pytest.raises(ValueError, match='size mismatch'):
        read_idx(path)
    path.write_bytes(b'\0\0\x09\x01\0\0\0\x02\x01\x02')
    with pytest.raises(ValueError, match='header'):
        read_idx(path)


def test_keyed_augmentation_no_rng_side_effect():
    class RealShapeDataset:
        name, split, indices, labels = 'cifar10', 'train', np.arange(2), np.arange(2)
        sample_ids = ['cifar10/train/00000', 'cifar10/train/00001']
        split_hash = ids_hash(sample_ids)
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return torch.arange(3 * 32 * 32).reshape(3, 32, 32).to(torch.uint8), index, self.sample_ids[index]
    wrapped = StatelessAugment(RealShapeDataset(), 4)
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    first = wrapped[0][0]
    wrapped[1]
    torch.rand(5)
    assert torch.equal(first, wrapped[0][0])
    torch.set_rng_state(torch_state)
    wrapped.set_epoch(1)
    assert not torch.equal(first, wrapped[0][0])
    assert torch.equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and np.array_equal(numpy_state[1], after[1]) and numpy_state[2:] == after[2:]
    assert keyed_seed('augmentation', 1, 2, 'i') != keyed_seed('sample_order', 1, 2, 'i')
    sampler = EpochSampler(wrapped, 4)
    before = list(sampler)
    torch.rand(10)
    assert list(sampler) == before
    assert sorted(before) == [0, 1]


@pytest.mark.parametrize('dataset,ntrain', [('mnist', 55000), ('cifar10', 45000)])
def test_real_clean_partitions_and_panels(dataset, ntrain):
    train, val, test = [load_clean(dataset, split) for split in ('train', 'val', 'test')]
    assert [len(train), len(val), len(test)] == [ntrain, 5000, 10000]
    assert not set(train.sample_ids).intersection(val.sample_ids)
    assert len(set(train.sample_ids) | set(val.sample_ids)) == ntrain + 5000
    assert train.split_hash == ids_hash(train.sample_ids)
    for partition in (train, val, test):
        panel = fixed_panel_indices(partition)
        assert [partition[i][1] for i in panel] == list(range(10))
        x, label, identity = partition[0]
        assert x.dtype == torch.uint8
        assert x.shape == ((1, 28, 28) if dataset == 'mnist' else (3, 32, 32))
        assert identity == partition.sample_ids[0]
        assert label == partition.labels[0]
    assert len(load_clean(dataset, protocol='full_refit')) == ntrain + 5000
    with pytest.raises(ValueError):
        load_clean(dataset, 'val', protocol='full_refit')


def test_real_corruption_coverage_and_severity_slices():
    manifest = load_manifest()
    for name in ('mnist', 'cifar10'):
        labels = load_clean(name, 'test').labels
        cells = corruption_cells(name)
        assert manifest['datasets'][name + '_c']['status'] == 'validated'
        assert len(manifest['datasets'][name + '_c']['cells']) == len(cells)
        seen = set()
        for corruption, severity in cells:
            data = CorruptionDataset(name, corruption, severity)
            assert len(data) == 10000
            assert np.array_equal(data.labels, labels)
            assert not seen.intersection(data.sample_ids)
            seen.update(data.sample_ids)
            for index in (0, 9999):
                image, label, identity = data[index]
                assert image.dtype == torch.uint8 and label == labels[index]
                assert identity.endswith(f'/{index:05d}')
            if name == 'cifar10':
                assert data.offset == (severity - 1) * 10000
                assert np.array_equal(data[0][0].numpy().transpose(1, 2, 0), data.images[data.offset])
        assert len(seen) == (150000 if name == 'mnist' else 750000)


def test_proportional_split_reproduction(tmp_path):
    labels = np.repeat(np.arange(10), [5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5455])
    test_labels = np.repeat(np.arange(10), 1000)
    result = make_splits('mnist', labels, test_labels, tmp_path)
    saved = json.loads(Path(result['json']).read_text())
    train = set(saved['partitions']['train']['official_indices'])
    val = set(saved['partitions']['val']['official_indices'])
    assert len(val) == 5000 and not train.intersection(val)
    assert train | val == set(range(len(labels)))
    assert sum(saved['partitions']['val']['class_counts']) == 5000
    repeated = make_splits('mnist', labels, test_labels, tmp_path)
    assert result['json_sha256'] == repeated['json_sha256']


def test_archive_path_traversal_rejected(tmp_path):
    import zipfile
    archive = tmp_path / 'attack.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('mnist_c/../../outside', 'unsafe')
    with pytest.raises(ValueError, match='Unsafe archive'):
        safe_extract(archive, tmp_path, 'mnist_c')
    assert not (tmp_path.parent / 'outside').exists()
