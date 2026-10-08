"""Display-only mappings preserve signed scientific tensors, state, and identities."""
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from sparse_contrast.frontend import Frontend, patch_support
from sparse_contrast.visualize import (
    atomic_npz, confusion_figure, preserve_diagnostic_state, preview_arrays,
    signed_display_scale, signed_panel_figure,
)


def test_diagnostics_restore_all_rngs_and_module_modes():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train(); model[0].eval()
    model.latest_diagnostics = {'original': 1}
    model.collect_diagnostics = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    modes = [module.training for module in model.modules()]
    with preserve_diagnostic_state(model):
        assert not model.training
        random.random(); np.random.rand(3); torch.rand(4)
        model.latest_diagnostics = {'changed': 2}
        model[0].train()
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert [module.training for module in model.modules()] == modes
    assert model.latest_diagnostics == {'original': 1}
    assert model.collect_diagnostics is True


def test_signed_preview_masks_and_affine_zero_offsets(tmp_path):
    raw = torch.zeros(2, 1, 28, 28, dtype=torch.uint8)
    raw[0, :, 12:15, 12:15] = 255
    raw[1, :, 9:12, 9:12] = 140
    frontend = Frontend('grayscale')
    mean, std = torch.tensor([0.2]).view(1, 1, 1, 1), torch.tensor([0.3]).view(1, 1, 1, 1)
    original = raw.clone()
    arrays = preview_arrays(raw, frontend, mean, std, 80)
    assert np.any(arrays['native_0'] < 0) and np.any(arrays['native_0'] > 0)
    assert np.array_equal(arrays['patch_support'], patch_support(torch.from_numpy(arrays['native_sparse'])).numpy())
    zero = arrays['native_sparse'] == 0
    assert np.allclose(arrays['normalized_sparse'][zero], -0.2 / 0.3)
    assert torch.equal(original, raw)
    c0 = torch.from_numpy(arrays['native_0'])
    scales = signed_display_scale(c0, mean, std)
    assert scales[1][0] >= abs(0.2 / 0.3)
    saved = tmp_path / 'panel.npz'
    ids = np.asarray(['mnist/train/00001', 'mnist/train/00002'])
    atomic_npz(saved, **arrays, sample_ids=ids)
    loaded = np.load(saved, allow_pickle=False)
    assert np.array_equal(loaded['sample_ids'], ids)
    assert np.array_equal(loaded['native_0'], arrays['native_0'])
    fig = signed_panel_figure(arrays, 0, scales, [0, 1], ids, 80, 'mean_rgb_contrast')
    fig.savefig(tmp_path / 'panel.png', dpi=80)
    # Rendering never changes the signed values supplied to the model.
    assert np.array_equal(loaded['native_0'], arrays['native_0'])
    plt.close(fig)


def test_confusion_figure_has_count_units():
    counts = np.eye(10, dtype=np.int64)
    fig = confusion_figure(counts, 'Held-out validation')
    assert fig.axes[0].get_xlabel() == 'Predicted class'
    assert fig.axes[0].get_ylabel() == 'True class'
    plt.close(fig)


def test_preview_recovery_never_steals_legacy_or_live_lease(tmp_path, monkeypatch):
    import json
    from sparse_contrast import campaign
    from sparse_contrast.visualize import recover_preview_lock
    lock = tmp_path / 'preview.lock'
    lock.mkdir()
    owner = lock / 'owner.json'
    owner.write_text(json.dumps({'pid': 999999999, 'host': 'remote'}))
    assert recover_preview_lock(lock) is False
    assert lock.exists()
    owner.write_text(json.dumps({'pid': 999999999, 'host': 'remote', 'job_id': '999'}))
    monkeypatch.setattr(campaign, 'job_states', lambda jobs: {'999': {'state': 'RUNNING'}})
    assert recover_preview_lock(lock) is False
    assert lock.exists()
    preview = tmp_path / 'preserved_preview.npz'
    preview.write_bytes(b'valuable existing artifact')
    monkeypatch.setattr(campaign, 'job_states', lambda jobs: {'999': {'state': 'COMPLETED'}})
    assert recover_preview_lock(lock) is True
    assert not lock.exists()
    assert len(list(tmp_path.glob('preview.lock.retired.*'))) == 1
    assert preview.read_bytes() == b'valuable existing artifact'
