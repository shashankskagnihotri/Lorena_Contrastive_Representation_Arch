import json

import pytest
import torch

from sparse_contrast.frontend import patch_support, patchify, sparsify
from sparse_contrast.normalization import (
    FrozenNormalization, PopulationMoments, artifact_lock, check_standardized_moments,
    load_normalization, validate_sample_ids, write_artifact,
)


def test_population_chan_unequal_batches_including_native_zeros():
    generator = torch.Generator().manual_seed(33)
    coeff = torch.randn(11, 3, 6, 4, generator=generator) * torch.tensor([1, 0.5, 1.5])[None, :, None, None]
    coeff[:2] = 0
    coeff[:, 1] = 0.25  # a guarded constant nonzero channel
    accum = PopulationMoments(3)
    for batch in (coeff[:3], coeff[3:10], coeff[10:]):
        accum.update(batch)
    stats = accum.finalize(expected_images=11, expected_pixels=11 * 24)
    direct = coeff.double().permute(1, 0, 2, 3).reshape(3, -1)
    torch.testing.assert_close(torch.tensor(stats["mean"], dtype=torch.float64), direct.mean(1), rtol=0, atol=2e-15)
    torch.testing.assert_close(torch.tensor(stats["sigma"], dtype=torch.float64), direct.std(1, correction=0), rtol=0, atol=2e-15)
    assert stats["guarded_channels"] == [False, True, False]
    assert stats["effective_std"][1] == 1.0
    assert stats["natural_zero_counts"] == [48, 0, 48]
    normalized = FrozenNormalization(stats["mean"], stats["effective_std"])(coeff)
    verification = PopulationMoments(3)
    verification.update(normalized)
    check_standardized_moments(verification.finalize(), stats["guarded_channels"])


def test_support_precedes_nonzero_mean_and_affine_zero_offsets():
    coefficients = torch.zeros(1, 2, 4, 4)
    coefficients[0, 0, 0, 1] = -2
    coefficients[0, 1, 3, 2] = 3
    norm = FrozenNormalization([2, -3], [2, 0.5])
    support = patch_support(coefficients)
    patches = patchify(coefficients)
    gather_then_normalize = norm.normalize_patches(patches[0, support[0]])
    normalize_then_gather = patchify(norm(coefficients))[0, support[0]]
    torch.testing.assert_close(gather_then_normalize, normalize_then_gather, rtol=0, atol=0)
    assert support.tolist() == [[True, False, False, True]]
    assert gather_then_normalize[0, 0] == -1  # zeroed coeff must NOT be re-zeroed
    assert gather_then_normalize[0, 4] == 6
    assert patch_support(norm(coefficients)).all()  # would incorrectly resurrect empty patches
    assert not list(norm.parameters())


def test_all_empty_native_image_stays_absent_despite_nonzero_normalized_values():
    coeff = torch.zeros(3, 2, 4, 4)
    norm = FrozenNormalization([0.5, -0.5], [1, 1])
    support = patch_support(coeff)
    assert not support.any()
    assert torch.count_nonzero(norm(coeff)) == coeff.numel()
    normalized_empty = norm.normalize_patches(patchify(coeff)[support])
    assert normalized_empty.shape == (0, 8)


def test_guarded_near_zero_channel_and_finite_normalization():
    coeff = torch.ones(3, 2, 2, 2)
    coeff[:, 1] = 1e-8
    coeff[0, 1, 0, 0] = 2e-8
    moments = PopulationMoments(2)
    moments.update(coeff)
    stats = moments.finalize()
    assert stats["guarded_channels"] == [True, True]
    assert stats["effective_std"] == [1, 1]
    assert torch.isfinite(FrozenNormalization(stats["mean"], stats["effective_std"])(coeff)).all()


def test_ids_no_leakage_duplicates_missing_or_wrong_order():
    ids = ["train/1", "train/3", "train/8"]
    validate_sample_ids(ids, ids)
    for wrong in (["train/1", "val/2", "train/8"], ["train/1", "train/1", "train/8"], ids[:-1], ids[::-1]):
        with pytest.raises(ValueError):
            validate_sample_ids(wrong, ids)


def test_stats_artifact_hash_identity_and_reuse(tmp_path):
    accum = PopulationMoments(1)
    coeff = torch.tensor([[[[0., 1.], [-1., 3.]]]])
    accum.update(coeff)
    payload = {**accum.finalize(), "dataset": "mnist", "split_hash": "train-only-hash", "sparsity_percent_for_fit": 0,
               "frontend_config": {"version": 1}}
    path = tmp_path / "stats.json"
    with artifact_lock(path):
        saved = write_artifact(path, payload)
    norm, artifact = load_normalization(path, expected={"dataset": "mnist", "split_hash": "train-only-hash"})
    for sparsity in (0, 20, 40, 60, 80):
        assert torch.isfinite(norm(sparsify(coeff, sparsity))).all()
        assert norm.artifact_hash == saved["artifact_sha256"]
    assert list(tmp_path.glob("stats.*.npz")) and list(tmp_path.glob("stats.*.json"))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_normalization(path, expected={"split_hash": "validation"})
    with pytest.raises(ValueError, match="identity mismatch"):
        load_normalization(path, expected={"frontend_config": {"version": 2}})
    artifact["mean"][0] += 1
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="payload hash"):
        load_normalization(path)


def test_coverage_nonfinite_and_invalid_std_fail():
    accum = PopulationMoments(1)
    accum.update(torch.zeros(2, 1, 2, 2))
    with pytest.raises(ValueError, match="Image coverage"):
        accum.finalize(expected_images=3)
    with pytest.raises(ValueError, match="Pixel coverage"):
        accum.finalize(expected_pixels=100)
    with pytest.raises(ValueError, match="Nonfinite"):
        accum.update(torch.full((1, 1, 2, 2), float("nan")))
    with pytest.raises(ValueError):
        FrozenNormalization([0], [0])


def test_normalization_entrypoint_selects_only_active_dataset_modes(tmp_path):
    import json
    from compute_normalization_stats import selected_representations
    from sparse_contrast.scope import read_scope
    legacy = read_scope(tmp_path)
    assert len(selected_representations('mnist', 'all', legacy)) == 3
    scope = {'schema_version': 1, 'active_representations': {'mnist': ['grayscale'], 'cifar10': ['single_color', 'grayscale', 'color_opponency']}}
    (tmp_path / 'experiment_scope.json').write_text(json.dumps(scope))
    active = read_scope(tmp_path)
    assert selected_representations('mnist', 'all', active) == ('grayscale',)
    assert selected_representations('cifar10', 'all', active) == ('single_color', 'grayscale', 'color_opponency')
    assert selected_representations('cifar10', 'color_opponency', active) == ('color_opponency',)
    for excluded in ('single_color', 'color_opponency'):
        with pytest.raises(ValueError, match='outside the active mnist scope'):
            selected_representations('mnist', excluded, active)


def test_withdrawn_normalization_request_fails_before_dataset_access(monkeypatch):
    from types import SimpleNamespace
    from compute_normalization_stats import fit_dataset
    from sparse_contrast import data
    scope = {'schema_version': 1, 'active_representations': {'mnist': ['grayscale'], 'cifar10': ['single_color', 'grayscale', 'color_opponency']}}
    def forbidden_loader(*args, **kwargs):
        raise AssertionError('A withdrawn fit must fail before loading any data')
    monkeypatch.setattr(data, 'load_clean', forbidden_loader)
    with pytest.raises(ValueError, match='outside the active mnist scope'):
        fit_dataset('mnist', SimpleNamespace(representation='single_color'), scope)
