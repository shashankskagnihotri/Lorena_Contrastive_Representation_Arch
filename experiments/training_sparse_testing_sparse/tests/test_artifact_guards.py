"""Scientific cache/checkpoint provenance guards and honest partial-log recovery."""
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparse_contrast import common
from sparse_contrast import train as training
from sparse_contrast.frontend import Frontend
from sparse_contrast.normalization import PopulationMoments, write_artifact


def make_config_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "datasets_manifest.json").write_text('{"immutable":"dataset-split-identity"}\n')
    (tmp_path / "requirements.lock.txt").write_text(f"numpy=={version('numpy')}\n")
    manifest = {"sparse_contrast/common.py": common.file_hash(common.__file__)}
    source_hash = common.digest(manifest)
    snapshot = tmp_path / "outputs/source" / source_hash
    snapshot.mkdir(parents=True)
    (snapshot / "source_manifest.json").write_text(json.dumps(manifest))
    config = {"source_hash": source_hash,
              "data_manifest_sha256": common.file_hash(tmp_path / "datasets_manifest.json"),
              "environment_hash": common.file_hash(tmp_path / "requirements.lock.txt")}
    config["config_hash"] = common.digest(config)
    return config, snapshot


def test_resolved_config_digest_and_actual_inputs_guard(tmp_path, monkeypatch):
    config, _ = make_config_fixture(tmp_path, monkeypatch)
    common.validate_config_artifacts(config)
    changed_config = {**config, "seed": 7}
    with pytest.raises(ValueError, match="configuration digest"):
        common.validate_config_artifacts(changed_config)
    (tmp_path / "datasets_manifest.json").write_text('{"immutable":"different-split"}\n')
    with pytest.raises(ValueError, match="datasets_manifest.json identity"):
        common.validate_config_artifacts(config)


def test_environment_lock_and_installed_version_are_both_checked(tmp_path, monkeypatch):
    config, _ = make_config_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.changed")
    with pytest.raises(ValueError, match="Installed dependency changed"):
        common.validate_config_artifacts(config)
    (tmp_path / "requirements.lock.txt").write_text("numpy==0.changed\n")
    with pytest.raises(ValueError, match="requirements.lock.txt identity"):
        common.validate_config_artifacts(config)


def test_snapshot_manifest_and_executing_source_checked(tmp_path, monkeypatch):
    config, snapshot = make_config_fixture(tmp_path, monkeypatch)
    manifest_path = snapshot / "source_manifest.json"
    manifest_path.write_text(json.dumps({"sparse_contrast/common.py": "modified"}))
    with pytest.raises(ValueError, match="Code snapshot identity"):
        common.validate_config_artifacts(config)
    # Even a self-consistent manifest/config cannot describe different running code.
    manifest = json.loads(manifest_path.read_text())
    changed_source_hash = common.digest(manifest)
    altered_snapshot = tmp_path / "outputs/source" / changed_source_hash
    altered_snapshot.mkdir()
    (altered_snapshot / "source_manifest.json").write_text(json.dumps(manifest))
    config["source_hash"] = changed_source_hash
    config["config_hash"] = common.digest({k: v for k, v in config.items() if k != "config_hash"})
    with pytest.raises(ValueError, match="Executing source changed"):
        common.validate_config_artifacts(config)


def make_normalization_fixture(tmp_path, monkeypatch, changes=None):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "load_clean", lambda *args, **kwargs: SimpleNamespace(split_hash="permitted_train_ids"))
    moments = PopulationMoments(1)
    moments.update(torch.tensor([[[[-1., 0.], [0., 2.]]]]))
    metadata = {"dataset": "mnist", "representation": "grayscale", "protocol": "heldout_val",
                "split_hash": "permitted_train_ids", "frontend_config": Frontend("grayscale").config,
                "partition": "clean_train", "sparsity_percent_for_fit": 0,
                "augmentation": "none", "input_range": [0., 1.]}
    metadata.update(changes or {})
    path = tmp_path / "normalization/mnist_grayscale_heldout_val.json"
    saved = write_artifact(path, {**moments.finalize(), **metadata})
    return {"dataset": "mnist", "representation": "grayscale", "protocol": "heldout_val",
            "normalization_hash": saved["artifact_sha256"]}


def test_valid_stats_load_and_hash_is_frozen(tmp_path, monkeypatch):
    config = make_normalization_fixture(tmp_path, monkeypatch)
    assert training.load_stats(config)["split_hash"] == "permitted_train_ids"
    with pytest.raises(ValueError, match="Normalization identity changed"):
        training.load_stats({**config, "normalization_hash": "another_artifact"})


@pytest.mark.parametrize("key,incorrect", [
    ("dataset", "cifar10"), ("representation", "single_color"), ("protocol", "full_refit"),
    ("split_hash", "heldout_validation_ids"), ("frontend_config", {"blur_depth": 3}),
    ("partition", "clean_test"), ("sparsity_percent_for_fit", 80),
    ("augmentation", "random_crop"), ("input_range", [-1., 1.]),
])
def test_self_consistent_but_scientifically_wrong_stats_rejected(tmp_path, monkeypatch, key, incorrect):
    config = make_normalization_fixture(tmp_path, monkeypatch, {key: incorrect})
    with pytest.raises(ValueError, match=f"identity mismatch for {key}"):
        training.load_stats(config)


def test_only_incomplete_terminal_jsonl_record_recovered_with_exact_archive(tmp_path):
    path = tmp_path / "steps.jsonl"
    original = b'{"optimizer_step":1}\n{"optimizer_step":2}\n{"optimizer_step":'
    path.write_bytes(original)
    rows = common.read_recoverable_jsonl(path)
    assert rows == [{"optimizer_step": 1}, {"optimizer_step": 2}]
    assert path.read_bytes() == b'{"optimizer_step":1}\n{"optimizer_step":2}\n'
    archives = list(tmp_path.glob("steps.jsonl.incomplete-tail.*"))
    assert len(archives) == 1 and archives[0].read_bytes() == original
    records = list(tmp_path.glob("steps.jsonl.recovery.*.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["retained_complete_rows"] == 2
    assert common.read_recoverable_jsonl(path) == rows
    assert len(list(tmp_path.glob("steps.jsonl.incomplete-tail.*"))) == 1


@pytest.mark.parametrize("content", [
    b'{"optimizer_step":1}\nnot_json\n{"optimizer_step":3}\n',
    b'{"optimizer_step":1}\nnot_json\n',
])
def test_internal_or_newline_terminated_jsonl_corruption_fails_without_rewrite(tmp_path, content):
    path = tmp_path / "steps.jsonl"
    path.write_bytes(content)
    with pytest.raises(json.JSONDecodeError):
        common.read_recoverable_jsonl(path)
    assert path.read_bytes() == content
    assert not list(tmp_path.glob("steps.jsonl.incomplete-tail.*"))
