import json
from pathlib import Path

import pytest

from sparse_contrast import common
from sparse_contrast.scope import (configurations, expected_group_count,
    expected_main_count, expected_worker_count, read_scope, validate_registry)


def test_revised_scope_is_exact_and_preserves_cifar(tmp_path):
    legacy = read_scope(tmp_path)
    scope = {**legacy, 'active_representations': {**legacy['active_representations'], 'mnist': ['grayscale']}}
    assert expected_group_count('mnist', scope) == 21
    assert expected_group_count('cifar10', scope) == 57
    assert expected_main_count(scope) == 156
    assert expected_worker_count(scope) == 552
    assert configurations('cifar10', scope) == configurations('cifar10', legacy)
    runs = [dict(dataset=d, architecture=a, representation=r, execution=e, sparsity_percent=p, seed=s)
            for d in ('mnist', 'cifar10') for a in ('vit_small', 'swin_tiny')
            for r,e,p in configurations(d, scope) for s in (0,1,2)]
    assert len(validate_registry(runs, scope)) == 156
    bad = [dict(row) for row in runs]
    bad[3]['representation'] = 'single_color'
    with pytest.raises(ValueError, match='authorized'):
        validate_registry(bad, scope)


@pytest.fixture
def compatible_sources(tmp_path, monkeypatch):
    root = tmp_path/'project'; execution = tmp_path/'execution'
    monkeypatch.setattr(common, 'ROOT', root)
    monkeypatch.setattr(common, '__file__', str(execution/'sparse_contrast/common.py'))
    payload = {name: ('unchanged '+name).encode() for name in common.SCIENTIFIC_SOURCE_FILES}
    payload['sparse_contrast/common.py'] = b'original validation'
    training = {name: common.hashlib.sha256(data).hexdigest() for name,data in payload.items()}
    training_hash = common.digest(training)
    training_path = root/'outputs/source'/training_hash
    training_path.mkdir(parents=True)
    (training_path/'source_manifest.json').write_text(json.dumps(training))
    for name,data in payload.items():
        p=execution/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
    (execution/'sparse_contrast/common.py').write_bytes(b'explicit source compatibility validation')
    manifest={name:common.file_hash(execution/name) for name in payload}
    actual_hash=common.digest(manifest)
    (execution/'source_manifest.json').write_text(json.dumps(manifest))
    approval=dict(schema_version=1,training_source_hash=training_hash,
                  execution_source_hash=actual_hash,
                  unchanged_scientific_files=list(common.SCIENTIFIC_SOURCE_FILES))
    path=root/'outputs/source_compatibility'/f'{actual_hash}.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(approval))
    return {'source_hash':training_hash},actual_hash,execution,path,manifest


def test_registered_orchestration_source_preserves_training_identity(compatible_sources):
    config,actual_hash,_,_,_=compatible_sources
    assert common.executing_source_hash(config)==actual_hash
    assert config['source_hash']!=actual_hash


def test_unregistered_orchestration_source_fails(compatible_sources):
    config,_,_,approval,_=compatible_sources
    approval.unlink()
    with pytest.raises(ValueError,match='compatibility approval'):
        common.executing_source_hash(config)


def test_changed_scientific_code_cannot_use_compatibility(compatible_sources):
    config,_,execution,approval,manifest=compatible_sources
    (execution/'sparse_contrast/models.py').write_bytes(b'changed architecture')
    manifest['sparse_contrast/models.py']=common.file_hash(execution/'sparse_contrast/models.py')
    actual_hash=common.digest(manifest)
    (execution/'source_manifest.json').write_text(json.dumps(manifest))
    record=json.loads(approval.read_text()); record['execution_source_hash']=actual_hash
    approval.with_name(actual_hash+'.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Scientific source changed'):
        common.executing_source_hash(config)


def test_mutated_execution_bytes_fail(compatible_sources):
    config,_,execution,_,_=compatible_sources
    (execution/'sparse_contrast/common.py').write_bytes(b'changed after approval')
    with pytest.raises(ValueError,match='Executing source changed'):
        common.executing_source_hash(config)
