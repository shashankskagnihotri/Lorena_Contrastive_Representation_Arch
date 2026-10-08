"""Reject semantic defects and provenance drift; preserve exact FP32 model state."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import worker
import validation as v


class TinyBackbone(torch.nn.Module):
    def __init__(self, semantic_error=0.):
        super().__init__()
        self.linear=torch.nn.Linear(4,2)
        with torch.no_grad():
            self.linear.weight.fill_(1e-4);self.linear.bias.zero_()
        self.semantic_error=semantic_error

    def forward_patches(self,patches,support,execution):
        return self.linear(patches[:,0])+(self.semantic_error if execution=='dense_masked' else 0.)


class TinyModel(torch.nn.Module):
    def __init__(self, semantic_error=0.):
        super().__init__();self.backbone=TinyBackbone(semantic_error)

    def prepare(self,raw,execution):
        return raw.float().reshape(len(raw),1,4),torch.ones(len(raw),1,dtype=torch.bool)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    source={'scientific_source_hash':v.SCIENTIFIC_SOURCE,'validation_source_hash':'a'*64,'policy':v.POLICY}
    monkeypatch.setattr(v,'source_receipt',lambda:source)
    identity={'source_hash':v.SCIENTIFIC_SOURCE,'test_config_hash':'b'*64}
    return tmp_path,identity,{'measurement_session_id':'fixture'},['image0','image1']


@torch.inference_mode()
def run_validation(setup,error=0.,semantic_error=0.):
    directory,identity,measurement,ids=setup
    model=TinyModel(semantic_error).eval();raw=torch.ones(2,1,2,2,dtype=torch.uint8)
    patches,support=model.prepare(raw,'compact')
    compact=model.backbone.forward_patches(patches,support,'compact')
    dense=model.backbone.forward_patches(patches,support,'dense_masked')+error
    before={k:x.clone() for k,x in model.state_dict().items()}
    v.validate_parity(model,raw,compact,dense,directory,identity,measurement,ids)
    assert all(torch.equal(before[k],x) for k,x in model.state_dict().items())
    assert all(x.dtype==torch.float32 for x in model.parameters())
    ref=v.validation_reference(directory)
    return v.verify_validation_reference(ref,identity,measurement),ref


def test_original_strict_path_retained(setup):
    saved,_=run_validation(setup)
    assert saved['state']=='original_strict_pass' and not saved['fallback_used']
    assert not list(setup[0].glob('*.npz'))


def test_reference_rounding_path_preserves_model_and_raw_evidence(setup):
    saved,_=run_validation(setup,error=3.5e-5)
    assert saved['state']=='reference_verified_rounding' and saved['fallback_used']
    assert 'original_strict_failure' in saved
    assert saved['reference_comparisons']['fp64_compact_vs_dense']['max_absolute_error']==0.


@pytest.mark.parametrize('error,semantic_error',[(.01,0.),(0.,.01)])
def test_reference_rejects_excess_rounding_and_real_semantic_error(setup,error,semantic_error):
    with pytest.raises(AssertionError):run_validation(setup,error,semantic_error)
    path=next(setup[0].glob('parity_validation.*.json'))
    assert json.loads(path.read_text())['state']=='failed'


def test_reference_archive_tampering_is_rejected(setup):
    saved,ref=run_validation(setup,error=3.5e-5)
    path=Path(saved['reference_archive']);path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='evidence changed'):
        v.verify_validation_reference(ref,setup[1],setup[2])


def test_wrong_scientific_identity_is_rejected(setup):
    setup[1]['source_hash']='wrong'
    with pytest.raises(ValueError,match='different scientific source'):run_validation(setup)


def test_nonfinite_fp32_is_rejected_with_failed_receipt(setup):
    with pytest.raises(ValueError,match='Nonfinite FP32'):run_validation(setup,error=float('inf'))
    saved=json.loads(next(setup[0].glob('parity_validation.*.json')).read_text())
    assert saved['state']=='failed' and saved['original_comparison']['all_finite'] is False


def test_copied_orchestration_matches_frozen_timed_computation():
    result=v.verify_copied_orchestration(Path(worker.benchmark.__file__),
        worker.SCIENTIFIC_ROOT/'dense_sparse/worker.py',HERE/'worker.py')
    assert 'AST_equal' in result['measure_cell']


def test_every_copied_global_binding_is_resolved_and_missing_import_fails(monkeypatch):
    from dense_sparse import worker as original
    v.verify_global_bindings(worker.benchmark.measure_cell,worker.measure_cell,{'validate_parity','validation_reference'})
    v.verify_global_bindings(original.benchmark_case,worker.benchmark_case,{'measure_cell_with_validation_receipt'})
    monkeypatch.delattr(worker,'timing_batches')
    with pytest.raises(ValueError,match='Missing copied-function global dependency'):
        v.verify_global_bindings(worker.benchmark.measure_cell,worker.measure_cell,{'validate_parity','validation_reference'})


def test_timing_body_change_cannot_be_hidden(tmp_path):
    changed=(HERE/'worker.py').read_text().replace('_serial_measure(call, repeats, device)',
                                               '_serial_measure(call, repeats+1, device)')
    p=tmp_path/'worker.py';p.write_text(changed)
    with pytest.raises(ValueError,match='differs beyond'):
        v.verify_copied_orchestration(Path(worker.benchmark.__file__),
            worker.SCIENTIFIC_ROOT/'dense_sparse/worker.py',p)


def test_source_manifest_and_exact_files_are_verified(tmp_path,monkeypatch):
    for name in ['worker.py','validation.py']:(tmp_path/name).write_text(name)
    manifest=dict(scientific_source_hash=v.SCIENTIFIC_SOURCE,base_source_hash=v.BASE_SOURCE,
                  policy=v.POLICY,files={name:v.file_hash(tmp_path/name) for name in ['worker.py','validation.py']})
    (tmp_path/'source_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(v,'__file__',str(tmp_path/'validation.py'))
    assert v.source_receipt()['validation_source_hash']==v.digest(manifest)
    (tmp_path/'worker.py').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):v.source_receipt()
