"""Adapter semantics and strict full-backbone parity; run under CPU/GPU allocation."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from dense_sparse.adapter import InferenceClassifier
from sparse_contrast.pipeline import Classifier
from sparse_contrast.frontend import Frontend, sparsify, patch_support, patchify


def config_and_stats(architecture='vit_small', representation='grayscale', dataset='mnist', execution='dense'):
    config = dict(architecture=architecture, representation=representation, dataset=dataset,
                  execution=execution, sparsity_percent=None if representation=='raw' else 0,
                  recipe={'stochastic_depth':0.1}, config_hash='original-training-identity')
    if representation == 'raw': return config,None
    frontend = Frontend(representation)
    config['frontend'] = frontend.config
    channels = frontend.config['native_channels']
    stats = dict(frontend_config=frontend.config, mean=[0.125]*channels, effective_std=[0.5]*channels)
    return config,stats


def policy(percent=60, execution='compact', domain='native_contrast'):
    return dict(inference_execution=execution,inference_sparsity_percent=percent,sparsification_domain=domain)


@pytest.fixture
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('architecture',['vit_small','swin_tiny'])
@pytest.mark.parametrize('execution',['dense','compact'])
def test_strict_original_state_layout_and_original_zero_policy_output(architecture,execution,cpu_threads):
    torch.manual_seed(123)
    config,stats=config_and_stats(architecture,execution=execution)
    unchanged=deepcopy(config)
    base=Classifier(config,stats).eval()
    adapter=InferenceClassifier(config,stats,policy(0,execution)).eval()
    assert list(adapter.state_dict())==list(base.state_dict())
    result=adapter.load_state_dict(base.state_dict(),strict=True)
    assert result.missing_keys==result.unexpected_keys==[]
    assert adapter.config==unchanged and config==unchanged
    raw=torch.randint(0,256,(1,1,28,28),dtype=torch.uint8)
    with torch.inference_mode():
        torch.testing.assert_close(adapter(raw),base(raw),rtol=0,atol=0)
    missing=dict(base.state_dict());missing.pop('backbone.head.bias')
    with pytest.raises(RuntimeError,match='Missing key'):
        adapter.load_state_dict(missing,strict=True)


@pytest.mark.parametrize('architecture',['vit_small','swin_tiny'])
def test_dense_trained_checkpoint_defaults_to_compact_and_matches_masked_reference(architecture,cpu_threads):
    torch.manual_seed(321)
    config,stats=config_and_stats(architecture)
    base=Classifier(config,stats).eval()
    adapter=InferenceClassifier(config,stats,policy()).eval()
    adapter.load_state_dict(base.state_dict(),strict=True)
    assert adapter.backbone.execution=='dense' and adapter.config['execution']=='dense'
    raw=torch.randint(0,256,(2,1,28,28),dtype=torch.uint8);raw[0].zero_()
    with torch.inference_mode():
        _,support=adapter.prepare(raw)
        assert not support[0].any() and support[1].any()
        compact=adapter(raw)
        assert adapter.backbone.latest_telemetry['execution']=='compact'
        reference=adapter(raw,execution='dense_masked')
    assert torch.isfinite(compact).all()
    torch.testing.assert_close(compact,reference,rtol=1e-4,atol=2e-5)
    # Natural-zero removal occurs even with no imposed coefficient sparsity.
    natural=InferenceClassifier(config,stats,policy(0)).eval()
    natural.load_state_dict(base.state_dict(),strict=True)
    assert not natural.prepare(raw[:1])[1].any()
    assert natural.prepare(raw[:1],execution='dense')[1] is None


class RecordingBackbone(nn.Module):
    def __init__(self,execution):
        super().__init__();self.execution=execution;self.record=None
    def set_profiling(self,enabled): self.profiling=enabled
    def forward_patches(self,patches,support,execution=None):
        self.record=(patches,support,execution)
        return patches.sum((1,2))[:,None]


@pytest.fixture
def preprocessing_only(monkeypatch):
    # Actual frontend/sparsification/normalization, without redundant backbone work.
    monkeypatch.setattr('sparse_contrast.pipeline.build_backbone',
                        lambda *args,execution,**kwargs:RecordingBackbone(execution))


@pytest.mark.parametrize('representation',['grayscale','single_color','color_opponency'])
def test_contrast_preparation_matches_existing_native_math_and_frozen_statistics(representation,preprocessing_only):
    config,stats=config_and_stats(representation=representation,dataset='cifar10')
    original_config=deepcopy(config);original_stats=deepcopy(stats)
    spec=policy();model=InferenceClassifier(config,stats,spec)
    raw=torch.arange(3*32*32,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(1,3,32,32)
    patches,support=model.prepare(raw)
    coefficients=Frontend(representation)(raw.float()/255)
    sparse=sparsify(coefficients,60)
    torch.testing.assert_close(patches,patchify((sparse-model.mean)/model.std,2),rtol=0,atol=0)
    assert torch.equal(support,patch_support(sparse,2,0))
    assert config==original_config and stats==original_stats and model.config==original_config
    spec['inference_sparsity_percent']=90
    assert model.inference_sparsity_percent==60
    with pytest.raises(TypeError): model.inference_spec['inference_sparsity_percent']=90


def test_raw_pixel_sparsification_is_explicit_before_normalization_with_diagnostics(preprocessing_only):
    config,stats=config_and_stats(representation='raw')
    raw=torch.arange(16,dtype=torch.uint8).reshape(1,1,4,4)
    model=InferenceClassifier(config,stats,policy(50,domain='raw_pixels'))
    model.collect_diagnostics=True
    model(raw)
    patches,support,execution=model.backbone.record
    expected=sparsify(raw.float()/255,50)
    assert execution=='compact' and support.tolist()==[[False,False,True,True]]
    torch.testing.assert_close(patches,patchify((expected-model.mean)/model.std,2),rtol=0,atol=0)
    assert (patches[~support]!=0).all()  # Centered values must never define support.
    assert model.latest_diagnostics['natural_zero_fraction'].item()==1/16
    assert model.latest_diagnostics['achieved_zero_fraction'].item()==0.5
    assert model.latest_diagnostics['retained_patches'].item()==2
    model(raw,execution='dense')
    assert model.backbone.record[1] is None and model.backbone.record[2]=='dense'
    reference=InferenceClassifier(config,stats,policy(0,'dense','none'))
    untouched,_=reference.prepare(raw)
    torch.testing.assert_close(untouched,patchify((raw.float()/255-reference.mean)/reference.std,2),rtol=0,atol=0)
    assert config['sparsity_percent'] is None


@pytest.mark.parametrize('spec',[
    policy(10,domain='none'),policy(20,domain='native_contrast'),policy(float('nan'),domain='raw_pixels'),
    policy(-1,domain='raw_pixels'),policy(101,domain='raw_pixels'),policy(True,domain='raw_pixels'),
    policy(10,'unknown','raw_pixels'),policy(10,domain='unknown'),{},
])
def test_raw_policy_validation_never_silently_ignores_requested_sparsity(spec,preprocessing_only):
    config,stats=config_and_stats(representation='raw')
    with pytest.raises(ValueError): InferenceClassifier(config,stats,spec)


def test_contrast_rejects_raw_domain_and_nonzero_training_checkpoint(preprocessing_only):
    config,stats=config_and_stats()
    with pytest.raises(ValueError,match='trained representation'):
        InferenceClassifier(config,stats,policy(domain='raw_pixels'))
    config['sparsity_percent']=20
    with pytest.raises(ValueError,match='zero-imposed'):
        InferenceClassifier(config,stats,policy())


def test_profiling_scopes_remain_separate_and_default_execution_is_explicit(preprocessing_only):
    config,stats=config_and_stats(representation='raw')
    model=InferenceClassifier(config,stats,policy(60,domain='raw_pixels'))
    raw=torch.arange(16,dtype=torch.uint8).reshape(1,1,4,4)
    scopes=[]
    from contextlib import contextmanager
    @contextmanager
    def recording_scope(name):
        scopes.append(name);yield
    model.scope=recording_scope
    model(raw)
    assert scopes==['input_conversion','native_sparsification','native_support','frozen_normalization_and_patch_layout']
    assert model.backbone.record[2]=='compact'
    with pytest.raises(ValueError,match='override'): model(raw,execution='')
