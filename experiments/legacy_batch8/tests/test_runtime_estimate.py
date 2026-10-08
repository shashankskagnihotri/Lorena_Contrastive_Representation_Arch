"""Hand-checkable arithmetic and synthetic timing-ledger records only."""
import json

import pytest

from sparse_contrast import runtime_estimate as runtime
from sparse_contrast.common import atomic_json
from sparse_contrast.scope import read_scope


def measured_groups():
    return {key:dict(state='measured',epoch_budget=50 if key.startswith('mnist/') else 200,
                     selected_epoch_plus_reporting_seconds=360.,selected_gpu_class='GPU-A',
                     selected_hardware_epoch_samples=3) for key in runtime.GROUP_KEYS}


def test_full_training_projection_units_and_idealized_parallelism(tmp_path):
    result=runtime.project_training_work(measured_groups(),scope=read_scope(tmp_path))
    # 58 runs/group * (50+50+200+200) epochs * 360 sec / 3600.
    assert result['all_232_training_reference_gpu_hours']==2900.
    assert result['idealized_elapsed_training_hours']=={'4':725.,'8':362.5}
    assert result['covered_main_runs']==228 and result['covered_pilot_runs']==4
    assert not result['heterogeneous_device_hour_total']
    assert 'isolated latency and throughput benchmarking' in result['exclusions']


def test_missing_group_cannot_be_reported_as_full_campaign_projection(tmp_path):
    groups=measured_groups(); del groups['cifar10/swin_tiny']
    result=runtime.project_training_work(groups,scope=read_scope(tmp_path))
    assert result['all_232_training_reference_gpu_hours'] is None
    assert result['idealized_elapsed_training_hours'] is None
    assert result['missing_groups']==['cifar10/swin_tiny']
    assert result['covered_main_runs']==171 and result['covered_pilot_runs']==3
    assert result['covered_groups_reference_gpu_hours']==1740.


def test_scoped_training_projection_uses_21_mnist_and_57_cifar_runs(tmp_path):
    scope=read_scope(tmp_path); scope['active_representations']['mnist']=['grayscale']
    result=runtime.project_training_work(measured_groups(),scope=scope)
    # (21+1)*2*50*0.1 + (57+1)*2*200*0.1 = 2540 device-hours.
    assert result['all_training_reference_gpu_hours']==2540.
    assert result['expected_main_runs']==result['covered_main_runs']==156
    assert result['expected_training_runs']==160
    assert result['idealized_elapsed_training_hours']=={'4':635.,'8':317.5}
    assert result['groups']['mnist/vit_small']['main_runs']==21
    assert result['groups']['cifar10/vit_small']['main_runs']==57
    assert 'all_232_training_reference_gpu_hours' not in result
    groups=measured_groups(); del groups['cifar10/swin_tiny']
    partial=runtime.project_training_work(groups,scope=scope)
    assert partial['covered_main_runs']==99
    assert partial['all_training_reference_gpu_hours'] is None
    assert partial['covered_groups_reference_gpu_hours']==1380.


def test_runtime_artifact_distinguishes_withdrawals_and_source_lineage(tmp_path):
    scope=read_scope(tmp_path); scope['active_representations']['mnist']=['grayscale']
    state=dict(phase='main',pilots=[],source_hash='original_training',
               orchestration_source_hash='scoped_controller',withdrawn_main=[{'registry_id':str(i)} for i in range(72)])
    path=runtime.write_runtime_estimate(state,output_path=tmp_path/'runtime.json',scope=scope)
    result=json.loads(path.read_text())
    assert result['expected_main_runs']==156 and result['withdrawn_main_runs']==72
    assert result['source_hash']=='original_training' and result['orchestration_source_hash']=='scoped_controller'


def timing_ledger(tmp_path,monkeypatch,epoch_seconds):
    monkeypatch.setattr(runtime,'run_dir',lambda config: tmp_path)
    monkeypatch.setattr(runtime,'tb_dir',lambda config,phase: tmp_path/'tensorboard'/phase)
    config=dict(dataset='mnist',architecture='vit_small',role='pilot',representation='raw',execution='dense',
                protocol='heldout_val',config_hash='verified-config',recipe={'epochs':50})
    atomic_json(tmp_path/'progress.json',{'epoch':3,'step':20625,'seconds':2000.,'updated':1})
    atomic_json(tmp_path/'attempt-1.json',{'start_epoch':0,'config_hash':'verified-config','hardware':{'gpu':'GPU-A'}})
    rows=[dict(epoch=index,epoch_seconds=seconds,train_online={'total':55000},validation={'total':5000})
          for index,seconds in enumerate(epoch_seconds)]
    (tmp_path/'epochs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    logging=[{'epoch':index,'seconds':seconds} for index,seconds in enumerate([200.,0.,0.])]
    (tmp_path/'logging.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in logging))
    return {'config':config}


def test_median_uses_paired_epoch_and_logging_costs(tmp_path,monkeypatch):
    item=timing_ledger(tmp_path,monkeypatch,[100.,200.,300.])
    result=runtime.pilot_epoch_evidence(item)
    # median([100+200, 200+0, 300+0]) = 300; summing separate medians gives 200.
    assert result['selected_epoch_plus_reporting_seconds']==300.
    assert result['initial_setup']['duration_seconds'] is None
    assert not result['checkpoint_cost']['projected']


def test_migrated_pilot_does_not_pool_gpu_classes(tmp_path,monkeypatch):
    item=timing_ledger(tmp_path,monkeypatch,[100.,200.,900.])
    atomic_json(tmp_path/'attempt-2.json',{'start_epoch':2,'config_hash':'verified-config','hardware':{'gpu':'GPU-B'}})
    result=runtime.pilot_epoch_evidence(item)
    assert result['hardware_classes']['GPU-A']['median_epoch_plus_reporting_seconds']==250.
    assert result['hardware_classes']['GPU-B']['median_epoch_plus_reporting_seconds']==900.
    assert result['selected_gpu_class']=='GPU-B' and result['selected_epoch_plus_reporting_seconds']==900.
    assert result['selected_hardware_epoch_samples']==1


def test_smoke_timing_cannot_become_a_production_estimate(tmp_path,monkeypatch):
    item=timing_ledger(tmp_path,monkeypatch,[100.,200.,300.])
    item['config']['role']='smoke'
    with pytest.raises(ValueError,match='never a smoke'):
        runtime.pilot_epoch_evidence(item)
