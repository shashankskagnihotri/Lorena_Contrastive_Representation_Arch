"""Provisional training device-hours from actual full raw-pilot epochs only.

No smoke timing, FLOP model, cross-GPU scaling factor, or invented setup residual
enters these estimates. They are explicitly not total campaign completion times.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
import time

from .common import ROOT, atomic_json, run_dir, tb_dir
from .scope import read_scope, expected_main_count, expected_group_count


GROUP_KEYS=tuple(dataset+'/'+architecture for dataset in ('mnist','cifar10')
                 for architecture in ('vit_small','swin_tiny'))
TRAIN_COUNTS={'mnist':55000,'cifar10':45000}
EXCLUSIONS=(
    'final clean-test and corruption evaluation',
    'isolated latency and throughput benchmarking',
    'queue time, resource availability, scheduling dependencies and retries',
    'initialization, data setup and initial diagnostic panels',
    'checkpoint serialization and post-checkpoint writer flush',
    'unknown contrast frontend, sparsification and compact execution cost differences',
    'reporting after training and optional full-data refits',
)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _complete_jsonl(path):
    """Read committed newline records without editing a live worker's log."""
    try: content=Path(path).read_bytes()
    except FileNotFoundError: return [],False
    partial=bool(content and not content.endswith(b'\n'))
    if partial: content=content.rsplit(b'\n',1)[0]+b'\n' if b'\n' in content else b''
    rows=[]
    for index,line in enumerate(content.splitlines()):
        if not line.strip(): raise ValueError(f'Blank interior record in {path}:{index+1}')
        rows.append(json.loads(line))
    return rows,partial


def _finite_seconds(value,name,positive=False):
    number=float(value)
    if not math.isfinite(number) or number<0 or (positive and number==0):
        raise ValueError(f'Invalid {name}: {value!r}')
    return number


def pilot_epoch_evidence(item):
    config=item['config']; dataset=config['dataset']; directory=run_dir(config)
    key=dataset+'/'+config['architecture']
    if (config.get('role')!='pilot' or config.get('representation')!='raw'
            or config.get('execution')!='dense' or config.get('protocol')!='heldout_val'
            or config.get('subset_ids') is not None):
        raise ValueError('Runtime evidence must be a full held-out-protocol raw pilot, never a smoke/overfit run')
    result=dict(group=key,config_hash=config['config_hash'],epoch_budget=int(config['recipe']['epochs']),
                state='pending_full_epoch',epoch_samples=0,artifact_directory=str(directory),
                initial_setup=dict(projected=False,duration_seconds=None,
                    separately_logged_component='setup/initial_panel_logging_seconds',
                    tensorboard_directory=str(tb_dir(config,'train')),
                    note='Initial panel setup is recorded in TensorBoard only; no residual duration is inferred.'),
                checkpoint_cost=dict(projected=False,duration_seconds=None,note='No separate numeric checkpoint duration is available.'))
    progress_path=directory/'progress.json'; completed_path=directory/'completed.json'
    if progress_path.exists():
        progress=_read_json(progress_path)
        committed=int(progress['epoch'])
        result['progress']=dict(completed_epoch=committed,optimizer_step=progress['step'],
                                elapsed_current_attempt_seconds=progress['seconds'],updated=progress.get('updated'))
    elif completed_path.exists():
        completed=_read_json(completed_path)
        if completed['config_hash']!=config['config_hash']: raise ValueError('Pilot completion identity mismatch')
        committed=int(completed['epochs'])
    else: return result
    if committed<0 or committed>result['epoch_budget']: raise ValueError('Pilot progress exceeds its frozen budget')
    epochs,partial_epochs=_complete_jsonl(directory/'epochs.jsonl')
    logging,partial_logging=_complete_jsonl(directory/'logging.jsonl')
    result['live_trailing_partial_record']=partial_epochs or partial_logging
    rows={}
    for row in epochs:
        epoch=int(row['epoch'])
        if epoch>=committed: continue
        if epoch in rows: raise ValueError('Duplicate committed pilot epoch')
        rows[epoch]=row
    if set(rows)!=set(range(committed)): raise ValueError('Committed full pilot epoch coverage is incomplete')
    # Reporting records may include replayed epochs after a crash. The latest
    # completed newline record for a committed epoch accompanies the new epoch.
    reporting={int(row['epoch']):row for row in logging if int(row['epoch'])<committed}
    attempts=[]
    for path in sorted(directory.glob('attempt-*.json')):
        attempt=_read_json(path)
        if attempt['config_hash']!=config['config_hash']: raise ValueError('Pilot attempt identity mismatch')
        attempts.append(dict(attempt,start_epoch=int(attempt['start_epoch']),artifact=str(path)))
    if not attempts:
        result.update(state='pending_hardware_provenance',committed_epochs=committed); return result
    samples=[]
    for epoch,row in sorted(rows.items()):
        if row['train_online']['total']!=TRAIN_COUNTS[dataset] or row.get('validation',{}).get('total')!=5000:
            raise ValueError('Runtime row does not cover the complete prescribed training/validation partitions')
        if epoch not in reporting:
            result.update(state='pending_reporting_duration',committed_epochs=committed); return result
        eligible=[attempt for attempt in attempts if attempt['start_epoch']<=epoch]
        if not eligible: raise ValueError('Epoch has no matching execution attempt')
        attempt=eligible[-1]; hardware=attempt['hardware']; gpu=hardware.get('gpu')
        if not gpu: raise ValueError('Pilot runtime was not measured in a recorded GPU allocation')
        elapsed=_finite_seconds(row['epoch_seconds'],'epoch_seconds',positive=True)
        log_seconds=_finite_seconds(reporting[epoch]['seconds'],'epoch_reporting_seconds')
        samples.append(dict(epoch=epoch,gpu=gpu,train_validation_seconds=elapsed,reporting_seconds=log_seconds,
                            epoch_plus_reporting_seconds=elapsed+log_seconds,attempt_artifact=attempt['artifact']))
    if not samples: return result
    # A migrated pilot contributes separate hardware-class distributions. The
    # projection uses only the latest observed class for this comparison group.
    latest_gpu=samples[-1]['gpu']; classes={}
    for gpu in sorted({sample['gpu'] for sample in samples}):
        selected=[sample for sample in samples if sample['gpu']==gpu]
        classes[gpu]=dict(epoch_samples=len(selected),epoch_ids=[sample['epoch'] for sample in selected],
            median_train_validation_seconds=median(sample['train_validation_seconds'] for sample in selected),
            median_reporting_seconds=median(sample['reporting_seconds'] for sample in selected),
            median_epoch_plus_reporting_seconds=median(sample['epoch_plus_reporting_seconds'] for sample in selected))
    result.update(state='measured',epoch_samples=len(samples),selected_gpu_class=latest_gpu,
                  hardware_classes=classes,
                  selected_epoch_plus_reporting_seconds=classes[latest_gpu]['median_epoch_plus_reporting_seconds'],
                  selected_hardware_epoch_samples=classes[latest_gpu]['epoch_samples'],
                  observed_epoch_plus_reporting_seconds=sum(sample['epoch_plus_reporting_seconds'] for sample in samples),
                  measured_epoch_records=samples)
    return result


def project_training_work(evidence,scope=None):
    """Scale each group's raw-pilot reference rate without crossing GPU classes."""
    scope=read_scope(ROOT) if scope is None else scope
    expected_main=expected_main_count(scope)
    estimates={}; missing=[]
    for key in GROUP_KEYS:
        measured=evidence.get(key,{})
        if measured.get('state')!='measured': missing.append(key); continue
        seconds=_finite_seconds(measured['selected_epoch_plus_reporting_seconds'],'reference_epoch_seconds',positive=True)
        epochs=int(measured['epoch_budget'])
        if epochs<=0: raise ValueError('Epoch budget must be positive')
        rate=epochs*seconds/3600.
        main_count=expected_group_count(key.split('/')[0],scope)
        estimates[key]=dict(gpu_class=measured['selected_gpu_class'],epoch_budget=epochs,
            sampled_epochs=measured['selected_hardware_epoch_samples'],
            median_epoch_plus_reporting_seconds=seconds,
            main_runs=main_count,pilot_runs=1,per_run_reference_gpu_hours=rate,
            main_reference_gpu_hours=main_count*rate,pilot_reference_gpu_hours=rate,
            combined_reference_gpu_hours=(main_count+1)*rate)
    covered=sum(value['combined_reference_gpu_hours'] for value in estimates.values())
    complete=not missing
    hardware={key:value['gpu_class'] for key,value in estimates.items()}
    result=dict(state='provisional_complete_group_coverage' if complete else 'incomplete_group_coverage',
        interpretation=f'Raw-pilot reference projection for {expected_main} active main training runs plus 4 pilots; not a total campaign ETA.',
        expected_main_runs=expected_main,expected_pilot_runs=4,expected_training_runs=expected_main+4,
        covered_main_runs=sum(value['main_runs'] for value in estimates.values()),covered_pilot_runs=len(estimates),
        missing_groups=missing,groups=estimates,assumed_gpu_class_by_group=hardware,
        heterogeneous_device_hour_total=len(set(hardware.values()))>1,
        covered_groups_reference_gpu_hours=covered,
        all_training_reference_gpu_hours=covered if complete else None,
        idealized_elapsed_training_hours={str(count):covered/count for count in (4,8)} if complete else None,
        elapsed_scenario_assumptions='Perfectly balanced 4 or 8 concurrent GPUs matching each group hardware class; excludes queues, pilot dependencies and resource availability.',
        transfer_assumption='Every contrast/control run is provisionally assigned its same-group raw-pilot epoch cost. Actual frontend and compaction effects remain unmeasured.',
        exclusions=list(EXCLUSIONS))
    if expected_main==228:
        # Compatibility for historical artifacts without mislabeling a reduced scope.
        result['all_232_training_reference_gpu_hours']=result['all_training_reference_gpu_hours']
    return result


def write_runtime_estimate(state,output_path=None,scope=None):
    evidence={}
    for item in state.get('pilots',[]):
        config=item['config']; key=config['dataset']+'/'+config['architecture']
        try: evidence[key]=pilot_epoch_evidence(item)
        except (ValueError,KeyError,OSError) as error:
            # This optional diagnostic must faithfully report malformed/missing
            # evidence instead of cancelling independent valid training work.
            evidence[key]=dict(state='invalid_evidence',error=str(error),config_hash=config.get('config_hash'))
    result=project_training_work(evidence,scope=scope)
    result.update(schema_version=1,updated=time.time(),campaign_phase=state['phase'],
                  source_hash=state['source_hash'],pilot_evidence=evidence,
                  orchestration_source_hash=state.get('orchestration_source_hash',state['source_hash']),
                  withdrawn_main_runs=len(state.get('withdrawn_main',[])))
    path=Path(output_path) if output_path is not None else ROOT/'outputs/runtime_estimate.json'
    atomic_json(path,result)
    return path
