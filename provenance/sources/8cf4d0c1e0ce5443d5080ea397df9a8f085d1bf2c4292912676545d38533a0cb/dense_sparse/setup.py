"""Freeze inference code and derive the complete authorized checkpoint matrix."""
import argparse
from pathlib import Path
import shutil
import hashlib
import tempfile
import os
from .common import *
from . import BASE_SOURCE_HASH


def snapshot():
    source=Path(__file__).resolve().parents[1]
    paths=sorted(list((source/'dense_sparse').glob('*.py'))+list((source/'tests').glob('*.py'))+[source/'PLAN.md',source/'MONITOR.md',source/'README.md',source/'audit_zero_checkpoints.py'])
    payloads={str(p.relative_to(source)):p.read_bytes() for p in paths}
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}
    identity=digest(manifest);dest=ROOT/'outputs/source'/identity
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        temporary=Path(tempfile.mkdtemp(prefix='.snapshot-',dir=dest.parent))
        try:
            for name,data in payloads.items():
                target=temporary/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            atomic_json(temporary/'source_manifest.json',manifest)
            os.rename(temporary,dest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    for name,want in manifest.items():
        if file_hash(dest/name)!=want:
            raise ValueError('Immutable inference snapshot collision')
    return identity,dest


def build(source_hash, source_path):
    existing=ROOT/'study_manifest.json'
    if existing.exists():
        manifest=load_manifest()
        if manifest['source_hash']!=source_hash:
            raise ValueError('An existing study requires an explicit recorded repair, not replacement')
        return manifest
    campaign=read_json(BASE_ROOT/'outputs/campaign.json')
    selected=[x for x in campaign['main'] if x['config']['sparsity_percent']==0 or x['config']['representation']=='raw']
    if len(selected)!=60 or len({x['config']['config_hash'] for x in selected})!=60:
        raise ValueError('Expected all60 distinct eligible final checkpoints')
    cases=[];sources=[]
    for item in sorted(selected,key=lambda x:x['config']['registry_id']):
        config=item['config'];rd=run_dir(config);done_path=rd/'completed.json';done=read_json(done_path)
        if item['state']!='completed' or item['stage']!='evaluation' or done['state']!='completed' or done['epochs']!=config['recipe']['epochs'] or done['config_hash']!=config['config_hash']:
            raise ValueError('Incomplete/inconsistent source training run')
        if read_json(item['config_path'])!=config:
            raise ValueError('Source configuration changed')
        common=dict(study_id=STUDY_ID,schema_version=1,training_config_path=item['config_path'],
            training_config_hash=config['config_hash'],checkpoint_path=str(rd/'final.pt'),checkpoint_sha256=done['checkpoint_sha256'],
            completed_receipt_sha256=file_hash(done_path),trained_execution=config['execution'],
            trained_sparsity_percent=config['sparsity_percent'],training_source_hash=config['source_hash'],
            normalization_hash=config['normalization_hash'],data_manifest_sha256=config['data_manifest_sha256'],
            dataset=config['dataset'],architecture=config['architecture'],representation=config['representation'],seed=config['seed'],
            source_hash=source_hash,base_source_hash=BASE_SOURCE_HASH,original_evaluation_path=str(rd/'evaluation/summary.json'),
            original_evaluation_sha256=file_hash(rd/'evaluation/summary.json'),original_epochs_path=str(rd/'epochs.jsonl'),
            support_policy='native_exact_zero_before_normalization',precision='fp32')
        sources.append(common)
        for percent in range(0,100,10):
            case=dict(common,inference_execution='compact',inference_sparsity_percent=percent,
                      sparsification_domain='raw_pixels' if config['representation']=='raw' else 'native_contrast',case_role='sparsity_sweep')
            case['test_config_hash']=digest(case);cases.append(case)
        if config['representation']=='raw':
            case=dict(common,inference_execution='dense',inference_sparsity_percent=0,sparsification_domain='none',case_role='unchanged_raw_reference')
            case['test_config_hash']=digest(case);cases.append(case)
    if len(cases)!=612 or len({x['test_config_hash'] for x in cases})!=612:
        raise ValueError('Incomplete inference matrix')
    # Establish unchanged references and every 0% anchor first, then the full
    # ascending grid. This affects scheduling order only, never case identity.
    cases.sort(key=lambda c:(c['case_role']!='unchanged_raw_reference',c['inference_sparsity_percent'],
                             c['dataset'],c['architecture'],c['representation'],c['trained_execution'],c['seed']))
    workers=[]
    for case in cases:
        for execution in (('compact','dense_masked') if case['inference_execution']=='compact' else ('dense',)):
            for batch in (1,8):
                work=dict(test_config_hash=case['test_config_hash'],execution=execution,batch_size=batch,shard=0,num_shards=1)
                work['worker_id']=digest(work);workers.append(work)
    cell_count=sum(16 if x['dataset']=='mnist' else 76 for x in cases)
    manifest=dict(study_id=STUDY_ID,schema_version=1,created=now(),source_hash=source_hash,source_path=str(source_path),
        base_source_hash=BASE_SOURCE_HASH,parent_root=str(BASE_ROOT),parent_study_id='training_sparse_testing_sparse',
        raw_policy='raw_pixels_sweep_with_unchanged_dense_reference',raw_policy_provisional=True,
        inference_sparsities=list(range(0,100,10)),seeds=[0,1,2],sources=sources,cases=cases,latency_workers=workers,
        expected=dict(source_checkpoints=60,evaluation_cases=612,evaluation_cells=cell_count,evaluation_predictions=cell_count*10000,
                      latency_workers=len(workers),latency_cells=140304),
        timing=dict(batch_sizes=[1,8],warmup=50,repeats=200,scopes=['gpu_raw_to_logits','host_raw_to_logits','loader_to_cpu_prediction','cached_input_model_only_diagnostic']))
    manifest['manifest_hash']=digest(manifest)
    for case in cases:
        atomic_json(ROOT/'configs'/f"{case['test_config_hash']}.json",case)
    atomic_json(existing,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    identity,path=snapshot();manifest=build(identity,path)
    print(json.dumps({'source_hash':identity,'source_path':str(path),'expected':manifest['expected']}))

if __name__=='__main__':main()
