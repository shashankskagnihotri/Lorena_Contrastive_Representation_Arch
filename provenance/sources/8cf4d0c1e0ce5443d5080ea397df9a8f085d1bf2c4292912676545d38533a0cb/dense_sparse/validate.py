"""Allocated final validation: tests, full registered matrix, and audit binding."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from .common import *
from sparse_contrast.data import corruption_cells


def main():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Validation requires an allocated CPU job')
    manifest=load_manifest();source_hash=verify_source()
    if source_hash!=manifest['source_hash']:
        raise ValueError('Validation source mismatch')
    output=ROOT/'outputs/checks';output.mkdir(parents=True,exist_ok=True)
    xml_path=output/f"integration-{os.environ['SLURM_JOB_ID']}.xml"
    source=Path(__file__).resolve().parents[1]
    subprocess.run([PYTHON,'-m','pytest',str(source/'tests'),'-q','--junitxml='+str(xml_path)],check=True,cwd=source)
    suites=ET.parse(xml_path).getroot()
    counts={key:sum(int(node.attrib.get(key,0)) for node in suites.iter('testsuite')) for key in ('tests','failures','errors','skipped')}
    if counts['tests']<=0 or any(counts[k] for k in ('failures','errors','skipped')):
        raise ValueError('Allocated source validation incomplete')
    sources={x['training_config_hash']:x for x in manifest['sources']}
    if len(sources)!=60:
        raise ValueError('Expected all60 distinct trained source checkpoints')
    by_source=defaultdict(list)
    for case in manifest['cases']:
        load_case(case)
        if read_json(ROOT/'configs'/f"{case['test_config_hash']}.json")!=case:
            raise ValueError('Resolved inference case file differs from manifest')
        if case['dataset']=='mnist' and case['representation'] not in ('raw','grayscale'):
            raise ValueError('Excluded MNIST color case')
        original=read_json(case['training_config_path'])
        if original['config_hash']!=case['training_config_hash'] or digest({k:v for k,v in original.items() if k!='config_hash'})!=original['config_hash']:
            raise ValueError('Original training configuration changed')
        by_source[case['training_config_hash']].append(case)
    for identity,cases in by_source.items():
        sweep=[c for c in cases if c['case_role']=='sparsity_sweep']
        if len(sweep)!=10 or {c['inference_sparsity_percent'] for c in sweep}!=set(range(0,100,10)):
            raise ValueError('Requested complete test sparsity grid is missing')
        if any(c['inference_execution']!='compact' for c in sweep):
            raise ValueError('Sweep execution must be explicit compact')
        raw=sources[identity]['representation']=='raw'
        if len(cases)!=(11 if raw else 10):
            raise ValueError('Incorrect unchanged-reference coverage')
        if raw and not any(c['case_role']=='unchanged_raw_reference' and c['sparsification_domain']=='none' and c['inference_execution']=='dense' for c in cases):
            raise ValueError('Raw baseline reference missing')
    cells=sum(1+len(corruption_cells(c['dataset'])) for c in manifest['cases'])
    expected_work={digest(dict(test_config_hash=c['test_config_hash'],execution=e,batch_size=b,shard=0,num_shards=1))
        for c in manifest['cases'] for e in (('compact','dense_masked') if c['inference_execution']=='compact' else ('dense',)) for b in (1,8)}
    if expected_work!={w['worker_id'] for w in manifest['latency_workers']} or len(expected_work)!=len(manifest['latency_workers']):
        raise ValueError('Timing work matrix is incomplete or duplicated')
    if cells!=manifest['expected']['evaluation_cells'] or len(manifest['cases'])!=manifest['expected']['evaluation_cases']:
        raise ValueError('Declared coverage mismatch')
    audit_path=output/'zero_checkpoint_audit.json';audit=read_json(audit_path)
    if audit['state']!='complete' or len(audit['checkpoints'])!=60:
        raise ValueError('Full original checkpoint audit required')
    if {(x['config_hash'],x['checkpoint_sha256']) for x in audit['checkpoints']}!={(x['training_config_hash'],x['checkpoint_sha256']) for x in sources.values()}:
        raise ValueError('Checkpoint audit differs from source set')
    result=dict(state='passed',time=now(),job_id=os.environ['SLURM_JOB_ID'],source_hash=source_hash,
                manifest_hash=manifest['manifest_hash'],tests=counts,xml_path=str(xml_path),xml_sha256=file_hash(xml_path),
                checkpoint_audit_sha256=file_hash(audit_path),expected=manifest['expected'])
    atomic_json(output/'validation.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
