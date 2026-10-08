"""Run the final frozen-source checks and reconcile previously verified diagnostics."""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparse_contrast.common import ROOT, atomic_json, file_hash
from sparse_contrast.campaign import verify_source_snapshot

source = Path(__file__).resolve().parents[1]
source_hash = verify_source_snapshot(source)
checks = ROOT / 'outputs/checks'
os.environ['SPARSE_BENCH_TEST_DEVICE'] = 'cuda'
subprocess.run([sys.executable, '-m', 'pytest', 'tests',
                '--junitxml=' + str(checks / 'final-tests.xml')], cwd=source, check=True)
suites = list(ET.parse(checks / 'final-tests.xml').getroot().iter('testsuite'))
totals = {key: sum(int(s.get(key, 0)) for s in suites)
          for key in ('tests', 'failures', 'errors', 'skipped')}
assert totals['tests'] > 0 and not any(totals[k] for k in ('failures', 'errors', 'skipped')), totals
# The final allocator/timing changes get the requested integration check on the
# exact frozen source. The parent has not initialized CUDA, preserving isolation.
subprocess.run([sys.executable, str(source / 'smoke.py'), '--device', 'cuda'],
               cwd=source, check=True)
import torch
from sparse_contrast.common import seed_all
from sparse_contrast.benchmark import device_metadata, scheduler_gpu_selection, gpu_isolation
seed_all(2026)
device = torch.device('cuda')
metadata = device_metadata(device)
selection = scheduler_gpu_selection(metadata)
isolation = gpu_isolation(device)
assert selection['gpu_partition'] == os.environ['SLURM_JOB_PARTITION']
atomic_json(checks / 'allocated-gpu-selection.json', dict(
    passed=True, hardware=metadata, scheduler_selection=selection, isolation=isolation))
subprocess.run([sys.executable, str(source / 'scripts/check_tensorboard_server.py')],
               cwd=source, check=True)
core = json.loads((checks / 'core-completed.json').read_text())
smoke = json.loads((checks / 'smoke-completed.json').read_text())
server = json.loads((checks / 'tensorboard-server.json').read_text())
overfit = sorted((checks / 'overfit').glob('*/*/*/completed.json'))
records = [json.loads(p.read_text()) for p in overfit]
assert core['passed'] and smoke['passed'] and server['passed']
assert {(r['dataset'], r['architecture']) for r in records} == {
    (d, a) for d in ('mnist', 'cifar10') for a in ('vit_small', 'swin_tiny')}
assert len(records) == 4 and all(
    r['state'] == 'verified' and r['final']['train_monitor_accuracy'] == 1.0
    and r['final']['train_monitor_loss'] < 0.1
    and len(r['sample_ids']) == len(set(r['sample_ids'])) == 20 for r in records)
subprocess.run([sys.executable, str(source / 'report.py')], cwd=source, check=True)
runtime = dict(passed=True, source_hash=source_hash, job_id=os.environ['SLURM_JOB_ID'],
               tests=totals, environment_hash=file_hash(ROOT / 'requirements.lock.txt'),
               integration_smoke_job=os.environ['SLURM_JOB_ID'],
               scope='Frozen-source full suite, training/validation/checkpoint/timing integration, scheduler GPU selection, TensorBoard server and report',
               smoke_config=smoke['config_path'])
atomic_json(checks / 'runtime-completed.json', runtime)
atomic_json(checks / 'completed.json', dict(passed=True, core=core, runtime=runtime,
            smoke_artifact=str(checks / 'smoke-completed.json'),
            overfit_artifacts=[str(p) for p in overfit]))
print(json.dumps(runtime, indent=2))
