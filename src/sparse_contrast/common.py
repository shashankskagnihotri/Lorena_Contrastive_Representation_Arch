"""Small, explicit persistence and reproducibility helpers."""
from __future__ import annotations
import contextlib, hashlib, json, os, random, socket, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ.get('SPARSE_CONTRAST_ROOT', Path(os.environ.get('SPARSE_CONTRAST_PROJECT_ROOT', Path(__file__).resolve().parents[2])) / 'experiments' / 'training_sparse_testing_sparse')).resolve()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_torch(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent); os.close(fd)
    try:
        torch.save(value,tmp)
        with open(tmp,'rb') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

@contextlib.contextmanager
def directory_lock(path):
    """Atomic mkdir ownership on shared Ceph; never automatically steal a lease."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.mkdir()  # Existing owner is a visible failure, not duplicate execution.
    atomic_json(path/'owner.json',{'host':socket.gethostname(),'pid':os.getpid(),'job_id':os.getenv('SLURM_JOB_ID'),'time':time.time()})
    try: yield
    finally:
        (path/'owner.json').unlink(); path.rmdir()

def seed_all(seed):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process launch')
    if os.environ.get('PYTHONHASHSEED') is None: raise RuntimeError('Set PYTHONHASHSEED before process launch')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])

def initialize_matched(model,seed):
    """Name-keyed init decouples backbone initialization from stem channel count."""
    with torch.no_grad():
        for name,p in model.named_parameters():
            value=int(hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest()[:15],16)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(value)
                if p.ndim>=2 or any(k in name for k in ('pos_embed','cls_token','relative_position_bias')):
                    torch.nn.init.trunc_normal_(p,std=.02)
                elif name.endswith('weight'): p.fill_(1.)
                else: p.zero_()

def parameter_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def hardware():
    return dict(host=socket.gethostname(),job_id=os.getenv('SLURM_JOB_ID'),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,tf32=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

def tb_dir(config,phase):
    p=config['sparsity_percent']; pct='NA' if p is None else str(p)
    return ROOT/'outputs/tensorboard'/config['protocol']/config['dataset']/config['architecture']/config['representation']/config['execution']/('p'+pct)/('seed'+str(config['seed']))/config['config_hash']/phase

def run_dir(config): return ROOT/'outputs/runs'/config['protocol']/config['registry_id']/config['config_hash']

def append_jsonl(path,row):
    with open(path,'a') as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n'); f.flush()

def stop_requested(): return (ROOT/'STOP').exists()


def flatten_metrics(value,prefix=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from flatten_metrics(v,prefix+'/'+str(k) if prefix else str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):yield from flatten_metrics(v,prefix+'/'+str(i))
    elif torch.is_tensor(value):yield prefix,value.detach()
    elif isinstance(value,(int,float)) and not isinstance(value,bool):yield prefix,torch.tensor(float(value))

SCIENTIFIC_SOURCE_FILES = (
    'sparse_contrast/__init__.py', 'sparse_contrast/data.py',
    'sparse_contrast/frontend.py', 'sparse_contrast/normalization.py',
    'sparse_contrast/pipeline.py', 'sparse_contrast/models.py',
    'sparse_contrast/train.py', 'sparse_contrast/evaluate.py',
    'sparse_contrast/metrics.py', 'sparse_contrast/visualize.py',
    'requirements.lock.txt', 'environment.yml', 'conda-explicit.txt',
    'provenance/source_context.json', 'provenance/original_frontend.py',
)


def executing_source_hash(config):
    """Verify exact execution bytes and an explicit orchestration-only migration.

    Existing training/evaluation workers keep their original immutable source.
    A revised benchmark/report controller may consume those checkpoints only
    with a recorded compatibility approval and byte-identical scientific code.
    """
    training_hash = config['source_hash']
    training_path = ROOT/'outputs/source'/training_hash
    training_manifest = json.loads((training_path/'source_manifest.json').read_text())
    if digest(training_manifest) != training_hash:
        raise ValueError('Code snapshot identity mismatch')
    executing = Path(__file__).resolve().parents[1]
    execution_manifest_path = executing/'source_manifest.json'
    if all((executing/name).is_file() and file_hash(executing/name)==want
           for name,want in training_manifest.items()):
        manifest = training_manifest
    elif execution_manifest_path.exists():
        manifest = json.loads(execution_manifest_path.read_text())
    else:
        campaign_path = ROOT/'outputs/campaign.json'
        if not campaign_path.exists():
            raise ValueError('Executing source changed without an approved orchestration snapshot')
        campaign = json.loads(campaign_path.read_text())
        approved_hash = campaign.get('orchestration_source_hash')
        if not approved_hash:
            raise ValueError('Executing source changed without an approved orchestration snapshot')
        manifest = json.loads((ROOT/'outputs/source'/approved_hash/'source_manifest.json').read_text())
        if digest(manifest) != approved_hash:
            raise ValueError('Orchestration snapshot identity mismatch')
    actual_hash = digest(manifest)
    for name,want in manifest.items():
        if not (executing/name).is_file() or file_hash(executing/name)!=want:
            raise ValueError(f'Executing source changed: {name}; use the matching immutable snapshot')
    if actual_hash != training_hash:
        approval_path = ROOT/'outputs/source_compatibility'/f'{actual_hash}.json'
        if not approval_path.exists():
            raise ValueError('Executing source changed without a registered training-source compatibility approval')
        approval = json.loads(approval_path.read_text())
        if (approval.get('schema_version')!=1 or approval.get('training_source_hash')!=training_hash
                or approval.get('execution_source_hash')!=actual_hash
                or set(approval.get('unchanged_scientific_files',[]))!=set(SCIENTIFIC_SOURCE_FILES)):
            raise ValueError('Unapproved training/measurement source compatibility')
        for name in SCIENTIFIC_SOURCE_FILES:
            if manifest.get(name)!=training_manifest.get(name) or name not in manifest:
                raise ValueError(f'Scientific source changed across orchestration migration: {name}')
        # The scope is executable policy for the new orchestration, not a change
        # to any retained run's scientific configuration or checkpoint identity.
        if 'experiment_scope.json' in manifest and file_hash(ROOT/'experiment_scope.json')!=manifest['experiment_scope.json']:
            raise ValueError('Live scope differs from the approved orchestration snapshot')
    return actual_hash


def validate_config_artifacts(config):
    """Fail on changed config, scientific inputs, installed packages, or code."""
    from importlib.metadata import version
    if digest({k:v for k,v in config.items() if k!='config_hash'})!=config['config_hash']:
        raise ValueError('Resolved configuration digest mismatch')
    for name,key in [('datasets_manifest.json','data_manifest_sha256'),('requirements.lock.txt','environment_hash')]:
        if file_hash(ROOT/name)!=config[key]: raise ValueError(f'{name} identity differs from resolved configuration')
    for line in (ROOT/'requirements.lock.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name,expected=line.split('==',1)
            if version(name)!=expected: raise ValueError(f'Installed dependency changed: {name}')
    executing_source_hash(config)

def read_recoverable_jsonl(path):
    """Recover only an incomplete terminal append, preserving its exact bytes."""
    path=Path(path); raw=path.read_bytes(); lines=raw.splitlines(keepends=True); rows=[]
    for index,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError):
            if index!=len(lines)-1 or line.endswith(b'\n'): raise
            archive=path.with_name(path.name+f'.incomplete-tail.{time.time_ns()}')
            archive.write_bytes(raw)
            path.write_bytes(b''.join(lines[:-1]))
            atomic_json(path.with_name(path.name+f'.recovery.{time.time_ns()}.json'),{'reason':'incomplete terminal JSONL append after interruption','archived':str(archive),'retained_complete_rows':len(rows)})
    return rows
