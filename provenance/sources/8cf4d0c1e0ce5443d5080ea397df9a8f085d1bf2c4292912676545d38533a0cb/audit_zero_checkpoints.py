"""Read-only full final-checkpoint audit; no training or benchmark samples run."""
from pathlib import Path
import gc
import json
import os
import sys
import time
import torch

PROJECT=Path('/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch')
BASE=PROJECT/'sparse_contrast_benchmark_large_batch'
SOURCE='7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'
sys.path.insert(0,str(BASE/'outputs/source'/SOURCE))
os.environ['SPARSE_CONTRAST_ROOT']=str(BASE)
from sparse_contrast.common import atomic_json,file_hash,validate_config_artifacts,run_dir,seed_all
from sparse_contrast.pipeline import Classifier
from sparse_contrast.train import load_stats

if not os.getenv('SLURM_JOB_ID'):
    raise RuntimeError('Full checkpoint audit requires a CPU allocation')
seed_all(0)
campaign=json.loads((BASE/'outputs/campaign.json').read_text())
selected=[r for r in campaign['main'] if r['config']['representation']=='raw' or r['config']['sparsity_percent']==0]
if len(selected)!=60:raise ValueError('Expected exactly60 source checkpoints')
rows=[]
for item in selected:
    c=item['config'];validate_config_artifacts(c)
    rd=run_dir(c);done=json.loads((rd/'completed.json').read_text());sha=file_hash(rd/'final.pt')
    if sha!=done['checkpoint_sha256'] or done['config_hash']!=c['config_hash'] or done['epochs']!=c['recipe']['epochs']:
        raise ValueError('Final checkpoint provenance mismatch')
    state=torch.load(rd/'final.pt',map_location='cpu',weights_only=False)
    if state['config_hash']!=c['config_hash'] or state['normalization_hash']!=c['normalization_hash'] or state['next_epoch']!=c['recipe']['epochs']:
        raise ValueError('Embedded final checkpoint identity is wrong')
    model=Classifier(c,load_stats(c))
    expected_buffers={n:b.clone() for n,b in model.named_buffers() if n in ('mean','std') or n.startswith('frontend.')}
    model.load_state_dict(state['model'],strict=True)
    for name,value in expected_buffers.items():
        torch.testing.assert_close(dict(model.named_buffers())[name],value,rtol=0,atol=0)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite learned checkpoint')
    rows.append({'config_hash':c['config_hash'],'registry_id':c['registry_id'],'checkpoint_path':str(rd/'final.pt'),
                 'checkpoint_sha256':sha,'normalization_hash':c['normalization_hash'],'strict_load':True,'frozen_buffers_exact':True,
                 'source_hash':c['source_hash'],'execution_source_hash':SOURCE,'epochs':done['epochs']})
    print(json.dumps({'completed':len(rows),'total':60,'registry_id':c['registry_id']}),flush=True)
    del state,model,expected_buffers
    gc.collect()
output=PROJECT/'training_dense_testing_sparse/outputs/checks/zero_checkpoint_audit.json'
atomic_json(output,{'state':'complete','count':len(rows),'job_id':os.environ['SLURM_JOB_ID'],'time':time.time(),
                    'script_sha256':file_hash(__file__),'checkpoints':rows,
                    'scope':'Full final checkpoint bytes and strict model loading; no new evaluation predictions or training.'})
