"""Explicitly requested end-to-end diagnostic; excluded from study aggregates."""
import argparse,gc,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from sparse_contrast.common import *
from sparse_contrast.campaign import resolve_config,source_snapshot
from sparse_contrast.pipeline import Classifier
from sparse_contrast.train import train,load_stats,loader,evaluate,log_counts
from sparse_contrast.data import load_clean

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');a=p.parse_args()
    source_manifest=Path(__file__).resolve().parent/'source_manifest.json'
    source=digest(json.loads(source_manifest.read_text())) if source_manifest.exists() else source_snapshot()[0]
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text())
    base=next(r for r in manifest['runs'] if r['dataset']=='mnist' and r['architecture']=='vit_small' and r['representation']=='single_color' and r['sparsity_percent']==80 and r['seed']==0)
    config,_=resolve_config(base,source)
    config.update(role='smoke',registry_id='smoke/'+base['registry_id'],subset_ids=list(range(8)),disable_augmentation=True)
    config['recipe'].update(epochs=1,warmup_epochs=0,histogram_epoch_interval=1,log_step_interval=1)
    config.pop('config_hash');config['config_hash']=digest(config);path=ROOT/'configs/resolved'/f"{config['config_hash']}.json";atomic_json(path,config)
    train(config,a.device);rd=run_dir(config);model=Classifier(config,load_stats(config));model.load_state_dict(torch.load(rd/'final.pt',map_location='cpu',weights_only=False)['model'],strict=True);device=torch.device(a.device);model.to(device)
    result=evaluate(model,loader(Subset(load_clean('mnist','val'),list(range(16))))[0],config,device,collect=True)
    ids,labels,pred=result.pop('predictions');result.pop('token_records');atomic_json(rd/'smoke_evaluation.json',result)
    writer=SummaryWriter(str(tb_dir(config,'evaluation')));log_counts(writer,'smoke_heldout_diagnostic_16',result,0);writer.flush();writer.close()
    del model
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    from sparse_contrast.benchmark import run_worker
    run_worker(config,'compact',1,a.device,smoke=True)
    from check_tensorboard import verify
    event_verification=verify(config)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags={'scalars':set(),'images':set(),'histograms':set()}
    for directory in sorted({p.parent for p in tb_dir(config,'train').parent.rglob('events.out.tfevents.*')}):
        acc=EventAccumulator(str(directory));acc.Reload()
        for key in tags:tags[key].update(acc.Tags().get(key,[]))
    if not all(tags.values()):raise AssertionError(f'Missing TensorBoard families: {tags}')
    if not any('heldout_val' in k for k in tags['scalars']):raise AssertionError('Missing actual validation events')
    if not any('gpu_raw_to_logits' in k for k in tags['scalars']):raise AssertionError('Missing actual latency events')
    atomic_json(ROOT/'outputs/checks/smoke-completed.json',{'passed':True,'config_path':str(path),'tags':{k:sorted(v) for k,v in tags.items()}})
if __name__=='__main__':main()
