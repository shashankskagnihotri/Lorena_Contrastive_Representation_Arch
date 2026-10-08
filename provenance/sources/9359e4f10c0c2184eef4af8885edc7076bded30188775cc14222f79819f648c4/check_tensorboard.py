"""Verify event scalars against durable sample-weighted training records."""
import argparse,json,math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sparse_contrast.common import ROOT,run_dir,tb_dir,atomic_json

def verify(config):
    root=tb_dir(config,'train');events={};images=[];histograms=[]
    for directory in sorted({p.parent for p in root.rglob('events.out.tfevents.*')}):
        a=EventAccumulator(str(directory),size_guidance={'scalars':0,'images':1,'histograms':1});a.Reload()
        for tag in a.Tags()['scalars']:
            for e in a.Scalars(tag):events[(tag,e.step)]=e.value
        images.extend(a.Tags()['images']);histograms.extend(a.Tags()['histograms'])
    rd=run_dir(config);checked=0
    for line in (rd/'epochs.jsonl').read_text().splitlines():
        row=json.loads(line)
        for key,prefix in [('train_online','train_online_epoch'),('validation','heldout_val_epoch')]:
            if row[key] is None:continue
            for metric in ('loss','accuracy','error','correct','total'):
                expected=row[key][metric];actual=events.get((f'{prefix}/{metric}',row['epoch']))
                if actual is None or not math.isclose(expected,actual,rel_tol=1e-6,abs_tol=1e-7):raise AssertionError((prefix,metric,row['epoch'],expected,actual))
                checked+=1
    for line in (rd/'steps.jsonl').read_text().splitlines():
        row=json.loads(line)
        for metric in ('loss','accuracy','gradient_norm','samples_seen'):
            expected=row[metric];actual=events.get((f'train_step/{metric}',row['optimizer_step']))
            if actual is None or not math.isclose(expected,actual,rel_tol=1e-6,abs_tol=1e-7):raise AssertionError((metric,row['optimizer_step'],expected,actual))
            checked+=1
    if not images:raise AssertionError('Training images absent')
    if (rd/'completed.json').exists() and not histograms:raise AssertionError('Completed run has no parameter/gradient histograms')
    result={'passed':True,'config_hash':config['config_hash'],'scalar_values_checked':checked,'image_tags':sorted(set(images)),'histogram_tags':len(set(histograms))}
    atomic_json(rd/'tensorboard_verified.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--all',action='store_true');a=p.parse_args()
    if bool(a.config)==bool(a.all):p.error('choose --config or --all')
    paths=[Path(a.config)] if a.config else sorted((ROOT/'outputs/runs').rglob('config.json'))
    results=[verify(json.loads(path.read_text())) for path in paths]
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
