#!/usr/bin/env python3
"""Full clean-training coefficient-to-patch diagnostic; no training sweep."""
import argparse,csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sparse_contrast.common import ROOT,atomic_json,seed_all
from sparse_contrast.data import load_clean
from sparse_contrast.frontend import Frontend,sparsify,patch_support
from sparse_contrast.scope import read_scope,representations

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');a=p.parse_args();seed_all(0)
    scope=read_scope(ROOT)
    out=ROOT/'outputs/preprocessing_pilot';out.mkdir(parents=True,exist_ok=True);rows=[]
    writer=SummaryWriter(str(ROOT/'outputs/tensorboard/preprocessing_pilot'))
    try:
        for name in ['mnist','cifar10']:
            ds=load_clean(name,'train');frontends={r:Frontend(r).to(a.device) for r in representations(name,scope)};values={};ids=[]
            with torch.inference_mode():
                for image,label,identity in DataLoader(ds,batch_size=128,shuffle=False,num_workers=0):
                    raw=image.to(a.device).float()/255;ids.extend(identity)
                    for rep,f in frontends.items():
                        c=f(raw)
                        for pct in [0,20,40,60,80]:
                            cs=sparsify(c,pct);zero=(cs==0).float().flatten(1).mean(1).cpu().numpy()
                            for patch in [1,2,4]:
                                active=patch_support(cs,patch,0).float();frac=(1-active.mean(1)).cpu().numpy()
                                values.setdefault((rep,pct,patch),[]).append(np.column_stack([zero,frac]))
            if len(ids)!=len(set(ids)) or len(ids)!=len(ds):raise ValueError('Preprocessing pilot coverage')
            save={'sample_ids':np.asarray(ids)}
            for (rep,pct,patch),batches in values.items():
                v=np.concatenate(batches);key=f'{rep}_p{pct}_patch{patch}';save[key]=v
                row=dict(dataset=name,representation=rep,sparsity_percent=pct,patch_size=patch,images=len(ids),achieved_zero_fraction=float(v[:,0].mean()),empty_patch_fraction=float(v[:,1].mean()),empty_patch_p95=float(np.quantile(v[:,1],.95)))
                rows.append(row)
                for metric in ['achieved_zero_fraction','empty_patch_fraction','empty_patch_p95']:writer.add_scalar(f'{name}/{rep}/patch{patch}/{metric}',row[metric],pct)
            np.savez_compressed(out/f'{name}.npz',**save)
        with (out/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        atomic_json(out/'completed.json',{'state':'verified','rows':len(rows),'training_counts':{'mnist':55000,'cifar10':45000},'patch_sizes':[1,2,4],'token_epsilon':0.})
    finally:writer.flush();writer.close()
if __name__=='__main__':main()
