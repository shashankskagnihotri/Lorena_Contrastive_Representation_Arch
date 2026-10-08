"""Create separate optional full-data refit configs; disabled unless --enable."""
import argparse,json
from sparse_contrast.common import ROOT,atomic_json
from sparse_contrast.campaign import resolve_config,source_snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--enable',action='store_true');a=p.parse_args()
    if not a.enable:p.error('Optional full_train_refit is disabled; --enable is required')
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text())
    if manifest['state']!='frozen':raise RuntimeError('Primary recipes must be frozen before full-data refits')
    source,_=source_snapshot();runs=[]
    for entry in manifest['runs']:
        run=dict(entry,protocol='full_train_refit');config,path=resolve_config(run,source)
        runs.append({'config':config,'config_path':str(path),'state':'planned'})
    atomic_json(ROOT/'full_train_refit_manifest.json',{'protocol':'full_train_refit','enabled':True,'runs':runs,'validation':'none','training_counts':{'mnist':60000,'cifar10':50000}})
if __name__=='__main__':main()
