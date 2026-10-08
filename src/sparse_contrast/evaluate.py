"""Final-checkpoint only, resumable full official clean/corruption evaluation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .common import *
from .pipeline import Classifier
from .train import load_stats,loader,evaluate,log_counts,batch_settings
from .data import load_clean,CorruptionDataset,corruption_cells
from .metrics import aggregate_cells

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    config=json.loads(Path(a.config).read_text()); validate_config_artifacts(config); seed_all(config['seed']); rd=run_dir(config)
    batches=batch_settings(config)
    done=json.loads((rd/'completed.json').read_text()); checkpoint=rd/'final.pt'
    if file_hash(checkpoint)!=done['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    out=rd/'evaluation'; out.mkdir(exist_ok=True)
    with directory_lock(out/'writer.lock'):
        model=Classifier(config,load_stats(config)); state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['config_hash']!=config['config_hash']: raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(state['model'],strict=True); device=torch.device(a.device); model.to(device).eval()
        writer=SummaryWriter(str(tb_dir(config,'evaluation')),flush_secs=30); rows=[]
        try:
            all_cells=[('clean','test')]+list(corruption_cells(config['dataset']))
            for index,(corruption,severity) in enumerate(all_cells):
                if stop_requested(): raise InterruptedError('STOP marker before evaluation cell')
                dest=out/f'{corruption}__{severity}.json'; predictions=dest.with_suffix('.npz')
                if dest.exists():
                    row=json.loads(dest.read_text())
                    if row['config_hash']!=config['config_hash'] or row.get('normalization_hash')!=config.get('normalization_hash') or file_hash(dest.with_suffix('.tokens.jsonl'))!=row['tokens_sha256'] or row['checkpoint_sha256']!=done['checkpoint_sha256'] or file_hash(predictions)!=row['predictions_sha256'] or row['total']!=10000: raise ValueError('Invalid saved evaluation cell')
                else:
                    ds=load_clean(config['dataset'],'test') if corruption=='clean' else CorruptionDataset(config['dataset'],corruption,severity)
                    data=loader(ds,batch_size=batches['eval_batch_size'],num_workers=batches['eval_num_workers'])[0]
                    row=evaluate(model,data,config,device,collect=True); ids,labels,pred=row.pop('predictions')
                    if len(ids)!=10000 or len(set(ids))!=10000: raise ValueError('Official cell coverage invalid')
                    token_records=row.pop('token_records')
                    token_path=dest.with_suffix('.tokens.jsonl')
                    token_path.write_text(''.join(json.dumps(r)+'\n' for r in token_records))
                    row['tokens_sha256']=file_hash(token_path)
                    tmp=predictions.with_suffix('.npz.tmp')
                    with open(tmp,'wb') as f: np.savez_compressed(f,sample_id=ids,label=labels,prediction=pred,correct=labels==pred)
                    os.replace(tmp,predictions)
                    row.update(corruption=corruption,severity=severity,checkpoint_sha256=done['checkpoint_sha256'],normalization_hash=config.get('normalization_hash'),config_hash=config['config_hash'],predictions_sha256=file_hash(predictions))
                    atomic_json(dest,row)
                if corruption!='clean':
                    from .visualize import log_corruption_panel
                    log_corruption_panel(model,config,writer,CorruptionDataset(config['dataset'],corruption,severity),index,device)
                rows.append(row); log_counts(writer,f'{corruption}/severity_{severity}',row,index)
                writer.add_scalar('evaluation_axis/cell_index',index,index)
                for name,metric in row['support'].items():
                    for key,value in metric.items(): writer.add_scalar(f'{corruption}/severity_{severity}/support/{name}/{key}',value,index)
                from .visualize import confusion_figure
                if corruption=='clean': writer.add_figure('clean/confusion',confusion_figure(row['confusion'],'Clean test'),index,close=True)
                writer.flush(); print(json.dumps({'corruption':corruption,'severity':severity,'accuracy':row['accuracy'],'total':row['total']}),flush=True)
            summary=aggregate_cells(rows[1:],corruption_cells(config['dataset']))
            summary.update(clean=rows[0],config=config,checkpoint_sha256=done['checkpoint_sha256'],training_count=done['training_count'],state='complete')
            for c,error in summary['per_corruption_error'].items(): writer.add_scalar(f'corruption_mean/{c}',error,0)
            writer.add_scalar('corruption_mean/mCE_raw',summary['mCE_raw'],0)
            confusion=np.sum([r['confusion'] for r in rows[1:]],axis=0)
            writer.add_figure('corruption_mean/confusion',confusion_figure(confusion,'Aggregate corruptions'),0,close=True)
            for severity in sorted({str(s) for _,s in corruption_cells(config['dataset'])}): writer.add_scalar(f'corruption_mean/severity_{severity}',float(np.mean([r['error'] for r in rows[1:] if str(r['severity'])==severity])),0)
            atomic_json(out/'summary.json',summary)
        finally: writer.flush(); writer.close()
if __name__=='__main__': main()
