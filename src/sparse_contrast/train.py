"""Deterministic epoch-boundary training with full held-out validation."""
from __future__ import annotations
import argparse, json, math, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from .common import *
from .data import load_clean, StatelessAugment, EpochSampler
from .pipeline import Classifier
from .normalization import load_normalization
from .metrics import ClassificationCounts

def autocast(config,device):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16' and device.type=='cuda')

def load_stats(config):
    if config['representation']=='raw': return None
    path=ROOT/'normalization'/f"{config['dataset']}_{config['representation']}_{config['protocol']}.json"
    from .frontend import Frontend
    ds=load_clean(config['dataset'],'train',protocol=config['protocol'])
    expected={'dataset':config['dataset'],'representation':config['representation'],'protocol':config['protocol'],'split_hash':ds.split_hash,'frontend_config':Frontend(config['representation']).config,'partition':'clean_train','sparsity_percent_for_fit':0,'augmentation':'none','input_range':[0.,1.]}
    _,stats=load_normalization(path,expected=expected)
    if stats['artifact_sha256']!=config['normalization_hash']: raise ValueError('Normalization identity changed')
    return stats

def _integer_setting(value,name,minimum):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value

def batch_settings(config):
    """Explicit physical batches; an old recipe retains its 8/8, zero-worker behavior."""
    recipe=config['recipe']
    batch=_integer_setting(recipe.get('batch_size',8),'batch_size',1)
    effective=_integer_setting(recipe.get('effective_batch_size',batch),'effective_batch_size',1)
    accumulation=_integer_setting(recipe.get('gradient_accumulation',1),'gradient_accumulation',1)
    if effective!=batch or accumulation!=1:
        raise ValueError('This trainer requires effective_batch_size == batch_size and no gradient accumulation')
    workers=_integer_setting(recipe.get('num_workers',0),'num_workers',0)
    return dict(train_batch_size=batch,
        eval_batch_size=_integer_setting(recipe.get('eval_batch_size',8),'eval_batch_size',1),
        train_num_workers=workers,
        eval_num_workers=_integer_setting(recipe.get('eval_num_workers',workers),'eval_num_workers',0))

def loader(dataset,shuffle_seed=None,batch_size=8,num_workers=0):
    batch_size=_integer_setting(batch_size,'batch_size',1)
    num_workers=_integer_setting(num_workers,'num_workers',0)
    generator=torch.Generator().manual_seed(71235)
    sampler=EpochSampler(dataset,shuffle_seed) if shuffle_seed is not None else None
    # Workers return CPU tensors. Fresh copies each epoch receive the current
    # StatelessAugment.epoch; persistent copies would retain a stale epoch.
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,shuffle=False,num_workers=num_workers,
        persistent_workers=False,pin_memory=torch.cuda.is_available(),generator=generator),sampler

class TrainingDiagnostics:
    """Original float32 batch sums, accumulated in float64 without scalar GPU reads."""
    def __init__(self):
        self.groups={}

    def update(self,metrics):
        grouped={}
        for name,value in flatten_metrics(metrics):
            value=value.detach().float()
            grouped.setdefault(value.device,[]).append((name,value.sum(),value.numel()))
        for device,rows in grouped.items():
            names=tuple(row[0] for row in rows)
            sums=torch.stack([row[1] for row in rows]).to(torch.float64)
            counts=[row[2] for row in rows]
            if device not in self.groups:
                self.groups[device]=dict(names=names,sums=sums,counts=counts)
            else:
                group=self.groups[device]
                if names!=group['names']: raise ValueError('Training diagnostic fields changed within an epoch')
                group['sums'].add_(sums)
                group['counts']=[left+right for left,right in zip(group['counts'],counts)]

    def means(self):
        result={}
        for group in self.groups.values():
            for name,total,count in zip(group['names'],group['sums'].cpu().tolist(),group['counts']):
                if count<=0: raise ValueError(f'No observations for training diagnostic {name}')
                result[name]=total/count
        return result

@torch.inference_mode()
def evaluate(model,data,config,device,collect=False):
    was_training=model.training; model.eval(); counts=ClassificationCounts(); ids=[]; labels=[]; preds=[]; token_records=[]; diag={}; started=time.perf_counter()
    model.collect_diagnostics=True
    for raw,y,identity in data:
        raw=raw.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with autocast(config,device): logits=model(raw); loss=torch.nn.functional.cross_entropy(logits.float(),y,reduction='sum')
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite evaluation logits')
        p=logits.argmax(1).cpu().numpy(); yy=y.cpu().numpy(); counts.add(yy,p,float(loss))
        if collect:
            ids.extend(identity); labels.extend(yy); preds.extend(p)
            token_records.append(dict(sample_ids=list(identity),metrics={name:v.detach().cpu().tolist() for name,v in flatten_metrics({**model.latest_diagnostics,**model.backbone.latest_telemetry})}))
        for name,v in flatten_metrics({**model.latest_diagnostics,**model.backbone.latest_telemetry}):
            if torch.is_tensor(v):
                value=v.detach().float().cpu().numpy().reshape(-1)
                diag.setdefault(name,[]).extend(value.tolist())
    model.collect_diagnostics=False
    result=counts.result(); result['duration_seconds']=time.perf_counter()-started
    result['support']={k:dict(mean=float(np.mean(v)),min=float(np.min(v)),max=float(np.max(v)),p95=float(np.quantile(v,.95))) for k,v in diag.items() if v}
    if collect:
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate sample identities')
        result['token_records']=token_records
        result['predictions']=(np.asarray(ids),np.asarray(labels,dtype=np.uint8),np.asarray(preds,dtype=np.uint8))
    model.train(was_training)
    return result

def log_counts(writer,prefix,row,step):
    for key in ('loss','accuracy','error','correct','total'):
        writer.add_scalar(f'{prefix}/{key}',row[key],step)
    for i,a in enumerate(row['per_class_accuracy']): writer.add_scalar(f'{prefix}/class_{i}_accuracy',a,step)

def make_optimizer(model,recipe):
    decay=[]; no=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        (no if p.ndim<2 or any(k in name for k in ('pos_embed','position','cls_token','norm','bias')) else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':recipe['weight_decay']},{'params':no,'weight_decay':0.}],lr=recipe['lr'],betas=tuple(recipe['betas']))

def learning_rate(recipe,step,steps_per_epoch):
    warm=recipe['warmup_epochs']*steps_per_epoch; total=recipe['epochs']*steps_per_epoch
    if step<warm: return recipe['lr']*(step+1)/max(1,warm)
    return recipe['min_lr']+(recipe['lr']-recipe['min_lr'])*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))

def train(config,device_name='cuda',resume=None):
    validate_config_artifacts(config)
    seed_all(config['seed']); device=torch.device(device_name)
    if device.type=='cuda' and config['precision']=='bf16' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 not supported; do not silently change precision')
    rd=run_dir(config); rd.mkdir(parents=True,exist_ok=True)
    if (rd/'completed.json').exists():
        done=json.loads((rd/'completed.json').read_text())
        if file_hash(rd/'final.pt')!=done['checkpoint_sha256']: raise ValueError('Completed checkpoint corrupt')
        return done
    with directory_lock(rd/'writer.lock'):
        return _train_locked(config,rd,device,resume)

def _train_locked(config,rd,device,resume):
    stats=load_stats(config); model=Classifier(config,stats)
    initialize_matched(model.backbone,config['seed']); initial_hash=parameter_hash(model)
    model.to(device); recipe=config['recipe']; batches=batch_settings(config); optimizer=make_optimizer(model,recipe)
    train_ds=load_clean(config['dataset'],'train',protocol=config['protocol'])
    if config.get('subset_ids') is not None: train_ds=Subset(train_ds,config['subset_ids'])
    # Tiny explicitly requested checks are separately identified, never main results.
    if config['dataset']=='cifar10' and not config.get('disable_augmentation',False): train_ds=StatelessAugment(train_ds,config['seed'])
    train_loader,sampler=loader(train_ds,config['seed'],batch_size=batches['train_batch_size'],num_workers=batches['train_num_workers'])
    val_ds=load_clean(config['dataset'],'val',protocol=config['protocol']) if config['protocol']!='full_train_refit' else None
    val_loader=loader(val_ds,batch_size=batches['eval_batch_size'],num_workers=batches['eval_num_workers'])[0] if val_ds is not None else None
    step=0; start_epoch=0; best=-1.; checkpoint=Path(resume) if resume else rd/'latest.pt'
    if resume and not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['config_hash']!=config['config_hash']: raise ValueError('Explicit checkpoint config mismatch')
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        for optstate in optimizer.state.values():
            for k,v in optstate.items():
                if torch.is_tensor(v): optstate[k]=v.to(device)
        step=state['step']; start_epoch=state['next_epoch']; best=state['best_val_accuracy']; restore_rng(state['rng'])
        # Preserve orphaned raw records, then retain only checkpoint-committed epochs/steps.
        for filename,key,limit in [('epochs.jsonl','epoch',start_epoch),('steps.jsonl','optimizer_step',step)]:
            fp=rd/filename
            if fp.exists():
                rows=read_recoverable_jsonl(fp)
                keep=[r for r in rows if r[key]<limit] if key=='epoch' else [r for r in rows if r[key]<=limit]
                if len(keep)!=len(rows):
                    fp.rename(rd/(filename+f'.orphaned.{time.time_ns()}'))
                    fp.write_text(''.join(json.dumps(r)+'\n' for r in keep))
    atomic_json(rd/'config.json',config)
    atomic_json(rd/'initialization.json',{'sha256':initial_hash,'method':'per-parameter name keyed truncated normal; LN scale one; biases zero','seed':config['seed']})
    atomic_json(rd/f'attempt-{time.time_ns()}.json',dict(hardware=hardware(),start_epoch=start_epoch,step=step,config_hash=config['config_hash'],source_hash=config.get('source_hash'),environment_hash=config.get('environment_hash')))
    writer=SummaryWriter(str(tb_dir(config,'train')),purge_step=step+1 if start_epoch else None,flush_secs=30)
    epoch_writer=SummaryWriter(str(tb_dir(config,'train')/'epochs'),purge_step=start_epoch if start_epoch else None,flush_secs=30)
    writer.add_text('provenance/config',json.dumps(config,indent=2),step)
    writer.add_text('provenance/hardware',json.dumps(hardware()),step)
    writer.add_scalar('model/parameters',sum(p.numel() for p in model.parameters()),step)
    if stats:
        for i,(mu,sd,eff) in enumerate(zip(stats['mean'],stats['sigma'],stats['effective_std'])):
            for key,v in [('mean',mu),('measured_std',sd),('effective_std',eff)]: writer.add_scalar(f'normalization/channel_{i}/{key}',v,step)
    started=time.perf_counter()
    try:
        from .visualize import log_training_panel, confusion_figure
        initial_logging_start=time.perf_counter()
        log_training_panel(model,config,writer,step,device,epoch=start_epoch)
        writer.add_scalar('setup/initial_panel_logging_seconds',time.perf_counter()-initial_logging_start,step)
        for epoch in range(start_epoch,recipe['epochs']):
            if stop_requested(): raise InterruptedError('Project STOP marker before epoch')
            epoch_start=time.perf_counter(); sampler.set_epoch(epoch)
            if hasattr(train_ds,'set_epoch'): train_ds.set_epoch(epoch)
            model.train(); model.collect_diagnostics=True; counts=ClassificationCounts(); diagnostics=TrainingDiagnostics()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            for batch_index,(raw,y,identities) in enumerate(train_loader):
                raw=raw.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                lr=learning_rate(recipe,step,len(train_loader))
                for group in optimizer.param_groups: group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                with autocast(config,device): logits=model(raw); loss=torch.nn.functional.cross_entropy(logits.float(),y)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss epoch={epoch} batch={batch_index}')
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['grad_clip'],error_if_nonfinite=True)
                optimizer.step(); step+=1
                labels_predictions=torch.stack((y.detach(),logits.detach().argmax(1)),dim=1).cpu().numpy()
                yy,pred=labels_predictions[:,0],labels_predictions[:,1]
                counts.add(yy,pred,float(loss.detach())*len(yy))
                diagnostics.update({**model.latest_diagnostics,**model.backbone.latest_telemetry})
                if batch_index==0 or batch_index==len(train_loader)-1 or step%recipe['log_step_interval']==0:
                    row=dict(epoch=epoch,optimizer_step=step,samples_seen=epoch*len(train_ds)+counts.total,loss=float(loss.detach()),accuracy=float(np.mean(yy==pred)),gradient_norm=float(grad),lr=lr,elapsed_seconds=time.perf_counter()-started)
                    append_jsonl(rd/'steps.jsonl',row)
                    for key in ('loss','accuracy','gradient_norm','elapsed_seconds','samples_seen'): writer.add_scalar('train_step/'+key,row[key],step)
                    writer.add_scalar('train_step/total_loss',row['loss'],step)
                    for j,g in enumerate(optimizer.param_groups): writer.add_scalar(f'train_step/lr_group{j}',g['lr'],step)
            if counts.total!=len(train_ds): raise ValueError('Training epoch omitted or duplicated samples')
            support_metrics=diagnostics.means()
            train_time=time.perf_counter()-epoch_start; train_metrics=counts.result()
            validation=evaluate(model,val_loader,config,device) if val_loader is not None else None
            row=dict(epoch=epoch,optimizer_step=step,train_online=train_metrics,validation=validation,train_seconds=train_time,epoch_seconds=time.perf_counter()-epoch_start,samples_per_second=len(train_ds)/train_time,lr=lr,support=support_metrics)
            if device.type=='cuda': row['memory']={k:getattr(torch.cuda,k)(device) for k in ['memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved']}
            logging_start=time.perf_counter()
            append_jsonl(rd/'epochs.jsonl',row); log_counts(epoch_writer,'train_online_epoch',train_metrics,epoch)
            if validation is not None:
                log_counts(epoch_writer,'heldout_val_epoch',validation,epoch)
                epoch_writer.add_scalar('heldout_val_epoch/duration_seconds',validation['duration_seconds'],epoch)
                fig=confusion_figure(validation['confusion'],'Held-out validation')
                epoch_writer.add_figure('heldout_val_epoch/confusion',fig,epoch,close=True)
            for k in ('train_seconds','epoch_seconds','samples_per_second','lr'): epoch_writer.add_scalar('epoch/'+k,row[k],epoch)
            for k,v in row['support'].items(): epoch_writer.add_scalar('support_epoch/'+k,v,epoch)
            for k,v in row.get('memory',{}).items(): epoch_writer.add_scalar('memory/'+k,v,epoch)
            if (epoch+1)%recipe['histogram_epoch_interval']==0 or epoch+1==recipe['epochs']:
                for name,p in model.named_parameters():
                    epoch_writer.add_histogram('parameters/'+name,p.detach().float().cpu(),epoch)
                    if p.grad is not None: epoch_writer.add_histogram('gradients/'+name,p.grad.detach().float().cpu(),epoch)
                log_training_panel(model,config,writer,step,device,epoch=epoch+1)
            logging_seconds=time.perf_counter()-logging_start
            epoch_writer.add_scalar('logging/epoch_reporting_seconds',logging_seconds,epoch)
            append_jsonl(rd/'logging.jsonl',{'epoch':epoch,'seconds':logging_seconds})
            improved=validation is not None and validation['accuracy']>best
            if improved: best=validation['accuracy']
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=dict(step=step,recipe=recipe),scaler=None,rng=rng_state(),next_epoch=epoch+1,step=step,sampler=dict(next_epoch=epoch+1,position=0),config_hash=config['config_hash'],normalization_hash=config.get('normalization_hash'),best_val_accuracy=best,initialization_sha256=initial_hash)
            atomic_torch(rd/'latest.pt',state)
            if improved: atomic_torch(rd/'best_validation_diagnostic.pt',state)
            writer.flush(); epoch_writer.flush()
            atomic_json(rd/'progress.json',dict(state='running',epoch=epoch+1,epochs=recipe['epochs'],step=step,seconds=time.perf_counter()-started,job_id=os.getenv('SLURM_JOB_ID'),last_epoch=row,updated=time.time()))
            print(json.dumps({'epoch':epoch+1,'epochs':recipe['epochs'],'train_accuracy':train_metrics['accuracy'],'val_accuracy':validation['accuracy'] if validation else None,'epoch_seconds':row['epoch_seconds']}),flush=True)
        atomic_torch(rd/'final.pt',state)
        done=dict(state='completed',epochs=recipe['epochs'],checkpoint_sha256=file_hash(rd/'final.pt'),config_hash=config['config_hash'],total_seconds=time.perf_counter()-started,training_count=len(train_ds),validation_count=len(val_ds) if val_ds is not None else 0,primary_checkpoint='final.pt')
        atomic_json(rd/'completed.json',done); writer.add_text('completion',json.dumps(done),step)
        return done
    except BaseException as exc:
        atomic_json(rd/f'failure-{time.time_ns()}.json',dict(exception=repr(exc),traceback=traceback.format_exc(),epoch=locals().get('epoch'),step=step,job_id=os.getenv('SLURM_JOB_ID')))
        raise
    finally: writer.flush(); writer.close(); epoch_writer.flush(); epoch_writer.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--resume'); a=p.parse_args()
    train(json.loads(Path(a.config).read_text()),a.device,a.resume)
if __name__=='__main__': main()
