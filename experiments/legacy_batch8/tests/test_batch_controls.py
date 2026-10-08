"""Batch coverage, worker epoch propagation, and detached telemetry arithmetic."""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from sparse_contrast.common import flatten_metrics
from sparse_contrast.data import StatelessAugment
from sparse_contrast.metrics import ClassificationCounts
from sparse_contrast.train import TrainingDiagnostics, batch_settings, learning_rate, loader


class IndexedImages(Dataset):
    def __init__(self, count=130):
        self.split='train'; self.name='cifar10'
        self.indices=np.arange(count); self.labels=np.arange(count)%10; self.targets=self.labels
        self.sample_ids=[f'fixture/{index}' for index in self.indices]; self.split_hash='fixture-only'
        self.base=torch.arange(3*32*32,dtype=torch.int64).reshape(3,32,32).remainder(251).to(torch.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        return self.base.clone(),int(self.labels[index]),self.sample_ids[index]


def test_batch_defaults_and_independent_training_evaluation_controls():
    assert batch_settings({'recipe':{}})==dict(train_batch_size=8,eval_batch_size=8,train_num_workers=0,eval_num_workers=0)
    config={'recipe':dict(batch_size=64,effective_batch_size=64,gradient_accumulation=1,
                          eval_batch_size=256,num_workers=2,eval_num_workers=0)}
    assert batch_settings(config)==dict(train_batch_size=64,eval_batch_size=256,train_num_workers=2,eval_num_workers=0)


@pytest.mark.parametrize('recipe',[
    {'batch_size':0},{'batch_size':64,'effective_batch_size':8},
    {'batch_size':64,'gradient_accumulation':2},{'batch_size':True},
    {'batch_size':64.0},{'eval_batch_size':0},{'num_workers':-1},{'eval_num_workers':-1},
])
def test_invalid_or_accumulated_recipe_fails_explicitly(recipe):
    with pytest.raises(ValueError): batch_settings({'recipe':recipe})


def test_large_batches_preserve_final_tail_coverage_and_weighting():
    data,_=loader(IndexedImages(),batch_size=64)
    counts=ClassificationCounts(); sizes=[]; samples_seen=[]; identities=[]
    for _,labels,ids in data:
        values=labels.numpy(); size=len(values); sizes.append(size); identities.extend(ids)
        batch_mean_loss=3. if size==2 else 1.
        counts.add(values,values,batch_mean_loss*size)
        samples_seen.append(2*len(data.dataset)+counts.total)
    assert sizes==[64,64,2] and len(data)==3
    assert len(identities)==len(set(identities))==130
    assert samples_seen==[324,388,390]
    assert counts.result()['total']==130 and counts.result()['loss']==pytest.approx(134/130)
    recipe=dict(lr=3e-4,min_lr=1e-6,warmup_epochs=1,epochs=3)
    assert learning_rate(recipe,0,len(data))==pytest.approx(1e-4)
    assert learning_rate(recipe,3,len(data))==pytest.approx(3e-4)


def epoch_images(dataset,data,sampler,epoch):
    dataset.set_epoch(epoch); sampler.set_epoch(epoch)
    return {identity:image.clone() for images,_,ids in data for identity,image in zip(ids,images)}


def test_nonpersistent_workers_receive_new_augmentation_epoch_and_same_samples():
    zero=StatelessAugment(IndexedImages(17),seed=7)
    two=StatelessAugment(IndexedImages(17),seed=7)
    baseline,base_sampler=loader(zero,shuffle_seed=11,batch_size=8,num_workers=0)
    workers,worker_sampler=loader(two,shuffle_seed=11,batch_size=8,num_workers=2)
    assert not workers.persistent_workers
    first=None
    for epoch in (0,1):
        expected=epoch_images(zero,baseline,base_sampler,epoch)
        actual=epoch_images(two,workers,worker_sampler,epoch)
        assert list(actual)==list(expected)
        assert all(torch.equal(actual[key],expected[key]) for key in expected)
        if first is None: first=actual
        else: assert any(not torch.equal(actual[key],first[key]) for key in actual)


def test_batched_telemetry_matches_original_float32_sums_with_unequal_tails():
    diagnostic=TrainingDiagnostics(); reference={}
    for length,scale in ((64,1.),(64,1e6),(2,.01)):
        values=torch.arange(length,dtype=torch.float32,requires_grad=True)*scale
        metrics={'support':values,'nested':{'constant':3,'ignored_boolean':False},
                 'channels':torch.tensor([.2,.3,.4],dtype=torch.float64,requires_grad=True)}
        diagnostic.update(metrics)
        for name,value in flatten_metrics(metrics):
            array=value.detach().float(); total,count=reference.get(name,(0.,0))
            reference[name]=(total+float(array.sum()),count+array.numel())
    assert diagnostic.means()=={name:total/count for name,(total,count) in reference.items()}
    assert all(group['sums'].dtype==torch.float64 and not group['sums'].requires_grad
               and group['sums'].grad_fn is None for group in diagnostic.groups.values())
