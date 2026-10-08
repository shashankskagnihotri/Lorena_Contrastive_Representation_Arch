"""Raw-input online classifier; no cached sample-specific frontend work."""
from contextlib import nullcontext
import torch
from torch import nn
from .models import build_backbone
from .frontend import Frontend, sparsify, patch_support, patchify

RAW_STATS={'mnist':((.1307,),(.3081,)),'cifar10':((.4914,.4822,.4465),(.2023,.1994,.2010))}
RAW_PROVENANCE={'mnist':'https://github.com/pytorch/examples/blob/main/mnist/main.py','cifar10':'https://github.com/kuangliu/pytorch-cifar/blob/master/main.py'}

class Classifier(nn.Module):
    def __init__(self, config, stats=None):
        super().__init__(); self.config=config; self.profiling=False; self.collect_diagnostics=False; self.latest_diagnostics={}
        self.frontend=None if config['representation']=='raw' else Frontend(config['representation'])
        if self.frontend is None:
            mean,std=RAW_STATS[config['dataset']]
        else:
            if stats is None: raise ValueError('Verified frozen 0%-training statistics required')
            if config.get('frontend')!=self.frontend.config or stats.get('frontend_config')!=self.frontend.config: raise ValueError('Frozen frontend identity mismatch')
            mean=stats['mean']; std=stats['effective_std']
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32).view(1,-1,1,1))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32).view(1,-1,1,1))
        self.backbone=build_backbone(config['architecture'],28 if config['dataset']=='mnist' else 32,len(mean),drop_path=config['recipe']['stochastic_depth'],execution=config['execution'])
    def scope(self,name): return torch.profiler.record_function(name) if self.profiling else nullcontext()
    def set_profiling(self,enabled):
        self.profiling=enabled; self.backbone.set_profiling(enabled)
        if self.frontend is not None: self.frontend.profiling=enabled
    def prepare(self,raw,execution=None):
        execution=execution or self.config['execution']
        with torch.autocast(device_type=raw.device.type,enabled=False):
            with self.scope('input_conversion'):
                x=raw.float()/255. if raw.dtype==torch.uint8 else raw.float()
            if self.frontend is not None:
                with self.scope('frontend_color_blur_subtraction'): c0=self.frontend(x)
                with self.scope('native_sparsification'): cs=sparsify(c0,self.config['sparsity_percent'])
            else: c0=cs=x
            if execution=='dense': support=None
            else:
                with self.scope('native_support'): support=patch_support(cs,2,0.)
            with self.scope('frozen_normalization_and_patch_layout'):
                # Dense normalize then gather by ORIGINAL support is the specified equivalent reference.
                patches=patchify((cs-self.mean)/self.std,2)
            if self.collect_diagnostics:
                support_diag=patch_support(cs,2,0.)
                self.latest_diagnostics=dict(natural_zero_fraction=(c0==0).float().flatten(1).mean(1).detach(),achieved_zero_fraction=(cs==0).float().flatten(1).mean(1).detach(),spatial_zero_fraction=(cs==0).all(1).float().flatten(1).mean(1).detach(),retained_patches=support_diag.sum(1).detach(),empty_image=(~support_diag.any(1)).float().detach(),native_mean=c0.mean((0,2,3)).detach(),sparse_native_mean=cs.mean((0,2,3)).detach(),normalized_mean=((cs-self.mean)/self.std).mean((0,2,3)).detach())
        return patches,support
    def forward(self,raw,execution=None):
        patches,support=self.prepare(raw,execution)
        return self.backbone.forward_patches(patches,support,execution=execution)
