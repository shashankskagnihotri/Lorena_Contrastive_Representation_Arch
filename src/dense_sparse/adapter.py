"""Inference-only sparsification with an unchanged original checkpoint layout."""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from types import MappingProxyType
from collections.abc import Mapping

import torch

from sparse_contrast.pipeline import Classifier
from sparse_contrast.frontend import sparsify, patch_support, patchify


class InferenceClassifier(Classifier):
    """Keep training identity immutable while selecting explicit test-time behavior.

    Checkpoint validation/loading belongs to the caller and must use the original
    training config and ``load_state_dict(..., strict=True)``. This subclass adds
    no parameters, buffers, or child modules. ``config`` remains the original
    training configuration; callers must not mistake it for the inference policy.
    """

    def __init__(self, original_training_config, stats, inference_spec):
        if not isinstance(inference_spec, Mapping):
            raise TypeError('inference_spec must be a mapping')
        required = {'inference_execution', 'inference_sparsity_percent', 'sparsification_domain'}
        if not required <= inference_spec.keys():
            raise ValueError('Inference policy requires explicit execution, sparsity and domain')
        policy = deepcopy(dict(inference_spec))
        execution = policy['inference_execution']
        percent = policy['inference_sparsity_percent']
        domain = policy['sparsification_domain']
        if execution not in ('dense', 'compact', 'dense_masked'):
            raise ValueError('Unknown inference execution')
        if isinstance(percent, bool) or not isinstance(percent, Real) or not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError('Inference sparsity must be a finite percentage in [0,100]')
        if domain not in ('native_contrast', 'raw_pixels', 'none'):
            raise ValueError('Unknown sparsification domain')
        is_raw = original_training_config['representation'] == 'raw'
        if (domain == 'native_contrast' and is_raw) or (domain == 'raw_pixels' and not is_raw):
            raise ValueError('Sparsification domain disagrees with the trained representation')
        if domain == 'none' and percent != 0:
            raise ValueError('An unsparsified reference requires inference sparsity 0')
        if not is_raw and original_training_config['sparsity_percent'] != 0:
            raise ValueError('This experiment requires zero-imposed-sparsity training checkpoints')
        super().__init__(deepcopy(original_training_config), stats)
        # Ordinary Python metadata does not change state_dict or checkpoint keys.
        self.inference_spec = MappingProxyType(policy)

    @property
    def inference_execution(self):
        return self.inference_spec['inference_execution']

    @property
    def inference_sparsity_percent(self):
        return self.inference_spec['inference_sparsity_percent']

    @property
    def sparsification_domain(self):
        return self.inference_spec['sparsification_domain']

    def _execution(self, execution):
        execution = self.inference_execution if execution is None else execution
        if execution not in ('dense', 'compact', 'dense_masked'):
            raise ValueError('Unknown inference execution override')
        return execution

    def prepare(self, raw, execution=None):
        execution = self._execution(execution)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            with self.scope('input_conversion'):
                x = raw.float() / 255. if raw.dtype == torch.uint8 else raw.float()
            if self.frontend is not None:
                with self.scope('frontend_color_blur_subtraction'):
                    c0 = self.frontend(x)
            else:
                c0 = x
            if self.sparsification_domain == 'none':
                cs = c0
            else:
                with self.scope('native_sparsification'):
                    cs = sparsify(c0, self.inference_sparsity_percent)
            if execution == 'dense':
                support = None
            else:
                with self.scope('native_support'):
                    support = patch_support(cs, 2, 0.)
            with self.scope('frozen_normalization_and_patch_layout'):
                patches = patchify((cs - self.mean) / self.std, 2)
            if self.collect_diagnostics:
                support_diag = patch_support(cs, 2, 0.)
                self.latest_diagnostics = dict(
                    natural_zero_fraction=(c0 == 0).float().flatten(1).mean(1).detach(),
                    achieved_zero_fraction=(cs == 0).float().flatten(1).mean(1).detach(),
                    spatial_zero_fraction=(cs == 0).all(1).float().flatten(1).mean(1).detach(),
                    retained_patches=support_diag.sum(1).detach(),
                    empty_image=(~support_diag.any(1)).float().detach(),
                    native_mean=c0.mean((0, 2, 3)).detach(),
                    sparse_native_mean=cs.mean((0, 2, 3)).detach(),
                    normalized_mean=((cs-self.mean)/self.std).mean((0, 2, 3)).detach())
        return patches, support

    def forward(self, raw, execution=None):
        # Resolve explicitly: the original backbone's default still describes
        # TRAINING execution, which can differ from this inference experiment.
        execution = self._execution(execution)
        patches, support = self.prepare(raw, execution)
        return self.backbone.forward_patches(patches, support, execution=execution)
