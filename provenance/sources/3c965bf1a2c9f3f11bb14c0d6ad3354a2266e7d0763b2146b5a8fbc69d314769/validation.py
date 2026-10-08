"""Audited, untimed numerical validation for unchanged FP32 inference."""
from __future__ import annotations

import ast
import copy
import builtins
import dis
import json
import os
from pathlib import Path
from types import CodeType

import numpy as np
import torch

from dense_sparse.common import atomic_json, digest, file_hash, now, read_json

SCIENTIFIC_SOURCE = '8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb'
BASE_SOURCE = '7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'
POLICY = {
    'version': 'fp32_strict_then_fp64_semantic_reference_v1',
    'strict_fp32': {'rtol': 1e-4, 'atol': 2e-5},
    'fp64_semantic': {'rtol': 1e-10, 'atol': 1e-11},
    'fp32_to_reference': {'rtol': 1e-4, 'atol': 1e-4},
    'scope': 'untimed_validation_only; measured_inference_remains_original_FP32',
}


def source_receipt():
    root = Path(__file__).resolve().parent
    path = root / 'source_manifest.json'
    saved = read_json(path)
    if (saved['scientific_source_hash'] != SCIENTIFIC_SOURCE or
            saved['base_source_hash'] != BASE_SOURCE or saved['policy'] != POLICY or
            set(saved['files']) != {'worker.py', 'validation.py'}):
        raise ValueError('Validation source manifest has an unexpected computation or policy')
    for name, expected in saved['files'].items():
        if file_hash(root / name) != expected:
            raise ValueError('Validation source changed: ' + name)
    return {'validation_source_hash': digest(saved), 'validation_source_path': str(root),
            'validation_manifest_path': str(path), 'validation_manifest_sha256': file_hash(path),
            'scientific_source_hash': SCIENTIFIC_SOURCE, 'base_source_hash': BASE_SOURCE,
            'policy': POLICY}


def _function(path, name):
    tree = ast.parse(Path(path).read_text())
    return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)


def verify_global_bindings(original, amended, allowed_new):
    """Check every global bytecode dependency, recursively including lambdas."""
    original=getattr(original,'__wrapped__',original)
    amended=getattr(amended,'__wrapped__',amended)
    def names(code):
        found={x.argval for x in dis.get_instructions(code) if x.opname in ('LOAD_GLOBAL','LOAD_NAME')}
        for value in code.co_consts:
            if isinstance(value,CodeType):found.update(names(value))
        return found
    baseline,updated=names(original.__code__),names(amended.__code__)
    if updated-baseline != allowed_new:
        raise ValueError('Undeclared copied-function global dependencies')
    for name in updated:
        if name not in amended.__globals__ and not hasattr(builtins,name):
            raise ValueError('Missing copied-function global dependency: '+name)
        if name in baseline:
            expected=original.__globals__.get(name,getattr(builtins,name,None))
            actual=amended.__globals__.get(name,getattr(builtins,name,None))
            if actual is not expected:
                raise ValueError('Copied-function dependency differs from frozen implementation: '+name)


def verify_copied_orchestration(base_benchmark, base_worker, amended_worker):
    """Fail closed unless static copies contain only the declared validation edits."""
    base = _function(base_benchmark, 'measure_cell')
    amended = _function(amended_worker, 'measure_cell')
    receipt_node = ast.parse('if config["execution"] == "compact":\n    result["validation_amendment"] = validation_reference(directory)').body[0]
    receipt_matches = [node for node in amended.body if ast.dump(node) == ast.dump(receipt_node)]
    if len(receipt_matches) != 1 or amended.body[-3] is not receipt_matches[0]:
        raise ValueError('Validation receipt must be bound immediately before atomic cell commit')
    amended.body.remove(receipt_matches[0])
    # Strip the single explicitly named validator call back to the original guard.
    replacements = 0
    for node in ast.walk(amended):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'validate_parity':
            expected = ast.parse('validate_parity(model, gpu[0], compact, dense, directory, identity, measurement, measured_ids[0])').body[0]
            if ast.dump(node) != ast.dump(expected):
                raise ValueError('Amended parity call arguments changed')
            node.value = ast.parse('torch.testing.assert_close(compact, dense, rtol=1e-4, atol=2e-5)').body[0].value
            replacements += 1
    if replacements != 1 or ast.dump(base) != ast.dump(amended):
        raise ValueError('Measured cell implementation differs beyond the untimed parity call')
    base = _function(base_worker, 'benchmark_case')
    amended = _function(amended_worker, 'benchmark_case')
    replacements = 0
    for node in ast.walk(amended):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'measure_cell_with_validation_receipt':
            node.func = ast.Attribute(value=ast.Name(id='benchmark', ctx=ast.Load()), attr='measure_cell', ctx=ast.Load())
            replacements += 1
    if replacements != 1 or ast.dump(base) != ast.dump(amended):
        raise ValueError('Worker orchestration differs beyond the explicit validated-cell entry point')
    return {'measure_cell': 'AST_equal_except_one_untimed_validation_call_and_atomic_receipt_binding',
            'benchmark_case': 'AST_equal_except_one_explicit_cell_function_call',
            'frozen_benchmark_sha256': file_hash(base_benchmark),
            'frozen_worker_sha256': file_hash(base_worker)}


def _comparison(left, right):
    left, right = left.double(), right.double()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return {'all_finite': False}
    delta = (left - right).abs()
    return {'all_finite': True, 'max_absolute_error': float(delta.max()),
            'rms_error': float(delta.square().mean().sqrt()),
            'reference_max_absolute': float(right.abs().max()),
            'same_argmax': bool(torch.equal(left.argmax(-1), right.argmax(-1)))}


def _reference_check(model, raw, compact, dense, receipt, archive):
    patches, support = model.prepare(raw, execution='compact')
    dense_patches, dense_support = model.prepare(raw, execution='dense_masked')
    if patches.dtype != torch.float32 or not torch.equal(patches, dense_patches) or not torch.equal(support, dense_support):
        raise ValueError('Compact/dense preparation differs before numerical validation')
    if model.training or model.backbone.training:
        raise ValueError('Parity requires disabled stochastic training layers')
    if any(p.dtype != torch.float32 for p in model.backbone.parameters()):
        raise ValueError('Production backbone must remain FP32')
    reference = copy.deepcopy(model.backbone).double().eval()
    try:
        fp64_compact = reference.forward_patches(patches.double(), support, execution='compact')
        fp64_dense = reference.forward_patches(patches.double(), support, execution='dense_masked')
        receipt['reference_comparisons'] = {
            'fp64_compact_vs_dense': _comparison(fp64_compact, fp64_dense),
            'fp32_compact_vs_reference': _comparison(compact, fp64_compact),
            'fp32_dense_vs_reference': _comparison(dense, fp64_dense),
        }
        # Save exact evidence before checks, including any rejected reference.
        from dense_sparse.worker import _atomic_npz
        _atomic_npz(archive, raw=raw.detach().cpu().numpy(), patches=patches.detach().cpu().numpy(),
                        support=support.detach().cpu().numpy(), sample_ids=np.asarray(receipt['sample_ids']),
                    fp32_compact=compact.detach().cpu().numpy(), fp32_dense=dense.detach().cpu().numpy(),
                    fp64_compact=fp64_compact.detach().cpu().numpy(), fp64_dense=fp64_dense.detach().cpu().numpy())
        receipt.update(reference_archive=str(archive), reference_archive_sha256=file_hash(archive))
        if not bool(torch.isfinite(fp64_compact).all() and torch.isfinite(fp64_dense).all()):
            raise ValueError('Nonfinite FP64 reference')
        torch.testing.assert_close(fp64_compact, fp64_dense, **POLICY['fp64_semantic'])
        torch.testing.assert_close(compact.double(), fp64_compact, **POLICY['fp32_to_reference'])
        torch.testing.assert_close(dense.double(), fp64_dense, **POLICY['fp32_to_reference'])
    finally:
        del reference
    if any(p.dtype != torch.float32 for p in model.backbone.parameters()):
        raise ValueError('Reference computation changed production parameter dtype')


@torch.inference_mode()
def validate_parity(model, raw, compact, dense, directory, identity, measurement, sample_ids):
    """Retain strict guard; diagnose its failures using independent FP64 semantics.

    The reference is a copied backbone with exact stored FP32 weights and exact
    FP32 prepared patches promoted to FP64. No parameter, buffer, inference dtype,
    sample, support decision, batch shape, or timed operation is modified.
    """
    source = source_receipt()
    if identity['source_hash'] != source['scientific_source_hash']:
        raise ValueError('Validation amendment belongs to a different scientific source')
    job = os.environ.get('SLURM_JOB_ID', 'unit_test')
    path = directory / f'parity_validation.{job}.json'
    archive = path.with_suffix('.npz')
    receipt = dict(state='checking', identity=identity, measurement=measurement,
                   sample_ids=sample_ids, created=now(), job_id=job, source=source,
                   original_comparison=_comparison(compact, dense), fallback_used=False)
    try:
        if not bool(torch.isfinite(compact).all() and torch.isfinite(dense).all()):
            raise ValueError('Nonfinite FP32 outputs cannot pass numerical validation')
        try:
            torch.testing.assert_close(compact, dense, **POLICY['strict_fp32'])
        except AssertionError as error:
            receipt.update(fallback_used=True, original_strict_failure=str(error))
            devices = [raw.device.index if raw.device.index is not None else torch.cuda.current_device()] if raw.is_cuda else []
            before = torch.cuda.memory_allocated(raw.device) if raw.is_cuda else None
            try:
                with torch.random.fork_rng(devices=devices):
                    _reference_check(model, raw, compact, dense, receipt, archive)
            finally:
                # The helper frame is gone: no FP64 backbone, output or patch is
                # live. Match the original scope policy before its 50 warmups.
                import gc
                gc.collect()
                if raw.is_cuda:
                    torch.cuda.synchronize(raw.device)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(raw.device)
                receipt['reference_cleanup'] = dict(
                    helper_frame_released=True, rng_restored=True,
                    synchronize_and_empty_cache_before_primary_warmup=bool(raw.is_cuda),
                    allocated_bytes_before=before,
                    allocated_bytes_after=torch.cuda.memory_allocated(raw.device) if raw.is_cuda else None,
                    reserved_bytes_after=torch.cuda.memory_reserved(raw.device) if raw.is_cuda else None)
            if raw.is_cuda and receipt['reference_cleanup']['allocated_bytes_after'] > before:
                raise ValueError('FP64 reference retained GPU allocations before primary warmup')
            receipt['state'] = 'reference_verified_rounding'
        else:
            receipt['state'] = 'original_strict_pass'
    except Exception as error:
        receipt.update(state='failed', error_type=type(error).__name__, error=str(error))
        atomic_json(path, receipt)
        raise
    atomic_json(path, receipt)


def validation_reference(directory):
    path = directory / f'parity_validation.{os.environ.get("SLURM_JOB_ID", "unit_test")}.json'
    value = read_json(path)
    return {'path': str(path), 'sha256': file_hash(path), 'state': value['state'],
            'validation_source_hash': value['source']['validation_source_hash']}


def verify_validation_reference(ref, identity, measurement):
    path = Path(ref['path'])
    if file_hash(path) != ref['sha256']:
        raise ValueError('Amended validation receipt changed')
    saved = read_json(path)
    if (saved['identity'] != identity or saved['measurement'] != measurement or
            saved['state'] not in ('original_strict_pass', 'reference_verified_rounding') or
            saved['state'] != ref['state'] or saved['source'] != source_receipt() or
            ref['validation_source_hash'] != saved['source']['validation_source_hash']):
        raise ValueError('Amended validation identity/source/state mismatch')
    if saved['fallback_used']:
        if file_hash(saved['reference_archive']) != saved['reference_archive_sha256']:
            raise ValueError('Saved FP64 numerical evidence changed')
    return saved
