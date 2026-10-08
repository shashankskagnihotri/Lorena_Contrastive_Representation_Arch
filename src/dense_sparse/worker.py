"""Resumable official evaluation and isolated timing of fixed trained weights.

A new-study STOP is graceful at official-cell boundaries. The current cell may
finish; the controller's immediate cancellation command stops its allocations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import BASE_SOURCE_HASH
from .common import (ROOT, BASE_ROOT, atomic_json, case_dir, digest, effective_config,
                     file_hash, load_case, load_model, now, read_json,
                     stop_requested, verify_source)
from .scheduler import recover_lock
from sparse_contrast import benchmark
from sparse_contrast.common import directory_lock, hardware, seed_all
from sparse_contrast.data import (CorruptionDataset, corruption_cells,
                                  fixed_panel_indices, ids_hash, load_clean)
from sparse_contrast.metrics import aggregate_cells
from sparse_contrast.train import autocast, batch_settings, evaluate, loader, log_counts


def expected_cells(dataset):
    return [('clean', 'test'), *corruption_cells(dataset)]


def _dataset(case, corruption, severity):
    return (load_clean(case['dataset'], 'test') if corruption == 'clean' else
            CorruptionDataset(case['dataset'], corruption, severity))


def _identity(case, original):
    keys = ('test_config_hash', 'training_config_hash', 'checkpoint_path',
            'checkpoint_sha256', 'completed_receipt_sha256', 'training_source_hash',
            'normalization_hash', 'source_hash', 'dataset', 'architecture',
            'representation', 'seed', 'trained_execution', 'inference_execution',
            'inference_sparsity_percent', 'sparsification_domain')
    result = {key: case[key] for key in keys}
    result.update(study_id='training_dense_testing_sparse', base_source_hash=BASE_SOURCE_HASH,
                  training_config_path=case['training_config_path'],
                  trained_sparsity_percent=original['sparsity_percent'],
                  data_manifest_sha256=original['data_manifest_sha256'],
                  environment_hash=original['environment_hash'], protocol=original['protocol'],
                  split_seed=original['split_seed'], precision=original['precision'])
    if verify_source() != case['source_hash']:
        raise ValueError('Worker source differs from inference case')
    return result


def _check_stop():
    if stop_requested():
        raise InterruptedError('Study or parent STOP marker before next official cell')


def _save_summary(path, summary):
    """Keep an already completed, fully revalidated receipt byte-for-byte."""
    if path.exists():
        previous = read_json(path)
        for key in ('completed_at', 'metadata'):
            if key in previous and key in summary:
                summary[key] = previous[key]
        if digest(previous) != digest(summary):
            raise ValueError('Completed worker summary differs from revalidated cell evidence')
        return previous
    atomic_json(path, summary)
    return summary


def _atomic_npz(path, **arrays):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_predictions(path, dataset, row):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'sample_id', 'label', 'prediction', 'correct'}:
            raise ValueError('Prediction archive fields differ from official evaluation protocol')
        ids = saved['sample_id'].tolist()
        labels, predictions, correct = saved['label'], saved['prediction'], saved['correct']
        if ids != dataset.sample_ids or len(ids) != 10000:
            raise ValueError('Predictions do not cover the exact ordered official sample IDs')
        if not np.array_equal(labels, np.asarray(dataset.labels)):
            raise ValueError('Prediction labels differ from official labels')
        if predictions.shape != labels.shape or np.any((predictions < 0) | (predictions > 9)):
            raise ValueError('Invalid saved predictions')
        if not np.array_equal(correct, labels == predictions):
            raise ValueError('Saved correctness differs from predictions')
        confusion = np.bincount(labels.astype(np.int64) * 10 + predictions,
                                minlength=100).reshape(10, 10)
        class_accuracy = np.divide(np.diag(confusion), confusion.sum(1),
                                   out=np.zeros(10, dtype=float), where=confusion.sum(1) > 0)
        if (int(correct.sum()) != row['correct'] or row['total'] != 10000 or
                not np.array_equal(confusion, row['confusion']) or
                not np.allclose(class_accuracy, row['per_class_accuracy'], rtol=0, atol=1e-12) or
                not math.isclose(row['accuracy'], row['correct'] / 10000, abs_tol=1e-12) or
                not math.isclose(row['error'], 1 - row['accuracy'], abs_tol=1e-12) or
                not math.isfinite(row['loss'])):
            raise ValueError('Saved classification metrics do not reconcile with predictions')


def _verify_token_ids(path, expected_ids):
    ids = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record['metrics'], dict):
                raise ValueError('Missing per-batch token telemetry')
            ids.extend(record['sample_ids'])
    if ids != expected_ids:
        raise ValueError('Token records do not cover the exact ordered official sample IDs')


def _evaluation_cell(path, dataset, identity, corruption, severity):
    row = read_json(path)
    predictions, tokens = path.with_suffix('.npz'), path.with_suffix('.tokens.jsonl')
    if (row.get('state') != 'complete' or row['identity'] != identity or
            row['corruption'] != corruption or row['severity'] != severity or
            row['dataset_split_hash'] != dataset.split_hash or
            row['sample_ids_sha256'] != ids_hash(dataset.sample_ids) or
            row['predictions_sha256'] != file_hash(predictions) or
            row['tokens_sha256'] != file_hash(tokens)):
        raise ValueError('Saved evaluation cell identity, sample coverage or files changed')
    _verify_predictions(predictions, dataset, row)
    _verify_token_ids(tokens, dataset.sample_ids)
    return row


def _evaluation_reference(path, row):
    return dict(corruption=row['corruption'], severity=row['severity'], path=str(path),
                sha256=file_hash(path), total=row['total'],
                predictions_path=str(path.with_suffix('.npz')),
                predictions_sha256=row['predictions_sha256'],
                tokens_path=str(path.with_suffix('.tokens.jsonl')),
                tokens_sha256=row['tokens_sha256'])


def _preview_scales(model, original):
    mean = model.mean.detach().cpu().numpy().reshape(-1)
    std = model.std.detach().cpu().numpy().reshape(-1)
    if model.frontend is None:
        # Raw shared previews do not exist. These are fixed analytic bounds,
        # independent of every clean/OOD observation and inference threshold.
        native = np.ones_like(mean)
        normalized = np.maximum(np.abs(mean / std), np.abs((1 - mean) / std))
        names = ['intensity'] if len(mean) == 1 else ['red', 'green', 'blue']
        provenance = dict(origin='fixed_raw_domain_[0,1]_and_frozen_raw_normalization',
                          mean=mean.tolist(), std=std.tolist())
    else:
        key = {name: original[name] for name in
               ('dataset', 'representation', 'protocol', 'normalization_hash')}
        path = (BASE_ROOT / 'outputs/previews' / original['protocol'] / original['dataset'] /
                original['representation'] / digest(key) / 'complete.json')
        saved = read_json(path)  # Deliberately never call ensure_shared_previews.
        if (saved['dataset'] != original['dataset'] or saved['representation'] != original['representation'] or
                saved['normalization_hash'] != original['normalization_hash'] or
                not np.array_equal(np.asarray(saved['mean'], dtype=np.float32), mean) or
                not np.array_equal(np.asarray(saved['std'], dtype=np.float32), std) or
                saved['channel_names'] != list(model.frontend.channel_names)):
            raise ValueError('Original frozen preview scales do not match checkpoint normalization/frontend')
        native, normalized = np.asarray(saved['native_scales']), np.asarray(saved['normalized_scales'])
        names = saved['channel_names']
        provenance = dict(origin='original_fixed_clean_training_panel', path=str(path), sha256=file_hash(path),
                          train_panel_split_hash=saved['train_panel_split_hash'])
    if (native.shape != mean.shape or normalized.shape != mean.shape or
            not np.isfinite(native).all() or not np.isfinite(normalized).all() or
            np.any(native <= 0) or np.any(normalized <= 0)):
        raise ValueError('Invalid fixed display scales')
    provenance.update(native_scales=native.tolist(), normalized_scales=normalized.tolist(), channel_names=names)
    return (native, normalized), names, provenance


def _raw_sparse_display(arrays):
    # Native raw values are original uint8/255 or exactly zero. Preserve display
    # uint8 RGB units rather than passing [0,255] floats to matplotlib RGB imshow.
    return np.where(arrays['coefficient_support'], arrays['raw'], 0).astype(np.uint8)


def _log_fixed_panel(model, case, original, identity, dataset, corruption, severity, writer, step, device):
    from sparse_contrast.visualize import (panel_tensors, preserve_diagnostic_state, preview_arrays,
                                          raw_prediction_figure, signed_panel_figure)
    raw, labels, ids = panel_tensors(dataset)
    scales, names, scale_provenance = _preview_scales(model, original)
    panel_identity = dict(identity=identity, corruption=corruption, severity=severity,
                          dataset_split_hash=dataset.split_hash, sample_ids=ids.tolist(),
                          display_scales=scale_provenance)
    output = case_dir(case) / 'diagnostics' / 'cells'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{corruption}__{severity}.json'
    artifact = path.with_suffix('.npz')
    if path.exists():
        previous = read_json(path)
        if previous['panel_identity'] != panel_identity or previous['tensor_sha256'] != file_hash(artifact):
            raise ValueError('Saved diagnostic panel identity or exact tensor artifact changed')
        return dict(path=str(path), sha256=file_hash(path), tensor_path=str(artifact),
                    tensor_sha256=previous['tensor_sha256'])
    with preserve_diagnostic_state(model), torch.inference_mode():
        with autocast(original, device):
            logits = model(raw.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite fixed official-panel logits')
        predictions = logits.argmax(1).cpu().numpy()
        frontend = model.frontend if model.frontend is not None else lambda tensor: tensor
        arrays = preview_arrays(raw.to(device), frontend, model.mean, model.std,
                                case['inference_sparsity_percent'])
        execution_support = (np.ones_like(arrays['patch_support'])
                             if case['inference_execution'] == 'dense' else arrays['patch_support'].copy())
        _atomic_npz(artifact, **arrays, labels=labels, sample_ids=ids, predictions=predictions,
                    logits=logits.float().cpu().numpy(), native_scales=scales[0], normalized_scales=scales[1],
                    mean=model.mean.detach().cpu().numpy(), std=model.std.detach().cpu().numpy(),
                    execution_patch_support=execution_support)
        tag = f'fixed_panels/{corruption}/severity_{severity}'
        image_tags = []
        if model.frontend is None:
            displayed = _raw_sparse_display(arrays)
            for first in (0, 5):
                image_tag = f'{tag}/raw_pixels_predictions/group{first // 5}'
                figure = raw_prediction_figure(raw[first:first + 5], displayed[first:first + 5],
                                               labels[first:first + 5], predictions[first:first + 5], ids[first:first + 5])
                writer.add_figure(image_tag, figure, step, close=True)
                image_tags.append(image_tag)
        if case['inference_execution'] != 'dense':
            for channel, name in enumerate(names):
                for first in (0, 5):
                    image_tag = f'{tag}/{name}/group{first // 5}'
                    figure = signed_panel_figure(arrays, channel, scales, labels, ids,
                                                 case['inference_sparsity_percent'], name, first)
                    writer.add_figure(image_tag, figure, step, close=True)
                    image_tags.append(image_tag)
        writer.add_text(f'{tag}/sample_ids', '\n'.join(ids.tolist()), step)
        writer.add_text(f'{tag}/display_scale_provenance', json.dumps(scale_provenance, sort_keys=True), step)
        writer.add_histogram(f'{tag}/logits', logits.float().cpu(), step)
        writer.flush()  # Commit TensorBoard evidence before declaring this panel complete.
    atomic_json(path, dict(state='complete', panel_identity=panel_identity, tensor_path=str(artifact),
                           tensor_sha256=file_hash(artifact), tensor_support_policy='patch_support=native; execution_patch_support=actual_execution',
                           image_tags=image_tags, completed_at=now()))
    return dict(path=str(path), sha256=file_hash(path), tensor_path=str(artifact), tensor_sha256=file_hash(artifact))


def evaluate_case(case_or_path, device='cuda'):
    case = load_case(case_or_path)
    _check_stop()
    seed_all(case['seed'])
    device = torch.device(device)
    model = load_model(case, device)  # Original config and checkpoint are strictly verified here.
    original = model.config
    control = effective_config(case, original)
    batches = batch_settings(original)
    identity = _identity(case, original)
    identity.update(evaluation_batch_size=batches['eval_batch_size'],
                    evaluation_num_workers=batches['eval_num_workers'])
    out = case_dir(case) / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    recover_lock(out / 'writer.lock')
    with directory_lock(out / 'writer.lock'):
        writer = SummaryWriter(str(ROOT / 'outputs/tensorboard' / case['test_config_hash'] / 'evaluation'),
                               flush_secs=30)
        rows, references, diagnostic_panels = [], [], []
        try:
            writer.add_text('identity', json.dumps(identity, sort_keys=True), 0)
            for index, (corruption, severity) in enumerate(expected_cells(case['dataset'])):
                _check_stop()
                dataset = _dataset(case, corruption, severity)
                if len(dataset) != 10000 or len(set(dataset.sample_ids)) != 10000:
                    raise ValueError('Official evaluation cell must contain exactly 10,000 unique samples')
                path = out / f'{corruption}__{severity}.json'
                if path.exists():
                    row = _evaluation_cell(path, dataset, identity, corruption, severity)
                else:
                    data = loader(dataset, batch_size=batches['eval_batch_size'],
                                  num_workers=batches['eval_num_workers'])[0]
                    row = evaluate(model, data, control, device, collect=True)
                    ids, labels, predictions = row.pop('predictions')
                    tokens = row.pop('token_records')
                    if ids.tolist() != dataset.sample_ids or len(ids) != 10000:
                        raise ValueError('Evaluator omitted, duplicated or reordered official samples')
                    if not np.array_equal(labels, np.asarray(dataset.labels)):
                        raise ValueError('Evaluator labels differ from official cell labels')
                    benchmark._write_jsonl(path.with_suffix('.tokens.jsonl'), tokens)
                    del tokens
                    _atomic_npz(path.with_suffix('.npz'), sample_id=ids, label=labels,
                                prediction=predictions, correct=labels == predictions)
                    row.update(state='complete', identity=identity, corruption=corruption,
                               severity=severity, dataset_split_hash=dataset.split_hash,
                               sample_ids_sha256=ids_hash(ids.tolist()),
                               predictions_sha256=file_hash(path.with_suffix('.npz')),
                               tokens_sha256=file_hash(path.with_suffix('.tokens.jsonl')),
                               hardware=hardware(), completed_at=now())
                    atomic_json(path, row)
                    row = _evaluation_cell(path, dataset, identity, corruption, severity)
                rows.append(row)
                references.append(_evaluation_reference(path, row))
                diagnostic_panels.append(_log_fixed_panel(model, case, original, identity, dataset,
                                                          corruption, severity, writer, index, device))
                prefix = f'{corruption}/severity_{severity}'
                log_counts(writer, prefix, row, index)
                for name, metrics in row['support'].items():
                    for statistic, value in metrics.items():
                        writer.add_scalar(f'{prefix}/support/{name}/{statistic}', value, index)
                writer.add_scalar('evaluation_axis/cell_index', index, index)
                if corruption == 'clean':
                    from sparse_contrast.visualize import confusion_figure
                    writer.add_figure('clean/confusion', confusion_figure(row['confusion'], 'Clean test'), index, close=True)
                writer.flush()
                print(json.dumps(dict(mode='evaluation', test_config_hash=case['test_config_hash'],
                                      corruption=corruption, severity=severity,
                                      accuracy=row['accuracy'], loss=row['loss'], total=row['total'])), flush=True)
                del dataset
            summary = aggregate_cells(rows[1:], corruption_cells(case['dataset']))
            losses = {corruption: float(np.mean([row['loss'] for row in rows[1:]
                                                if row['corruption'] == corruption]))
                      for corruption in sorted({row['corruption'] for row in rows[1:]})}
            severity_errors = {severity: float(np.mean([row['error'] for row in rows[1:]
                                                        if str(row['severity']) == severity]))
                               for severity in sorted({str(row['severity']) for row in rows[1:]})}
            aggregate_confusion = np.sum([row['confusion'] for row in rows[1:]], axis=0)
            summary.update(state='complete', identity=identity, case=case, clean=rows[0],
                           per_corruption_loss=losses, mean_corruption_loss=float(np.mean(list(losses.values()))),
                           expected_corruption_cells=summary['expected_cells'],
                           completed_corruption_cells=summary['completed_cells'],
                           completed_cells=len(rows), total_predictions=sum(row['total'] for row in rows),
                           expected_cells=expected_cells(case['dataset']), cells=references,
                           diagnostic_panels=diagnostic_panels, severity_macro_error=severity_errors,
                           corruption_confusion=aggregate_confusion.tolist(),
                           completed_at=now())
            for corruption, error in summary['per_corruption_error'].items():
                writer.add_scalar(f'corruption_mean/{corruption}/error', error, 0)
                writer.add_scalar(f'corruption_mean/{corruption}/loss', losses[corruption], 0)
            writer.add_scalar('corruption_mean/mCE_raw', summary['mCE_raw'], 0)
            writer.add_scalar('corruption_mean/loss', summary['mean_corruption_loss'], 0)
            for severity, error in severity_errors.items():
                writer.add_scalar(f'corruption_mean/severity_{severity}', error, 0)
            from sparse_contrast.visualize import confusion_figure
            writer.add_figure('corruption_mean/confusion',
                              confusion_figure(aggregate_confusion, 'Aggregate corruptions'), 0, close=True)
            summary = _save_summary(out / 'summary.json', summary)
        finally:
            writer.flush()
            writer.close()
    return summary


def _freeze_hardware_contract(metadata, source_hash):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Primary timing requires an allocated scheduler GPU job')
    if (metadata['gpu'] != 'NVIDIA RTX A6000' or
            metadata['cpu_model'] != 'AMD EPYC 7413 24-Core Processor'):
        raise RuntimeError('This study requires the homogeneous A6000/EPYC 7413 timing class')
    fields = ('hardware_identity', *benchmark.HARDWARE_CLASS_FIELDS, *benchmark.CPU_CLASS_FIELDS)
    contract = {key: metadata[key] for key in fields}
    if any(not isinstance(contract[key], int) or contract[key] <= 0
           for key in benchmark.CPU_CLASS_FIELDS[1:]):
        raise ValueError('Timing thread counts must be positive integers')
    path = ROOT / 'outputs/benchmarks/hardware_contract.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(schema_version=1, study_id='training_dense_testing_sparse', source_hash=source_hash,
                 contract=contract, first_device_uuid=metadata['device_uuid'],
                 first_host=metadata['host'], first_scheduler_job=os.environ['SLURM_JOB_ID'],
                 policy='same_GPU_CPU_threads_backend; retain physical devices and sessions; no cross-class pooling')
    # link() publishes one complete immutable file, avoiding a first-worker race.
    fd, temporary = tempfile.mkstemp(prefix='.hardware_contract.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            saved = read_json(path)  # Verify the winning worker's complete contract.
        else:
            saved = value
    finally:
        os.unlink(temporary)
    if saved['source_hash'] != source_hash or saved['contract'] != contract:
        raise RuntimeError('Frozen study hardware/backend/source contract differs from this worker')
    return path, saved


def _verify_timing_cell(path, identity, corruption, severity, contract, dataset):
    row = read_json(path)
    if (row.get('state') != 'complete' or row['identity'] != identity or
            row['corruption'] != corruption or row['severity'] != severity or
            row['warmup_batches_per_scope'] != 50 or row['measured_batches_per_scope'] != 200 or
            row['batch_size'] != identity['batch_size'] or
            set(row['scopes']) != set(benchmark.SCOPES) or
            any(value['count'] != 200 or value['batch_size'] != identity['batch_size']
                for value in row['scopes'].values()) or
            set(row['file_hashes']) != {'timings.jsonl', 'tokens.jsonl', 'batch_membership.json', 'profile.json', 'trace.json'}):
        raise ValueError('Timing cell identity or required measurement scope differs')
    for name, expected in row['file_hashes'].items():
        if file_hash(path.parent / name) != expected:
            raise ValueError(f'Timing evidence hash mismatch: {name}')
    warmup, measured = benchmark.timing_batches(fixed_panel_indices(dataset, 'timing'),
                                                identity['batch_size'], 50, 200)
    expected_membership = dict(warmup_indices=warmup, measured_indices=measured,
                               warmup_ids=[[dataset.sample_ids[index] for index in batch] for batch in warmup],
                               measured_ids=[[dataset.sample_ids[index] for index in batch] for batch in measured],
                               seed=benchmark.BENCHMARK_SEED, dataset_split_hash=dataset.split_hash,
                               independent_of_predictions_sparsity_latency=True)
    if read_json(path.parent / 'batch_membership.json') != expected_membership:
        raise ValueError('Timing sample panel differs from the frozen, intervention-independent panel')
    metadata = benchmark.validate_measurement_session(row['measurement'])
    if any(metadata.get(key) != value for key, value in contract.items()):
        raise ValueError('Resumed timing cell belongs to a different hardware class')
    for when in ('isolation_before', 'isolation_after'):
        isolation = row[when]
        if (not isolation['verified'] or isolation['device_uuid'] != metadata['device_uuid'] or
                set(isolation['observed_compute_pids']) - {isolation['self_pid']}):
            raise ValueError('Saved timing cell lacks verified isolated GPU execution')
    return row


def _timing_stage_metadata(row, case):
    row['sparsification_domain'] = case['sparsification_domain']
    if case['sparsification_domain'] == 'raw_pixels':
        row['stage_applicability']['coefficient_ranking'] = (
            'NA_0percent_bypass_no_sort' if case['inference_sparsity_percent'] == 0 else
            'stable_raw_pixel_abs_sort_and_zero_before_normalization')
    return row


def benchmark_case(case_or_path, execution, batch_size, shard=0, num_shards=1, device='cuda'):
    case = load_case(case_or_path)
    allowed = ('compact', 'dense_masked') if case['inference_execution'] == 'compact' else ('dense',)
    if execution not in allowed or batch_size not in (1, 8):
        raise ValueError('Timing execution/batch differs from the registered inference intervention')
    if (isinstance(shard, bool) or isinstance(num_shards, bool) or
            not isinstance(shard, int) or not isinstance(num_shards, int) or
            num_shards < 1 or not 0 <= shard < num_shards):
        raise ValueError('Invalid timing shard coordinates')
    all_cells = expected_cells(case['dataset'])
    cells = [cell for index, cell in enumerate(all_cells) if index % num_shards == shard]
    if not cells:
        raise ValueError('Timing shard contains no official cells')
    _check_stop()
    device = torch.device(device)
    if device.type != 'cuda' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Primary timing requires an allocated CUDA GPU')
    seed_all(case['seed'])
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Timing workers require one scheduler-visible GPU')
    model = load_model(case, device)
    original = model.config
    if original['precision'] != 'fp32':
        raise ValueError('Matched timing requires the frozen FP32 protocol')
    control = effective_config(case, original)
    identity = _identity(case, original)
    metadata = benchmark.device_metadata(device)
    benchmark.gpu_isolation(device)
    contract_path, saved_contract = _freeze_hardware_contract(metadata, case['source_hash'])
    identity.update(execution=execution, batch_size=batch_size, shard=shard, num_shards=num_shards,
                    hardware_identity=metadata['hardware_identity'], benchmark_rng=benchmark.BENCHMARK_SEED,
                    latency_protocol_version=2, warmup_batches=50, measured_batches=200)
    out = case_dir(case) / 'latency' / execution / f'batch{batch_size}' / f'shard{shard}'
    out.mkdir(parents=True, exist_ok=True)
    recover_lock(out / 'writer.lock')
    with directory_lock(out / 'writer.lock'):
        measurement = benchmark.save_measurement_session(out, metadata)
        # config_hash in this diagnostic identifies the TEST, never a relabelled checkpoint.
        zero_config = {**control, 'config_hash': case['test_config_hash']}
        zero_path = out / 'zero_input_diagnostic.json'
        empty = benchmark.measure_zero_input(model, zero_config, execution, batch_size,
                                             device, zero_path, measurement)
        if (empty['config_hash'] != case['test_config_hash'] or empty['execution'] != execution or
                empty['batch_size'] != batch_size or empty['normalization_hash'] != case['normalization_hash']):
            raise ValueError('All-empty diagnostic belongs to a different intervention')
        empty_metadata = benchmark.validate_measurement_session(empty['measurement'])
        if any(empty_metadata.get(key) != value for key, value in saved_contract['contract'].items()):
            raise ValueError('All-empty diagnostic hardware class changed')
        references, sessions = [], {empty['measurement']['measurement_session_id']: empty['measurement']}
        for index, (corruption, severity) in enumerate(cells):
            _check_stop()
            dataset = _dataset(case, corruption, severity)
            destination = out / 'cells' / f'{corruption}__{severity}'
            row = benchmark.measure_cell(model, control, execution, dataset, corruption, severity,
                                         batch_size, device, destination, identity, measurement=measurement)
            before = digest(row)
            row = _timing_stage_metadata(row, case)
            if digest(row) != before:
                atomic_json(destination / 'summary.json', row)
            row = _verify_timing_cell(destination / 'summary.json', identity, corruption, severity,
                                      saved_contract['contract'], dataset)
            references.append(dict(corruption=corruption, severity=severity,
                                   path=str(destination / 'summary.json'),
                                   sha256=file_hash(destination / 'summary.json'), measurement=row['measurement']))
            sessions[row['measurement']['measurement_session_id']] = row['measurement']
            # No writer exists during primary timings, sustained throughput or profiler.
            logdir = (ROOT / 'outputs/tensorboard' / case['test_config_hash'] / 'latency' /
                      metadata['hardware_identity'] / execution / f'batch{batch_size}' / f'shard{shard}')
            with SummaryWriter(str(logdir)) as writer:
                writer.add_text('identity', json.dumps(identity, sort_keys=True), 0)
                benchmark._log_cell(writer, row, index)
                if index == 0:
                    for scope, values in empty['scopes'].items():
                        writer.add_scalar(f'synthetic_zero_input/{scope}/median_ms',
                                          values['summary']['median_ms'], 0)
                profile = read_json(destination / 'profile.json')
                for event in profile['events']:
                    if '/' in event['name'] or event['name'] in (
                            'input_conversion', 'native_sparsification', 'native_support',
                            'frozen_normalization_and_patch_layout'):
                        writer.add_scalar(f'{corruption}/severity_{severity}/profile/{event["name"]}/cpu_self_ms',
                                          event['cpu_self_ms'], index)
                        writer.add_scalar(f'{corruption}/severity_{severity}/profile/{event["name"]}/device_inclusive_ms',
                                          event['device_total_ms_inclusive'], index)
            print(json.dumps(dict(mode='latency', test_config_hash=case['test_config_hash'],
                                  corruption=corruption, severity=severity, execution=execution,
                                  batch_size=batch_size, shard=shard, state='complete')), flush=True)
            del dataset
        summary = dict(state='complete', identity=identity, case=case, execution=execution,
                       batch_size=batch_size, shard=shard, num_shards=num_shards,
                       expected_cells=cells, all_expected_cells=all_cells, cells=references,
                       hardware_contract_path=str(contract_path), hardware_contract_sha256=file_hash(contract_path),
                       metadata=metadata, measurement_sessions=list(sessions.values()),
                       physical_device_uuids=sorted({session['measurement_device_uuid'] for session in sessions.values()}),
                       zero_input_diagnostic_sha256=file_hash(zero_path), completed_at=now(),
                       comparison_policy='same_GPU_CPU_threads_backend; physical-device/session transitions retained')
        if (out / 'summary.json').exists():
            previous = read_json(out / 'summary.json')
            if any(previous['metadata'].get(key) != value for key, value in saved_contract['contract'].items()):
                raise ValueError('Completed worker summary hardware class changed')
        summary = _save_summary(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evaluate', 'latency'))
    parser.add_argument('--case', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--execution', choices=('compact', 'dense_masked', 'dense'))
    parser.add_argument('--batch-size', type=int, choices=(1, 8))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'latency' and (args.execution is None or args.batch_size is None):
        parser.error('latency requires --execution and --batch-size')
    case = load_case(args.case)
    try:
        if args.mode == 'evaluate':
            result = evaluate_case(case, args.device)
        else:
            result = benchmark_case(case, args.execution, args.batch_size, args.shard,
                                    args.num_shards, args.device)
        print(json.dumps(dict(state=result['state'], mode=args.mode,
                              test_config_hash=case['test_config_hash'], cells=len(result['cells']))), flush=True)
    except Exception as error:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        path = case_dir(case) / 'failures' / f'{args.mode}.{os.getenv("SLURM_JOB_ID", "local")}.{stamp}.json'
        atomic_json(path, dict(state='failed', mode=args.mode, case=case, execution=args.execution,
                               batch_size=args.batch_size, shard=args.shard, num_shards=args.num_shards,
                               job_id=os.getenv('SLURM_JOB_ID'), time=now(),
                               error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
