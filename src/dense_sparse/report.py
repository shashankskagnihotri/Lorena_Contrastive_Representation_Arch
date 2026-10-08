"""Report the registered inference interventions without changing training evidence.

Large timing files are consumed one cell at a time. Only one measurement worker's
paired repeats and small aggregate rows are retained, never the full campaign's
per-repeat token/profiler records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from sparse_contrast.data import corruption_cells
from .common import ROOT, BASE_ROOT, load_manifest, case_dir, read_json, file_hash, digest, atomic_json, now, verify_source

sns.set_theme(context='talk', style='whitegrid', font='serif', font_scale=1.0)
SCOPES = ('gpu_raw_to_logits', 'host_raw_to_logits', 'loader_to_cpu_prediction',
          'cached_input_model_only_diagnostic')
IDENTITY = ('test_config_hash', 'training_config_hash', 'checkpoint_sha256',
            'normalization_hash', 'source_hash', 'trained_execution',
            'inference_execution', 'inference_sparsity_percent', 'sparsification_domain')
GROUP = ('dataset', 'architecture', 'representation', 'trained_execution',
         'inference_execution', 'sparsification_domain', 'inference_sparsity_percent')
METRICS = ('clean_accuracy', 'clean_loss', 'mCE_raw', 'mean_corruption_loss',
           'brightness_accuracy', 'contrast_accuracy', 'fog_accuracy')
COLORS = dict(zip(('raw', 'single_color', 'grayscale', 'color_opponency'), sns.color_palette('colorblind', 4)))


def pairs(dataset):
    return [('clean', 'test')] + [(c, str(s)) for c, s in corruption_cells(dataset)]


def key(row, fields=GROUP):
    return tuple(row[k] for k in fields)


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite evidence cannot enter aggregates')
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def metadata(case):
    return {k: case[k] for k in (*GROUP, 'seed', *IDENTITY) if k in case} | {
        'protocol': 'heldout_val', 'training_count': 55000 if case['dataset'] == 'mnist' else 45000,
        'validation_count': 5000, 'test_count_per_cell': 10000}


def validate_identity(value, case):
    identity = value.get('identity', value)
    for field in IDENTITY:
        if identity.get(field) != case[field]:
            raise ValueError(f'Inference identity differs: {field}')


def resolve(path, parent):
    path = Path(path)
    return path if path.is_absolute() else parent / path


def save_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ['state']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


class CellTable:
    """Incremental CSV output for evidence that can have hundreds of thousands of rows."""
    def __init__(self, path, fields):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('w', newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.handle.close()


def reference(case, cache):
    """An original dense-0 result remains distinct from a new compact-0 anchor."""
    h = case['training_config_hash']
    if h in cache:
        return cache[h]
    rd = Path(case['checkpoint_path']).parent
    config = read_json(case['training_config_path'])
    done = read_json(rd / 'completed.json')
    if digest({k: v for k, v in config.items() if k != 'config_hash'}) != h:
        raise ValueError('Original training configuration changed')
    if done['config_hash'] != h or done['checkpoint_sha256'] != case['checkpoint_sha256']:
        raise ValueError('Original training completion identity changed')
    if file_hash(rd / 'completed.json') != case['completed_receipt_sha256']:
        raise ValueError('Original training completion receipt changed')
    if done['state'] != 'completed' or done['training_count'] != metadata(case)['training_count'] or done['validation_count'] != 5000:
        raise ValueError('Original training completion/count invalid')
    result = {k: case[k] for k in ('dataset', 'architecture', 'representation', 'seed', 'trained_execution')}
    result.update(protocol='heldout_val', training_count=done['training_count'], validation_count=done['validation_count'])
    result.update({'original_run_directory': str(rd), 'original_training_config_hash': h,
              'original_training_execution': config['execution'],
              'original_training_sparsity_percent': config['sparsity_percent'],
              'original_epochs': done['epochs'], 'training_batch_size': config['recipe']['batch_size'],
              'training_lr': config['recipe']['lr']})
    last = None
    with (rd / 'epochs.jsonl').open() as handle:
        for line in handle:
            if line.strip(): last = json.loads(line)
    if last is None or int(last['epoch']) + 1 != done['epochs']:
        raise ValueError('Original final epoch evidence is missing')
    for split, field in [('train_online', 'train_online'), ('validation', 'validation')]:
        for metric in ('accuracy', 'loss'):
            result[f'original_{split}_{metric}'] = last[field][metric]
    original_evaluation = Path(case['original_evaluation_path'])
    if original_evaluation != rd / 'evaluation/summary.json' or file_hash(original_evaluation) != case['original_evaluation_sha256']:
        raise ValueError('Frozen original evaluation reference changed')
    summary = read_json(original_evaluation)
    if summary['state'] != 'complete' or summary['config']['config_hash'] != h or summary['checkpoint_sha256'] != case['checkpoint_sha256']:
        raise ValueError('Original evaluation reference identity changed')
    expected = len(pairs(case['dataset'])) - 1
    if summary['completed_cells'] != expected or summary['expected_cells'] != expected:
        raise ValueError('Original corruption reference is incomplete')
    result.update(original_clean_accuracy=summary['clean']['accuracy'], original_clean_loss=summary['clean']['loss'],
                  original_mCE_raw=summary['mCE_raw'])
    cache[h] = result
    return result


def validate_cell(cell, path, case, deep=True):
    validate_identity(cell, case)
    matrix = np.asarray(cell['confusion'])
    if cell['total'] != 10000 or matrix.shape != (10, 10) or matrix.sum() != 10000 or matrix.trace() != cell['correct']:
        raise ValueError(f'Cell counts/confusion differ: {path}')
    if not math.isclose(cell['accuracy'], cell['correct'] / 10000, abs_tol=1e-12) or not math.isclose(cell['error'], 1 - cell['accuracy'], abs_tol=1e-12):
        raise ValueError(f'Cell accuracy/error differs: {path}')
    if not all(math.isfinite(float(cell[k])) for k in ('accuracy', 'error', 'loss')):
        raise ValueError(f'Nonfinite cell metric: {path}')
    pred = path.with_suffix('.npz'); tokens = path.with_suffix('.tokens.jsonl')
    if file_hash(pred) != cell['predictions_sha256'] or file_hash(tokens) != cell['tokens_sha256']:
        raise ValueError(f'Cell evidence hash differs: {path}')
    if deep:
        with np.load(pred, allow_pickle=False) as value:
            if len(value['sample_id']) != 10000 or len(set(value['sample_id'])) != 10000:
                raise ValueError(f'Prediction identities incomplete: {pred}')
            ids_hash = hashlib.sha256(''.join(f'{x}\n' for x in value['sample_id']).encode()).hexdigest()
            if ids_hash != cell['sample_ids_sha256']:
                raise ValueError(f'Prediction sample order/hash differs: {pred}')
            if int((value['label'] == value['prediction']).sum()) != cell['correct']:
                raise ValueError(f'Predictions disagree with accuracy: {pred}')


def collect_accuracy(case, references, cell_table):
    row = metadata(case) | {'state': 'missing', 'completed_cells': 0, 'expected_cells': len(pairs(case['dataset']))}
    row.update(reference(case, references))
    directory = case_dir(case) / 'evaluation'; available = {}
    token_values = defaultdict(dict)
    for corruption, severity in pairs(case['dataset']):
        path = directory / f'{corruption}__{severity}.json'
        if not path.exists():
            continue
        cell = read_json(path); validate_cell(cell, path, case)
        if cell['corruption'] != corruption or str(cell['severity']) != severity:
            raise ValueError(f'Wrong official cell: {path}')
        available[(corruption, severity)] = cell
        cell_table.write(metadata(case) | {'corruption': corruption, 'severity': severity,
            **{k: cell[k] for k in ('accuracy', 'error', 'loss', 'correct', 'total')}, 'path': str(path)})
        for metric, distribution in cell.get('support', {}).items():
            if isinstance(distribution, dict) and 'mean' in distribution:
                token_values[metric][(corruption, severity)] = float(distribution['mean'])
    row['completed_cells'] = len(available)
    if available: row['state'] = 'partial'
    if ('clean', 'test') in available:
        row.update(clean_accuracy=available[('clean', 'test')]['accuracy'], clean_loss=available[('clean', 'test')]['loss'])
    corruptions = defaultdict(list)
    for (c, s), value in available.items():
        if c != 'clean': corruptions[c].append(value)
    required = Counter(c for c, s in pairs(case['dataset']) if c != 'clean')
    for c in ('brightness', 'contrast', 'fog'):
        row[f'{c}_available_in_benchmark'] = c in required
        if c in required and len(corruptions[c]) == required[c]:
            row[f'{c}_accuracy'] = float(np.mean([v['accuracy'] for v in corruptions[c]]))
    if all(len(corruptions[c]) == n for c, n in required.items()):
        row['mCE_raw'] = float(np.mean([np.mean([v['error'] for v in corruptions[c]]) for c in required]))
        row['mean_corruption_loss'] = float(np.mean([np.mean([v['loss'] for v in corruptions[c]]) for c in required]))
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path); validate_identity(summary, case)
        if summary['state'] != 'complete' or len(available) != row['expected_cells']:
            raise ValueError('Evaluation completion receipt has missing cells')
        expected = set(pairs(case['dataset']))
        seen = set()
        for record in summary['cells']:
            k = (record['corruption'], str(record['severity']))
            if k in seen or k not in expected: raise ValueError('Repeated/unexpected evaluation receipt cell')
            seen.add(k); p = resolve(record['path'], directory)
            if file_hash(p) != record['sha256']: raise ValueError('Evaluation receipt cell hash differs')
        if seen != expected: raise ValueError('Evaluation receipt coverage differs from manifest')
        if {(c, str(s)) for c, s in summary['expected_cells']} != expected or summary['completed_cells'] != len(expected):
            raise ValueError('Evaluation summary counts differ from manifest')
        if not math.isclose(summary['mCE_raw'], row['mCE_raw'], abs_tol=1e-12): raise ValueError('Corruption macro mean differs')
        panels = summary['diagnostic_panels']; panel_cells = set()
        for record in panels:
            p = Path(record['path']); tensor = Path(record['tensor_path'])
            if file_hash(p) != record['sha256'] or file_hash(tensor) != record['tensor_sha256']:
                raise ValueError('Diagnostic panel evidence hash differs')
            panel = read_json(p); validate_identity(panel['panel_identity'], case)
            cell_key = (panel['panel_identity']['corruption'], str(panel['panel_identity']['severity']))
            if cell_key in panel_cells or panel['state'] != 'complete': raise ValueError('Invalid/duplicated diagnostic panel')
            panel_cells.add(cell_key)
        if panel_cells != expected: raise ValueError('Fixed diagnostic panel coverage is incomplete')
        row['completed_diagnostic_panels'] = len(panels)
        row['state'] = 'complete'
    for metric in ('clean_accuracy', 'clean_loss', 'mCE_raw'):
        if metric in row: row[f'delta_{metric}_from_original'] = row[metric] - row[f'original_{metric}']
    corruption_rows = []
    for c, count in required.items():
        if len(corruptions[c]) == count:
            corruption_rows.append(metadata(case) | {'corruption': c, 'severity': 'mean',
                'accuracy': float(np.mean([v['accuracy'] for v in corruptions[c]])),
                'error': float(np.mean([v['error'] for v in corruptions[c]])),
                'loss': float(np.mean([v['loss'] for v in corruptions[c]]))})
    for s in sorted({s for c, s in pairs(case['dataset']) if c != 'clean'}):
        selected = [v for (c, severity), v in available.items() if c != 'clean' and severity == s]
        if len(selected) == 15:
            corruption_rows.append(metadata(case) | {'corruption': 'macro_all_primary', 'severity': s,
                'accuracy': float(np.mean([v['accuracy'] for v in selected])),
                'error': float(np.mean([v['error'] for v in selected])),
                'loss': float(np.mean([v['loss'] for v in selected]))})
    token_rows = []
    for metric, values in token_values.items():
        if ('clean', 'test') in values:
            token_rows.append(metadata(case) | {'input_group': 'clean', 'metric': metric, 'value': values[('clean', 'test')]})
        needed = set(pairs(case['dataset'])) - {('clean', 'test')}
        if needed <= values.keys():
            token_rows.append(metadata(case) | {'input_group': 'balanced_corruption', 'metric': metric,
                'value': float(np.mean([values[k] for k in sorted(needed)]))})
    return row, token_rows, corruption_rows


def aggregate_accuracy(rows, cases):
    groups = defaultdict(list); expected = defaultdict(set)
    for case in cases: expected[key(case)].add(case['seed'])
    for row in rows: groups[key(row)].append(row)
    result = []
    metrics = (*METRICS, 'original_train_online_accuracy', 'original_train_online_loss',
               'original_validation_accuracy', 'original_validation_loss', 'original_clean_accuracy',
               'original_clean_loss', 'original_mCE_raw', 'delta_clean_accuracy_from_original',
               'delta_clean_loss_from_original', 'delta_mCE_raw_from_original',
               'delta_clean_accuracy_from_compact0', 'delta_clean_loss_from_compact0', 'delta_mCE_raw_from_compact0')
    for group, seeds in expected.items():
        item = dict(zip(GROUP, group)); item.update(protocol='heldout_val', training_count=55000 if item['dataset'] == 'mnist' else 45000,
            expected_seeds=len(seeds), completed_seeds=sum(r['state'] == 'complete' for r in groups[group]))
        for metric in metrics:
            values = [float(r[metric]) for r in groups[group] if metric in r and r[metric] is not None]
            item[metric + '_mean'], item[metric + '_sd'] = mean_sd(values)
            item[metric + '_n'] = len(values)
            if metric.startswith('delta_'):
                item[metric + '_all_seeds_positive'] = len(values) == len(seeds) and all(v > 0 for v in values)
                item[metric + '_all_seeds_negative'] = len(values) == len(seeds) and all(v < 0 for v in values)
        result.append(item)
    return result


def timing_values(path, expected_hash, case, execution, batch, measurement=None):
    """Validate and retain only 800 scalar durations for the current cell."""
    values = {s: {} for s in SCOPES}; sample_ids = {}; h = hashlib.sha256()
    with path.open('rb') as handle:
        for line in handle:
            h.update(line)
            if not line.strip(): continue
            row = json.loads(line); validate_identity(row, case)
            if measurement:
                for field in ('hardware_identity', 'measurement_session_id', 'measurement_device_uuid'):
                    if row.get(field) != measurement[field]: raise ValueError('Timing record hardware/session differs')
            scope, repeat = row['scope'], int(row['repeat_id'])
            if scope not in values or repeat in values[scope] or row['execution'] != execution or row['batch_size'] != batch:
                raise ValueError(f'Unexpected or repeated timing record: {path}')
            ms = float(row['elapsed_ms'])
            if not math.isfinite(ms) or ms <= 0 or len(row['sample_ids']) != batch:
                raise ValueError(f'Invalid timing value/panel: {path}')
            ids = tuple(row['sample_ids'])
            if repeat in sample_ids and sample_ids[repeat] != ids:
                raise ValueError('Timing scopes do not use identical paired images')
            sample_ids[repeat] = ids; values[scope][repeat] = ms
    if h.hexdigest() != expected_hash:
        raise ValueError(f'Raw timing hash differs: {path}')
    for scope in SCOPES:
        if set(values[scope]) != set(range(200)):
            raise ValueError(f'Timing must contain all 200 repeats: {path}/{scope}')
    return {s: [values[s][i] for i in range(200)] for s in SCOPES}, digest([sample_ids[i] for i in range(200)])


def hardware_label(measurement, cache):
    path = Path(measurement['hardware_metadata_file']); h = measurement['hardware_metadata_sha256']
    if h not in cache:
        if file_hash(path) != h: raise ValueError('Hardware session file hash differs')
        value = read_json(path)
        if value['hardware_identity'] != measurement['hardware_identity']:
            raise ValueError('Hardware class differs from measurement receipt')
        cache[h] = value
    for saved_key, measured_key in [('hardware_identity', 'hardware_identity'), ('measurement_session_id', 'measurement_session_id'),
                                    ('device_uuid', 'measurement_device_uuid')]:
        if cache[h].get(saved_key) != measurement[measured_key]:
            raise ValueError('Hardware metadata UUID/session differs from measurement')
    return {k: cache[h].get(k) for k in ('gpu', 'cpu_model', 'torch_num_threads', 'torch_num_interop_threads',
                                       'precision', 'torch', 'cuda', 'cudnn', 'driver_version')}


def collect_latency(case, panel_hashes, hardware_cache, cell_table):
    expected = set(pairs(case['dataset'])); workers, timing_rows, stage_rows = [], [], []
    executions = ('compact', 'dense_masked') if case['inference_execution'] == 'compact' else (case['inference_execution'],)
    for execution in executions:
        for batch in (1, 8):
            directory = case_dir(case) / 'latency' / execution / f'batch{batch}'
            receipts = sorted(directory.glob('shard*/summary.json'))
            worker = metadata(case) | {'execution': execution, 'batch_size': batch, 'expected_cells': len(expected),
                                       'completed_cells': 0, 'state': 'missing'}
            cells = {}; hardware_ids = set(); shard_ids = set(); shard_counts = set()
            pooled = defaultdict(list); memories = defaultdict(list); throughput = defaultdict(list)
            stage_values = defaultdict(list); sessions = set(); devices = set(); hardware_by_id = {}
            for path in receipts:
                receipt = read_json(path); validate_identity(receipt, case)
                if receipt['state'] != 'complete' or receipt['execution'] != execution or receipt['batch_size'] != batch:
                    raise ValueError(f'Invalid latency shard completion: {path}')
                if file_hash(receipt['hardware_contract_path']) != receipt['hardware_contract_sha256']:
                    raise ValueError('Hardware contract receipt changed')
                contract = read_json(receipt['hardware_contract_path'])['contract']
                if file_hash(path.parent / 'zero_input_diagnostic.json') != receipt['zero_input_diagnostic_sha256']:
                    raise ValueError('Zero-input diagnostic receipt changed')
                shard = int(receipt['shard']); shards = int(receipt['num_shards'])
                if shard in shard_ids or shard < 0 or shard >= shards: raise ValueError('Repeated/invalid latency shard')
                shard_ids.add(shard); shard_counts.add(shards)
                declared = {(c, str(s)) for c, s in receipt['expected_cells']}; observed = set()
                for record in receipt['cells']:
                    cell_key = (record['corruption'], str(record['severity']))
                    if cell_key not in expected or cell_key in cells: raise ValueError('Repeated/unexpected timing cell')
                    cp = resolve(record['path'], path.parent)
                    if file_hash(cp) != record['sha256']: raise ValueError('Latency cell completion hash differs')
                    summary = read_json(cp); validate_identity(summary, case)
                    if summary['state'] != 'complete' or summary['identity']['execution'] != execution or summary['batch_size'] != batch:
                        raise ValueError('Latency execution/batch differs from receipt')
                    if (summary['corruption'], str(summary['severity'])) != cell_key:
                        raise ValueError('Latency cell differs from receipt')
                    if summary['warmup_batches_per_scope'] != 50 or summary['measured_batches_per_scope'] != 200:
                        raise ValueError('Timing repetition protocol changed')
                    measurement = summary['measurement']; hardware = hardware_label(measurement, hardware_cache)
                    if any(hardware_cache[measurement['hardware_metadata_sha256']].get(k) != v for k, v in contract.items()):
                        raise ValueError('Observed hardware differs from the frozen contract')
                    hid = measurement['hardware_identity']; hardware_ids.add(hid)
                    hardware_by_id[hid] = hardware
                    sessions.add(measurement['measurement_session_id']); devices.add(measurement['measurement_device_uuid'])
                    values, panel_hash = timing_values(cp.parent / 'timings.jsonl', summary['file_hashes']['timings.jsonl'], case, execution, batch, measurement)
                    pk = (case['dataset'], case['architecture'], case['seed'], batch, *cell_key)
                    if pk in panel_hashes and panel_hashes[pk] != panel_hash: raise ValueError('Compared methods use different timing panels')
                    panel_hashes[pk] = panel_hash
                    for name, h in summary['file_hashes'].items():
                        if name != 'timings.jsonl' and file_hash(cp.parent / name) != h:
                            raise ValueError(f'Latency diagnostic artifact hash differs: {cp.parent / name}')
                    input_group = 'clean' if cell_key[0] == 'clean' else 'balanced_corruption'
                    for scope in SCOPES:
                        saved = summary['scopes'][scope]; array = np.asarray(values[scope])
                        calculated = {'mean_ms': float(array.mean()), 'median_ms': float(np.median(array)), 'p95_ms': float(np.percentile(array, 95))}
                        if saved['count'] != 200 or any(not math.isclose(saved[k], v, rel_tol=1e-10, abs_tol=1e-10) for k, v in calculated.items()):
                            raise ValueError('Saved timing statistics disagree with raw repeats')
                        pooled[(hid, input_group, scope)].extend(values[scope])
                        memory = saved['memory']; memories[(hid, input_group, scope)].append(memory)
                        throughput[(hid, input_group, scope)].append(saved['sustained_throughput'])
                        cell_table.write(metadata(case) | {'execution': execution, 'batch_size': batch,
                            'corruption': cell_key[0], 'severity': cell_key[1], 'scope': scope,
                            'hardware_identity': hid, **hardware, **measurement, **calculated,
                            'max_memory_allocated': memory['max_memory_allocated'], 'max_memory_reserved': memory['max_memory_reserved'],
                            'sustained_images_per_second': saved['sustained_throughput']['images_per_second'], 'summary_path': str(cp)})
                    profile = read_json(cp.parent / 'profile.json')
                    # Inclusive attribution is kept separate; these events cannot be stacked into a total.
                    for event in profile.get('events', []):
                        if event['name'].startswith(('vit/', 'swin/')) or event['name'] in ('input_conversion', 'frontend_color_blur_subtraction', 'native_sparsification',
                            'native_support', 'frozen_normalization_and_patch_layout', 'patch_embedding', 'transformer',
                            'patch_merging', 'pooling', 'classifier', 'compaction'):
                            for metric in ('cpu_self_ms', 'cpu_total_ms_inclusive', 'device_self_ms', 'device_total_ms_inclusive'):
                                if metric in event:
                                    stage_values[(hid, input_group, event['name'], metric)].append(float(event[metric]))
                    observed.add(cell_key); cells[cell_key] = hid
                if observed != declared: raise ValueError('Latency shard cell list differs from its declared coverage')
            if len(shard_counts) > 1: raise ValueError('Mixed latency sharding plans')
            worker['completed_cells'] = len(cells)
            worker['hardware_identities'] = ';'.join(sorted(hardware_ids))
            worker['state'] = 'partial' if cells else 'missing'
            if cells.keys() == expected and len(shard_counts) == 1 and shard_ids == set(range(next(iter(shard_counts)))):
                if len(hardware_ids) != 1: raise ValueError('A complete worker mixes hardware classes')
                worker['state'] = 'complete'
            workers.append(worker)
            for (hid, input_group, scope), values in pooled.items():
                need = {('clean', 'test')} if input_group == 'clean' else expected - {('clean', 'test')}
                observed = {k for k, h in cells.items() if h == hid and (k[0] == 'clean') == (input_group == 'clean')}
                if observed != need: continue
                values = np.asarray(values, dtype=np.float64)
                through = throughput[(hid, input_group, scope)]
                times = sum(v['elapsed_seconds'] for v in through); images = sum(v['images'] for v in through)
                mem = memories[(hid, input_group, scope)]
                timing_rows.append(metadata(case) | {'execution': execution, 'batch_size': batch, 'scope': scope,
                    'hardware_identity': hid, **hardware_by_id[hid], 'input_group': input_group,
                    'cell_count': len(need), 'repeat_count': len(values), 'mean_ms': float(values.mean()),
                    'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
                    'amortized_mean_ms_per_image': float(values.mean()) / batch,
                    'serial_images_per_second': 1000 * batch / float(values.mean()),
                    'sustained_images_per_second': images / times,
                    'max_memory_allocated': max(v['max_memory_allocated'] for v in mem),
                    'max_memory_reserved': max(v['max_memory_reserved'] for v in mem),
                    'measurement_sessions': ';'.join(sorted(sessions)), 'physical_device_uuids': ';'.join(sorted(devices))})
            for (hid, input_group, stage, metric), values in stage_values.items():
                need = {('clean', 'test')} if input_group == 'clean' else expected - {('clean', 'test')}
                if {k for k, h in cells.items() if h == hid and (k[0] == 'clean') == (input_group == 'clean')} != need:
                    continue
                stage_rows.append(metadata(case) | {'execution': execution, 'batch_size': batch,
                    'hardware_identity': hid, **hardware_by_id[hid], 'input_group': input_group,
                    'stage': stage, 'metric': metric, 'value_ms_mean': float(np.mean(values)),
                    'observed_profile_events': len(values), 'additive': False,
                    'scope': 'gpu_raw_to_logits_profile_diagnostic'})
    return workers, timing_rows, stage_rows


def add_paired_changes(accuracy, timing):
    anchors = {r['training_config_hash']: r for r in accuracy
               if r['inference_execution'] == 'compact' and r['inference_sparsity_percent'] == 0}
    for row in accuracy:
        anchor = anchors.get(row['training_config_hash'])
        if anchor and row['inference_execution'] == 'compact':
            for metric in ('clean_accuracy', 'clean_loss', 'mCE_raw'):
                if metric in row and metric in anchor:
                    row[f'delta_{metric}_from_compact0'] = row[metric] - anchor[metric]
    fields = ('dataset', 'architecture', 'seed', 'batch_size', 'scope', 'input_group', 'hardware_identity')
    raw = {key(r, fields): r for r in timing if r['representation'] == 'raw'
           and r['inference_execution'] == 'dense' and r['sparsification_domain'] == 'none'}
    masked = {(r['test_config_hash'], *key(r, fields[2:])): r for r in timing if r['execution'] == 'dense_masked'}
    for row in timing:
        baseline = raw.get(key(row, fields))
        if baseline:
            row['paired_raw_dense_speedup'] = baseline['median_ms'] / row['median_ms']
        control = masked.get((row['test_config_hash'], *key(row, fields[2:])))
        if row['execution'] == 'compact' and control:
            row['paired_dense_masked_speedup'] = control['median_ms'] / row['median_ms']


def aggregate_timing(rows, cases, workers):
    additional = ('execution', 'batch_size', 'scope', 'input_group', 'hardware_identity')
    groups = defaultdict(list); expected = defaultdict(set)
    hardware = sorted({r['hardware_identity'] for r in rows}) or ['unmeasured']
    for case in cases:
        executions = ('compact', 'dense_masked') if case['inference_execution'] == 'compact' else (case['inference_execution'],)
        for execution in executions:
            for batch in (1, 8):
                for scope in SCOPES:
                    for input_group in ('clean', 'balanced_corruption'):
                        for hid in hardware:
                            expected[(*key(case), execution, batch, scope, input_group, hid)].add(case['seed'])
    for row in rows: groups[key(row, (*GROUP, *additional))].append(row)
    metrics = ('mean_ms', 'median_ms', 'p95_ms', 'amortized_mean_ms_per_image', 'serial_images_per_second',
               'sustained_images_per_second', 'max_memory_allocated', 'max_memory_reserved',
               'paired_raw_dense_speedup', 'paired_dense_masked_speedup')
    labels = {r['hardware_identity']: {k: r[k] for k in ('gpu', 'cpu_model', 'torch_num_threads', 'torch_num_interop_threads')} for r in rows}
    result = []
    for group, seeds in expected.items():
        row = dict(zip((*GROUP, *additional), group)); row.update(protocol='heldout_val',
            training_count=55000 if row['dataset'] == 'mnist' else 45000, expected_seeds=len(seeds),
            **labels.get(row['hardware_identity'], {}))
        for metric in metrics:
            values = [float(r[metric]) for r in groups[group] if metric in r]
            row[metric + '_mean'], row[metric + '_sd'] = mean_sd(values); row[metric + '_n'] = len(values)
            if 'speedup' in metric:
                row[metric + '_all_seeds_faster'] = len(values) == len(seeds) and all(v > 1 for v in values)
                row[metric + '_all_seeds_slower'] = len(values) == len(seeds) and all(v < 1 for v in values)
        result.append(row)
    return result


def save_figure(fig, stem, rows, writer):
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_csv(stem.with_suffix('.csv'), rows)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + suffix), dpi=300, bbox_inches='tight')
    writer.add_figure(stem.relative_to(ROOT / 'outputs/figures').as_posix(), fig, global_step=0, close=False)
    plt.close(fig)
    return str(stem.with_suffix('.png').relative_to(ROOT))


def curve_plot(rows, metric, ylabel, stem, writer):
    if not any(r.get(metric + '_n') == r['expected_seeds'] and r.get(metric + '_mean') is not None for r in rows):
        return None
    fig, ax = plt.subplots(figsize=(9, 6.6), layout='constrained')
    families = sorted({(r['representation'], r['trained_execution'], r['inference_execution']) for r in rows})
    for rep, trained, inference in families:
        group = sorted((r for r in rows if (r['representation'], r['trained_execution'], r['inference_execution']) == (rep, trained, inference)), key=lambda r: r['inference_sparsity_percent'])
        label = f'{rep.replace("_", " ")} / trained {trained}'
        if inference == 'dense':
            point = group[0]
            if point.get(metric + '_n') == point['expected_seeds']:
                ax.axhline(point[metric + '_mean'], color=COLORS[rep], linestyle=':', label='Unchanged raw dense reference')
            continue
        y = [r[metric + '_mean'] if r.get(metric + '_n') == r['expected_seeds'] else np.nan for r in group]
        sd = [r.get(metric + '_sd') or 0 for r in group]
        ax.errorbar([r['inference_sparsity_percent'] for r in group], y, yerr=sd,
                    marker='o' if trained == 'dense' else 's', linestyle='-' if trained == 'dense' else '--',
                    color=COLORS[rep], label=label, capsize=3)
    ax.set(xlabel='Inference sparsity (%)', ylabel=ylabel, xticks=list(range(0, 100, 10)))
    complete = sum(r.get(metric + '_n') == r['expected_seeds'] for r in rows)
    ax.set_title(f'{complete}/{len(rows)} conditions with all seeds; gaps are incomplete', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, fontsize=10)
    return save_figure(fig, stem, rows, writer)


def render_figures(conditions, timing, tokens):
    figures = []
    labels = {'clean_accuracy': 'Clean accuracy (fraction)', 'clean_loss': 'Clean cross-entropy loss',
              'mCE_raw': 'Mean corruption error (fraction)', 'mean_corruption_loss': 'Mean corruption cross-entropy loss',
              'brightness_accuracy': 'Brightness accuracy (fraction)', 'contrast_accuracy': 'Contrast accuracy (fraction)',
              'fog_accuracy': 'Fog accuracy (fraction)', 'delta_clean_accuracy_from_original': 'Clean change from original execution (fraction)',
              'delta_mCE_raw_from_original': 'Corruption-error change from original execution (fraction)'}
    groups = sorted({(r['dataset'], r['architecture']) for r in conditions})
    for dataset, architecture in groups:
        base = ROOT / 'outputs/figures' / dataset / architecture
        writer_path = ROOT / 'outputs/tensorboard/study_summary' / dataset / architecture
        with SummaryWriter(str(writer_path)) as writer:
            rows = [r for r in conditions if r['dataset'] == dataset and r['architecture'] == architecture]
            for metric, label in labels.items():
                path = curve_plot(rows, metric, label, base / metric, writer)
                if path: figures.append(path)
            selected = [r for r in timing if r['dataset'] == dataset and r['architecture'] == architecture]
            timing_groups = sorted({(r['hardware_identity'], r['batch_size'], r['scope'], r['input_group']) for r in selected})
            for hid, batch, scope, input_group in timing_groups:
                subset = [r for r in selected if (r['hardware_identity'], r['batch_size'], r['scope'], r['input_group']) == (hid, batch, scope, input_group)
                          and r['execution'] != 'dense_masked']
                folder = base / 'latency' / hid / f'batch{batch}' / scope / input_group
                for metric, label in [('median_ms', 'Latency: mean seed median (ms/batch)'), ('p95_ms', 'Latency: mean seed p95 (ms/batch)'),
                                      ('paired_raw_dense_speedup', 'Paired raw dense / method median latency'),
                                      ('paired_dense_masked_speedup', 'Paired dense-masked / compact median latency'),
                                      ('max_memory_reserved', 'Peak reserved device memory (bytes)')]:
                    path = curve_plot(subset, metric, label, folder / metric, writer)
                    if path: figures.append(path)
                metric = 'clean_accuracy' if input_group == 'clean' else 'mCE_raw'
                lookup = {key(r): r for r in rows}
                points = [(r, lookup[key(r)]) for r in subset if key(r) in lookup and r['median_ms_n'] == r['expected_seeds']
                          and lookup[key(r)][metric + '_n'] == lookup[key(r)]['expected_seeds']]
                if points:
                    fig, ax = plt.subplots(figsize=(9, 6.6), layout='constrained')
                    plot_rows = []
                    for timed, accuracy in points:
                        x, y = timed['median_ms_mean'], accuracy[metric + '_mean']
                        ax.errorbar(x, y, xerr=timed['median_ms_sd'] or 0, yerr=accuracy[metric + '_sd'] or 0,
                            color=COLORS[timed['representation']], marker='o' if timed['trained_execution'] == 'dense' else 's', capsize=2)
                        ax.annotate(str(timed['inference_sparsity_percent']), (x, y), fontsize=8, xytext=(3, 3), textcoords='offset points')
                        plot_rows.append({**timed, metric + '_mean': y, metric + '_sd': accuracy[metric + '_sd']})
                    ax.set(xlabel='Mean seed median latency (ms/batch)', ylabel=labels[metric],
                           title=f'{scope}, B{batch}, {input_group}\n{hid[:12]}; labels are inference sparsity (%)')
                    from matplotlib.lines import Line2D
                    families = sorted({(r['representation'], r['trained_execution']) for r, a in points})
                    ax.legend(handles=[Line2D([], [], color=COLORS[rep], marker='o' if trained == 'dense' else 's',
                        linestyle='', label=f'{rep.replace("_", " ")} / trained {trained}') for rep, trained in families],
                        loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, fontsize=10)
                    figures.append(save_figure(fig, folder / ('tradeoff_' + metric), plot_rows, writer))
    return figures


def aggregate_named(rows, cases, extra, value='value'):
    expected = defaultdict(set)
    for case in cases: expected[key(case)].add(case['seed'])
    groups = defaultdict(list)
    for row in rows: groups[key(row, (*GROUP, *extra))].append(row)
    result = []
    for group, observations in groups.items():
        seeds = [r['seed'] for r in observations]
        if len(set(seeds)) != len(seeds): raise ValueError('Duplicate seed in diagnostic aggregate')
        row = dict(zip((*GROUP, *extra), group)); row.update(protocol='heldout_val',
            training_count=55000 if row['dataset'] == 'mnist' else 45000, expected_seeds=len(expected[group[:len(GROUP)]]))
        row['value_mean'], row['value_sd'] = mean_sd([r[value] for r in observations]); row['value_n'] = len(observations)
        result.append(row)
    return result


def condition_order(row):
    return (row['inference_execution'] != 'dense', ('raw', 'single_color', 'grayscale', 'color_opponency').index(row['representation']),
            row['trained_execution'] != 'dense', row['inference_sparsity_percent'])


def condition_name(row):
    if row['inference_execution'] == 'dense': return 'Unchanged raw dense reference'
    return f"{row['representation'].replace('_', ' ')} / trained {row['trained_execution']} / {row['inference_sparsity_percent']}%"


def diagnostic_figures(conditions, timing, timing_seeds, tokens, corruptions, stages):
    figures = []
    for dataset, architecture in sorted({(r['dataset'], r['architecture']) for r in conditions}):
        rows = sorted([r for r in conditions if (r['dataset'], r['architecture']) == (dataset, architecture)], key=condition_order)
        base = ROOT / 'outputs/figures' / dataset / architecture
        with SummaryWriter(str(ROOT / 'outputs/tensorboard/study_summary' / dataset / architecture)) as writer:
            support = [r for r in tokens if (r['dataset'], r['architecture']) == (dataset, architecture)]
            selected_metrics = {'natural_zero_fraction', 'achieved_zero_fraction', 'spatial_zero_fraction', 'retained_patches', 'padding_tokens',
                *[f'stages/{stage}/blocks/0/{metric}' for stage in range(4) for metric in ('active_per_image', 'padding_rows')]}
            for metric, source in sorted({(r['metric'], r['input_group']) for r in support if r['metric'] in selected_metrics}):
                subset = [r for r in support if r['metric'] == metric and r['input_group'] == source]
                path = curve_plot(subset, 'value', metric.replace('/', ' / '), base / 'support' / source / metric.replace('/', '_'), writer)
                if path: figures.append(path)
            corrupt = [r for r in corruptions if (r['dataset'], r['architecture']) == (dataset, architecture)]
            if corrupt:
                names = sorted({r['corruption'] for r in corrupt if r['severity'] == 'mean'})
                lookup = {(key(r), r['corruption']): r for r in corrupt if r['severity'] == 'mean'}
                grid = np.full((len(rows), len(names)), np.nan)
                for i, condition in enumerate(rows):
                    for j, name in enumerate(names):
                        v = lookup.get((key(condition), name))
                        if v and v['value_n'] == v['expected_seeds']: grid[i, j] = 100 * v['value_mean']
                if np.isfinite(grid).any():
                    fig, ax = plt.subplots(figsize=(14, max(6, len(rows) * .3)), layout='constrained')
                    sns.heatmap(grid, mask=~np.isfinite(grid), vmin=0, vmax=100, cmap='magma', ax=ax,
                        yticklabels=[condition_name(r) for r in rows], xticklabels=names, cbar_kws={'label': 'Corruption error (%)'})
                    ax.tick_params(axis='y', labelsize=8); ax.tick_params(axis='x', labelsize=9)
                    figures.append(save_figure(fig, base / 'per_corruption_error', corrupt, writer))
                if dataset == 'cifar10':
                    severity = [r for r in corrupt if r['corruption'] == 'macro_all_primary']
                    fig, axes = plt.subplots(2, 5, figsize=(22, 9), layout='constrained', sharex=True, sharey=True)
                    for pct, ax in zip(range(0, 100, 10), axes.flat):
                        subset = [r for r in severity if r['inference_sparsity_percent'] == pct or r['inference_execution'] == 'dense']
                        for family in sorted({(r['representation'], r['trained_execution'], r['inference_execution']) for r in subset}):
                            rep, trained, execution = family
                            curve = sorted([r for r in subset if (r['representation'], r['trained_execution'], r['inference_execution']) == family], key=lambda r: int(r['severity']))
                            y = [100 * r['value_mean'] if r['value_n'] == r['expected_seeds'] else np.nan for r in curve]
                            ax.errorbar([int(r['severity']) for r in curve], y, yerr=[100 * (r['value_sd'] or 0) for r in curve],
                                color=COLORS[rep], marker='o' if trained == 'dense' else 's',
                                linestyle=':' if execution == 'dense' else '-' if trained == 'dense' else '--',
                                label='Raw dense reference' if execution == 'dense' else f'{rep} / trained {trained}', capsize=2)
                        ax.set(title=f'Inference sparsity {pct}%', xlabel='Severity', xticks=range(1, 6))
                    axes[0, 0].set_ylabel('Macro corruption error (%)'); axes[1, 0].set_ylabel('Macro corruption error (%)')
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=10)
                    figures.append(save_figure(fig, base / 'corruption_severity', severity, writer))
            timed = [r for r in timing if (r['dataset'], r['architecture']) == (dataset, architecture) and r['execution'] != 'dense_masked']
            tgroups = sorted({(r['hardware_identity'], r['batch_size'], r['scope'], r['input_group']) for r in timed})
            for hid, batch, scope, source in tgroups:
                selected = sorted([r for r in timed if (r['hardware_identity'], r['batch_size'], r['scope'], r['input_group']) == (hid, batch, scope, source)], key=condition_order)
                if not any(r['median_ms_n'] for r in selected): continue
                seeds = [r for r in timing_seeds if (r['dataset'], r['architecture'], r['hardware_identity'], r['batch_size'], r['scope'], r['input_group']) == (dataset, architecture, hid, batch, scope, source) and r['execution'] != 'dense_masked']
                fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(selected) * .28)), layout='constrained', sharey=True)
                for ax, metric in zip(axes, ('median_ms', 'p95_ms')):
                    for i, row in enumerate(selected):
                        individual = [v[metric] for v in seeds if key(v) == key(row)]
                        ax.scatter(individual, [i] * len(individual), marker='|', color='gray', s=40)
                        if row[metric + '_n']:
                            ax.errorbar(row[metric + '_mean'], i, xerr=row[metric + '_sd'] or 0,
                                marker='o', color=COLORS[row['representation']], capsize=2)
                        else: ax.text(0, i, 'missing', fontsize=7, va='center')
                    ax.set(xlabel='Median latency (ms/batch)' if metric == 'median_ms' else 'p95 latency (ms/batch)', yticks=range(len(selected)),
                           yticklabels=[f"{condition_name(r)} ({r[metric + '_n']}/{r['expected_seeds']} seeds)" for r in selected]); ax.tick_params(axis='y', labelsize=8)
                    ax.set_xlim(left=0)
                axes[0].set_ylim(len(selected) - .5, -.5)
                fig.suptitle(f'{dataset} / {architecture}; B{batch}; {scope}; {source}\n{selected[0].get("gpu", "")} / {hid[:12]}; seed points and mean +/- sample SD', fontsize=13)
                figures.append(save_figure(fig, base / 'latency' / hid / f'batch{batch}' / scope / source / 'direct_overview', selected, writer))
            observed_stages = [r for r in stages if (r['dataset'], r['architecture']) == (dataset, architecture)]
            stage_groups = sorted({(r['hardware_identity'], r['execution'], r['batch_size'], r['input_group'], r['metric']) for r in observed_stages})
            for hid, execution, batch, source, metric in stage_groups:
                if metric not in ('cpu_total_ms_inclusive', 'device_total_ms_inclusive'): continue
                selected = [r for r in observed_stages if (r['hardware_identity'], r['execution'], r['batch_size'], r['input_group'], r['metric']) == (hid, execution, batch, source, metric)]
                names = sorted({r['stage'] for r in selected if '/attention' not in r['stage'] and 'residual' not in r['stage']})
                order = [r for r in rows if (r['inference_execution'] == 'compact') == (execution in ('compact', 'dense_masked'))]
                lookup = {(key(r), r['stage']): r for r in selected}; grid = np.full((len(order), len(names)), np.nan)
                for i, condition in enumerate(order):
                    for j, name in enumerate(names):
                        value = lookup.get((key(condition), name))
                        if value and value['value_n'] == value['expected_seeds']: grid[i, j] = value['value_mean']
                if np.isfinite(grid).any():
                    fig, ax = plt.subplots(figsize=(max(12, len(names) * .45), max(5, len(order) * .28)), layout='constrained')
                    sns.heatmap(grid, mask=~np.isfinite(grid), cmap='viridis', ax=ax, yticklabels=[condition_name(r) for r in order],
                        xticklabels=names, cbar_kws={'label': 'Mean reported inclusive event interval (ms)'})
                    ax.tick_params(axis='both', labelsize=8)
                    ax.set_title(f'GPU raw-to-logits profile; {execution}; B{batch}; {source}\n{hid[:12]}; inclusive events are nonadditive', fontsize=12)
                    figures.append(save_figure(fig, base / 'stages' / hid / execution / f'batch{batch}' / source / metric, selected, writer))
    return figures


def normalization_rows(cases):
    rows = []
    for dataset, representation in sorted({(c['dataset'], c['representation']) for c in cases}):
        selected = next(c for c in cases if (c['dataset'], c['representation']) == (dataset, representation))
        config = read_json(selected['training_config_path'])
        common = {'dataset': dataset, 'representation': representation, 'protocol': 'heldout_val',
                  'training_count': 55000 if dataset == 'mnist' else 45000, 'normalization_hash': config['normalization_hash']}
        if representation == 'raw':
            rows.append(common | {'kind': 'published_raw_constants', **config['normalization']})
        else:
            path = BASE_ROOT / 'normalization' / f'{dataset}_{representation}_heldout_val.json'; fit = read_json(path)
            if fit['artifact_sha256'] != config['normalization_hash'] or digest({k: v for k, v in fit.items() if k != 'artifact_sha256'}) != fit['artifact_sha256']:
                raise ValueError('Frozen normalization reference changed')
            if file_hash(path.parent / fit['npz_file']) != fit['npz_sha256']: raise ValueError('Frozen normalization NPZ changed')
            rows.append(common | {'kind': 'frozen_0pct_train_population_fit', 'path': str(path), 'json_sha256': file_hash(path),
                **{k: fit[k] for k in ('mean', 'sigma', 'effective_std', 'guarded_channels', 'image_count', 'split_hash', 'source_file_sha256')}})
    return rows


def verify_figure_outputs(figures):
    if not figures: return False
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    expected = defaultdict(set)
    for name in figures:
        path = ROOT / name
        for extension in ('png', 'pdf', 'svg', 'csv'):
            artifact = path.with_suffix('.' + extension)
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise ValueError(f'Report figure artifact missing/empty: {artifact}')
        with path.with_suffix('.csv').open(newline='') as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or next(rows, None) is None: raise ValueError('Figure plot-data CSV is empty')
        relative = path.relative_to(ROOT / 'outputs/figures')
        expected[relative.parts[:2]].add(relative.with_suffix('').as_posix())
    for (dataset, architecture), tags in expected.items():
        observed = set()
        directory = ROOT / 'outputs/tensorboard/study_summary' / dataset / architecture
        for path in directory.glob('events.out.tfevents.*'):
            for event in EventFileLoader(str(path)).Load():
                for value in event.summary.value:
                    if value.HasField('image') or value.metadata.plugin_data.plugin_name == 'images': observed.add(value.tag)
        if not tags <= observed: raise ValueError(f'TensorBoard figure mirrors missing: {sorted(tags - observed)}')
    return True


def coverage_record(manifest, accuracy, workers, errors, figures, figures_complete=False):
    cases = manifest['cases']
    expected_cells = sum(len(pairs(c['dataset'])) for c in cases)
    expected_workers = sum(4 if c['inference_execution'] == 'compact' else 2 for c in cases)
    expected_timing_cells = sum(len(pairs(c['dataset'])) * (4 if c['inference_execution'] == 'compact' else 2) for c in cases)
    complete = sum(r['state'] == 'complete' for r in accuracy)
    complete_workers = sum(w['state'] == 'complete' for w in workers)
    hids = {h for w in workers for h in w.get('hardware_identities', '').split(';') if h}
    expected_ids = {c['test_config_hash']: len(pairs(c['dataset'])) for c in cases}
    actual_ids = [r['test_config_hash'] for r in accuracy]
    expected_worker_ids = {(c['test_config_hash'], execution, batch) for c in cases
        for execution in (('compact', 'dense_masked') if c['inference_execution'] == 'compact' else (c['inference_execution'],)) for batch in (1, 8)}
    actual_worker_ids = [(w['test_config_hash'], w['execution'], w['batch_size']) for w in workers]
    accuracy_ok = (complete == len(cases) and len(set(actual_ids)) == len(actual_ids) and set(actual_ids) == expected_ids.keys()
        and all(r['completed_cells'] == expected_ids[r['test_config_hash']] for r in accuracy)
        and not any(e['phase'] == 'evaluation' for e in errors))
    timing_ok = (complete_workers == expected_workers and len(hids) == 1 and len(set(actual_worker_ids)) == len(actual_worker_ids)
        and set(actual_worker_ids) == expected_worker_ids
        and all(w['completed_cells'] == expected_ids[w['test_config_hash']] for w in workers)
        and not any(e['phase'] == 'latency' for e in errors))
    return {'created_at': now(), 'study_id': manifest['study_id'], 'manifest_hash': manifest['manifest_hash'],
        'source_hash': manifest['source_hash'], 'expected_evaluation_cases': len(cases),
        'completed_evaluation_cases': complete, 'expected_evaluation_cells': expected_cells,
        'completed_evaluation_cells': sum(r['completed_cells'] for r in accuracy),
        'accuracy_complete': accuracy_ok, 'expected_latency_workers': expected_workers,
        'completed_latency_workers': complete_workers, 'expected_latency_cells': expected_timing_cells,
        'completed_latency_cells': sum(w['completed_cells'] for w in workers), 'latency_complete': timing_ok,
        'figures_complete': bool(figures_complete and figures),
        'all_requested_work_complete': bool(accuracy_ok and timing_ok and figures_complete and figures), 'hardware_identities': sorted(hids),
        'evaluation_states': dict(Counter(r['state'] for r in accuracy)),
        'latency_worker_states': dict(Counter(w['state'] for w in workers)), 'errors': errors, 'figures': figures}


def format_metric(row, metric, percent=False):
    count, expected = row[metric + '_n'], row['expected_seeds']
    mean, sd = row[metric + '_mean'], row[metric + '_sd']
    if mean is None: return f'missing (0/{expected})'
    scale = 100 if percent else 1
    value = f'{scale * mean:.3f}' + (f' +/- {scale * sd:.3f}' if sd is not None else '')
    return value + f' ({count}/{expected})'


def write_report(manifest, coverage, conditions, timing, references):
    lines = ['# Training dense, testing sparse', '',
        f"Study `{manifest['study_id']}`; manifest `{manifest['manifest_hash']}`; source `{manifest['source_hash']}`.", '',
        f"Accuracy: {coverage['completed_evaluation_cases']}/{coverage['expected_evaluation_cases']} registered cases and "
        f"{coverage['completed_evaluation_cells']}/{coverage['expected_evaluation_cells']} official cells. "
        f"Timing: {coverage['completed_latency_workers']}/{coverage['expected_latency_workers']} complete execution/batch workers and "
        f"{coverage['completed_latency_cells']}/{coverage['expected_latency_cells']} cells. "
        f"All requested work complete: **{coverage['all_requested_work_complete']}**.", '',
        'Protocol `heldout_val`: MNIST Train N=55,000; CIFAR-10 Train N=45,000; validation N=5,000 each. '
        'Every official clean or corruption cell has 10,000 images. MNIST-C has 15 fixed-severity cells; '
        'CIFAR-10-C has 15 corruptions at five severities. mCE below is the unnormalized macro mean corruption error. '
        'MNIST-C has no contrast corruption, so that field is unavailable rather than zero.', '',
        'No model is retrained. "Dense training" here means zero imposed coefficient sparsity; original dense execution '
        'and original compact execution at 0% remain separate checkpoint families. Dense inference raw references '
        'remain separate from compact raw-pixel 0% anchors, which can already drop naturally empty patches. '
        'Raw-pixel sparsification ranks [0,1] input values before published raw normalization; contrast sparsification '
        'ranks signed native contrast magnitudes before frozen affine normalization. No normalization statistics are refit.', '',
        '[Frozen normalization references and hashes](outputs/tables/normalization.csv) include published raw constants, '
        'contrast population means/sigmas, effective guarded scales, and exact training partitions. '
        'The original training source and this inference source remain separate identities.', '',
        '| Dataset / architecture | Representation / training execution | Inference execution / domain / sparsity | Clean accuracy (%) | Mean corruption error (%) | Clean loss | Brightness accuracy (%) | Contrast accuracy (%) | Fog accuracy (%) |',
        '|---|---|---|---|---|---|---|---|---|']
    for row in conditions:
        contrast = 'not in MNIST-C' if row['dataset'] == 'mnist' else format_metric(row, 'contrast_accuracy', True)
        lines.append(f"| {row['dataset']} / {row['architecture']} | {row['representation']} / {row['trained_execution']} | "
            f"{row['inference_execution']} / {row['sparsification_domain']} / {row['inference_sparsity_percent']}% | "
            f"{format_metric(row, 'clean_accuracy', True)} | {format_metric(row, 'mCE_raw', True)} | "
            f"{format_metric(row, 'clean_loss')} | {format_metric(row, 'brightness_accuracy', True)} | {contrast} | "
            f"{format_metric(row, 'fog_accuracy', True)} |")
    lines += ['', 'Values show mean +/- sample SD across the registered training seeds, with observed/expected seed counts. '
        'Incomplete means are labeled; exported curves require every registered seed at a point. No missing or failed case '
        'is removed from coverage. [All seed results](outputs/tables/accuracy_seeds.csv), '
        '[condition means and paired deltas](outputs/tables/accuracy_conditions.csv), '
        '[official cell results](outputs/tables/evaluation_cells.csv).', '',
        '[Per-corruption and severity tables](outputs/tables/corruption_conditions.csv) preserve all official conditions. '
        'Brightness and fog are released corruption types, not a claim about all real-world lighting changes. '
        'Fixed diagnostic tensors, masks, IDs, logits, and display-scale provenance are stored under each '
        'case\'s `diagnostics/cells/`; evaluation completion requires one validated panel per official cell.', '',
        '**Original training and 0% references.** [Checkpoint reference table](outputs/tables/original_references.csv) '
        'preserves the original final-epoch online training, held-out validation, clean-test, and mCE values. '
        'These are reused evidence from `training_sparse_testing_sparse`, not new training measurements. '
        'Each new seed row records differences from that checkpoint\'s original execution and, when available, '
        'its new compact 0% anchor. A dense-trained checkpoint evaluated compact at 0% is a new intervention. '
        'The original dense 0% result cannot be relabeled as that compact anchor. '
        'All-three-seed sign agreement in the CSV is descriptive, not a significance test.', '',
        '**Measured timing and memory.** [Seed timing table](outputs/tables/timing_seeds.csv), '
        '[condition timing table](outputs/tables/timing_conditions.csv), '
        '[cell timing/memory table](outputs/tables/timing_cells.csv), '
        '[worker coverage](outputs/tables/timing_workers.csv), and '
        '[separate profiler attribution](outputs/tables/stages.csv). '
        'All four scopes remain separate: GPU-resident raw input to logits; pinned-host raw input to logits; '
        'loader input through CPU prediction; and cached-input model-only diagnostics. The cached scope excludes '
        'online preprocessing and cannot establish an online speedup. Batch latency is milliseconds per batch; '
        'amortized batch latency is not a batch-one request. Compact and dense-masked timings use the same checkpoint, '
        'support, and frozen affine normalization.', '',
        'Balanced corrupted timing pools the same 200 paired repetitions from every official cell. '
        'Per-seed quantiles are computed from those raw measurements before averaging seeds. '
        'Sustained throughput is total measured images divided by summed observed throughput duration; '
        'memory is the maximum observed cell peak. Repetition spread is distinct from sample SD across training seeds. '
        'Profiler intervals are inclusive where named and are never stacked into an invented total. '
        'Hardware classes, CPU/thread configurations, physical devices, and measurement sessions are retained in tables; '
        'different classes are never pooled.', '',
        '| Dataset / architecture | Hardware / GPU / CPU | Batch / scope / input | Fully paired conditions | Raw dense speedup range over all fully paired conditions |',
        '|---|---|---|---|---|']
    fields = ('dataset', 'architecture', 'hardware_identity', 'batch_size', 'scope', 'input_group')
    groups = defaultdict(list)
    for row in timing:
        if row['execution'] != 'dense_masked' and row['inference_execution'] == 'compact': groups[key(row, fields)].append(row)
    for group, rows in sorted(groups.items()):
        d, a, hid, batch, scope, input_group = group
        complete = [r for r in rows if r['paired_raw_dense_speedup_n'] == r['expected_seeds']]
        values = [r['paired_raw_dense_speedup_mean'] for r in complete]
        label = rows[0]
        text = f'{min(values):.3f} to {max(values):.3f}x' if values else 'incomplete; no paired gain established'
        lines.append(f"| {d} / {a} | {hid} / {label.get('gpu', 'unmeasured')} / {label.get('cpu_model', 'unmeasured')} | "
            f'{batch} / {scope} / {input_group} | {len(complete)}/{len(rows)} | {text} |')
    lines += ['', '**Token behavior.** [Saved support table](outputs/tables/token_support.csv) separates clean input '
        'and complete equally weighted corruption cells. Native coefficient zeros, empty patches, padding, Swin '
        'stage activity, and observed latency are distinct quantities. MNIST background sparsity also affects the '
        'raw-pixel sweep and prevents attributing empty-token benefits uniquely to contrast.', '',
        '**Interpretation limits.** This is a follow-up on already observed benchmarks. The checkpoint set and full '
        'inference grid are registered before this study executes. No checkpoint, threshold, recipe, representation, '
        'or corruption-specific setting may be selected using clean-test or corruption-test scores. All levels and '
        'all seeds remain reportable, including worse results. Raw-pixel and native-contrast percentages have different '
        'denominators and should not be treated as the same image transformation.', '',
        f"Coverage and failures: [coverage.json](outputs/tables/coverage.json). Invalid evidence records: {len(coverage['errors'])}. "
        'An invalid artifact is excluded from numerical aggregates and remains an explicit error in coverage.', '',
        'TensorBoard figures and scalars are under `outputs/tensorboard/study_summary/`. '
        'Use `tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006` in this study root.', '',
        'Generated figures (each includes PDF, SVG, 300-dpi PNG, and the exact plot-data CSV):']
    lines += [f'- [{Path(p).stem}]({p})' for p in coverage['figures']]
    lines += ['', 'Original frozen training/validation learning curves (reused references; no new training):']
    seen = set()
    for reference in references.values():
        family = (reference['dataset'], reference['architecture'], reference['representation'], reference['original_training_execution'])
        if family in seen: continue
        seen.add(family); d, a, rep, execution = family
        percent = 'None' if rep == 'raw' else '0'
        for metric in ('loss', 'accuracy'):
            path = BASE_ROOT / 'outputs/figures' / d / a / f'learning_{rep}_{execution}_p{percent}_{metric}.png'
            if path.is_file(): lines.append(f'- [{d}/{a}/{rep}/{execution}: {metric}]({path})')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(); manifest = load_manifest(); cases = manifest['cases']
    if verify_source() != manifest['source_hash']: raise ValueError('Reporter executing source differs from study manifest')
    if not cases or len({c['test_config_hash'] for c in cases}) != len(cases):
        raise ValueError('Manifest cases are empty or duplicated')
    tables = ROOT / 'outputs/tables'; tables.mkdir(parents=True, exist_ok=True)
    common_fields = list(metadata(cases[0]))
    cells = CellTable(tables / 'evaluation_cells.csv', common_fields + ['corruption', 'severity', 'accuracy', 'error', 'loss', 'correct', 'total', 'path'])
    hardware_fields = ['gpu', 'cpu_model', 'torch_num_threads', 'torch_num_interop_threads', 'precision', 'torch', 'cuda', 'cudnn', 'driver_version']
    measurement_fields = ['measurement_session_id', 'measurement_device_uuid', 'hardware_identity', 'hardware_metadata_file', 'hardware_metadata_sha256', 'host', 'scheduler_job_id']
    timed_cells = CellTable(tables / 'timing_cells.csv', common_fields + ['execution', 'batch_size', 'corruption', 'severity', 'scope'] + hardware_fields + measurement_fields +
        ['mean_ms', 'median_ms', 'p95_ms', 'max_memory_allocated', 'max_memory_reserved', 'sustained_images_per_second', 'summary_path'])
    accuracy, tokens, workers, timings, stages, errors, corruptions, failures = [], [], [], [], [], [], [], []
    references, panel_hashes, hardware_cache = {}, {}, {}
    try:
        for case in cases:
            recorded_failures = []
            for path in sorted((case_dir(case) / 'failures').glob('*.json')):
                value = read_json(path)
                failures.append(metadata(case) | {'mode': value['mode'], 'execution': value.get('execution'),
                    'batch_size': value.get('batch_size'), 'shard': value.get('shard'), 'job_id': value.get('job_id'),
                    'time': value.get('time'), 'error_type': value.get('error_type'), 'error': value.get('error'),
                    'path': str(path), 'sha256': file_hash(path)})
                recorded_failures.append((path, value))
            try:
                row, support, corruption = collect_accuracy(case, references, cells)
                row['failure_records'] = ';'.join(str(p) for p, v in recorded_failures if v['mode'] == 'evaluate')
                if row['state'] != 'complete' and row['failure_records']: row['state'] = 'failed_or_partial'
                accuracy.append(row); tokens.extend(support); corruptions.extend(corruption)
            except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
                errors.append({'phase': 'evaluation', 'test_config_hash': case['test_config_hash'], 'error': repr(error)})
                accuracy.append(metadata(case) | {'state': 'invalid', 'completed_cells': 0, 'expected_cells': len(pairs(case['dataset'])), 'error': repr(error)})
            try:
                work, timing, stage = collect_latency(case, panel_hashes, hardware_cache, timed_cells)
                for row in work:
                    row['failure_records'] = ';'.join(str(p) for p, v in recorded_failures if v['mode'] == 'latency'
                        and v.get('execution') == row['execution'] and v.get('batch_size') == row['batch_size'])
                    if row['state'] != 'complete' and row['failure_records']: row['state'] = 'failed_or_partial'
                workers.extend(work); timings.extend(timing); stages.extend(stage)
            except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
                errors.append({'phase': 'latency', 'test_config_hash': case['test_config_hash'], 'error': repr(error)})
                executions = ('compact', 'dense_masked') if case['inference_execution'] == 'compact' else (case['inference_execution'],)
                for execution in executions:
                    for batch in (1, 8):
                        workers.append(metadata(case) | {'execution': execution, 'batch_size': batch, 'state': 'invalid',
                            'completed_cells': 0, 'expected_cells': len(pairs(case['dataset'])), 'error': repr(error)})
    finally:
        cells.close(); timed_cells.close()
    add_paired_changes(accuracy, timings)
    conditions = aggregate_accuracy(accuracy, cases); timing_conditions = aggregate_timing(timings, cases, workers)
    token_conditions = aggregate_named(tokens, cases, ('input_group', 'metric'))
    corruption_conditions = aggregate_named(corruptions, cases, ('corruption', 'severity'), value='error')
    stage_conditions = aggregate_named(stages, cases, ('execution', 'batch_size', 'hardware_identity', 'input_group', 'stage', 'metric'), value='value_ms_mean')
    for name, rows in [('accuracy_seeds', accuracy), ('accuracy_conditions', conditions), ('original_references', list(references.values())),
                       ('normalization', normalization_rows(cases)), ('failures', failures), ('corruption_seeds', corruptions), ('corruption_conditions', corruption_conditions),
                       ('token_support', tokens), ('token_conditions', token_conditions), ('timing_workers', workers), ('timing_seeds', timings),
                       ('timing_conditions', timing_conditions), ('stages', stages), ('stage_conditions', stage_conditions)]:
        save_csv(tables / (name + '.csv'), rows)
    figures = [] if args.no_plots else render_figures(conditions, timing_conditions, tokens) + diagnostic_figures(
        conditions, timing_conditions, timings, token_conditions, corruption_conditions, stage_conditions)
    coverage = coverage_record(manifest, accuracy, workers, errors, figures, verify_figure_outputs(figures))
    coverage['recorded_failure_attempts'] = len(failures)
    atomic_json(tables / 'coverage.json', coverage)
    write_report(manifest, coverage, conditions, timing_conditions, references)
    with SummaryWriter(str(ROOT / 'outputs/tensorboard/study_summary/coverage')) as writer:
        for name in ('completed_evaluation_cases', 'expected_evaluation_cases', 'completed_latency_workers', 'expected_latency_workers'):
            writer.add_scalar(name, coverage[name], 0)
        for row in conditions:
            prefix = '/'.join(str(row[k]) for k in GROUP)
            for metric in METRICS:
                if row[metric + '_mean'] is not None:
                    writer.add_scalar(prefix + '/' + metric + '_mean', row[metric + '_mean'], 0)
                    if row[metric + '_sd'] is not None: writer.add_scalar(prefix + '/' + metric + '_sample_sd', row[metric + '_sd'], 0)
    print(json.dumps({k: v for k, v in coverage.items() if k not in ('figures', 'errors')}, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} invalid evidence records; see outputs/tables/coverage.json')


if __name__ == '__main__':
    main()
