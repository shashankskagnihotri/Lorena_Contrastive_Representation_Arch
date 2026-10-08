"""Evidence-only tables, scientific figures, and an explicit coverage report."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="talk", style="whitegrid", font='serif', font_scale=1.0)
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .common import ROOT, atomic_json, file_hash
from .data import corruption_cells
from .metrics import aggregate_cells
from .scope import configurations as scoped_configurations, representations as scoped_representations, read_scope, expected_main_count, validate_registry

REPRESENTATIONS = ('single_color', 'grayscale', 'color_opponency')
COLORS = dict(zip(REPRESENTATIONS, sns.color_palette('colorblind', 3)))
MARKERS = dict(zip(REPRESENTATIONS, ('o', 's', '^')))
SCOPES = ('gpu_raw_to_logits', 'host_raw_to_logits', 'loader_to_cpu_prediction')


LEGACY_SCOPE = {'schema_version': 1, 'active_representations': {dataset: list(REPRESENTATIONS) for dataset in ('mnist', 'cifar10')}}


def configurations(dataset='cifar10', scope=None):
    """Stable dataset-specific order; standalone callers retain legacy defaults."""
    return scoped_configurations(dataset, LEGACY_SCOPE if scope is None else scope)


def representations(dataset, scope=None):
    return scoped_representations(dataset, LEGACY_SCOPE if scope is None else scope)


def condition_label(representation, execution, percent):
    if representation == 'raw':
        return 'Raw dense'
    prefix = {'single_color': 'Color', 'grayscale': 'Grayscale', 'color_opponency': 'Opponent'}[representation]
    return f'{prefix}, {execution}, {int(percent)}%'


def _condition(row):
    percent = row.get('sparsity_percent')
    percent = None if percent in (None, '', 'NA', 'None', 'null') else int(float(percent))
    return row['representation'], row['execution'], percent


def _float(value):
    if value in (None, '', 'NA', 'None', 'null'):
        return None
    return float(value)


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None, None
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite saved metric')
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else None



def training_metadata(row):
    """Exact primary partition metadata; never infer a completed run's count."""
    protocol = row.get('protocol', 'heldout_val')
    if protocol != 'heldout_val':
        raise ValueError('Primary report cannot mix held-out and refit protocols')
    expected = {'mnist': 55000, 'cifar10': 45000}[row['dataset']]
    if 'training_count' in row and _float(row['training_count']) != expected:
        raise ValueError('Result training count differs from its primary partition')
    return {'protocol': protocol, 'training_count': expected}


def validate_completed_training_count(completed, row):
    expected = training_metadata(row)['training_count']
    if 'training_count' not in completed or completed['training_count'] != expected:
        raise ValueError('Completed training count is missing or differs from the primary partition')
    return expected

def save_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**row, **training_metadata(row)} if row.get('dataset') in ('mnist', 'cifar10') else row for row in rows]
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ['status']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_run(root, entry):
    config_hash = entry.get('resolved_config_hash') or entry.get('config_hash')
    if config_hash:
        return root / 'outputs' / 'runs' / entry['protocol'] / entry['registry_id'] / config_hash
    # An unfrozen planned entry must not select an arbitrary pilot or old attempt.
    return None


def collect_evidence(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    scope = read_scope(root)
    registry = manifest['runs']
    validate_registry(registry, scope)
    seed_rows, cell_rows, curve_rows, token_rows, timing_rows, stage_rows = [], [], [], [], [], []
    for entry in registry:
        common = {key: entry[key] for key in ('registry_id', 'dataset', 'architecture', 'representation', 'execution', 'sparsity_percent', 'seed', 'protocol')}
        common.update(training_metadata(common))
        row = {**common, 'training_status': 'planned', 'evaluation_status': 'missing', 'clean_accuracy': None, 'clean_error': None, 'mCE_raw': None, 'clean_count': 0, 'corruption_cells': 0, 'corruption_count': 0}
        rd = resolve_run(root, entry)
        if rd is None or not rd.exists():
            seed_rows.append(row); continue
        row['run_directory'] = str(rd)
        done_path, failure_path = rd / 'completed.json', rd / 'failure.json'
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if not (rd / 'final.pt').exists() or file_hash(rd / 'final.pt') != done['checkpoint_sha256']:
                raise ValueError(f'Invalid final checkpoint: {rd}')
            row.update(training_status='completed', checkpoint_sha256=done['checkpoint_sha256'], training_count=validate_completed_training_count(done, common))
        elif failure_path.exists() or any(rd.glob('failure-*.json')):
            row['training_status'] = 'failed'
        elif (rd / 'latest.pt').exists() or (rd / 'writer.lock').exists():
            row['training_status'] = 'running_or_interrupted'
        for epoch in read_jsonl(rd / 'epochs.jsonl'):
            for split, values in [('train_online', epoch['train_online']), ('heldout_val', epoch.get('validation'))]:
                if values:
                    curve_rows.append({**common, 'epoch': epoch['epoch'], 'split': split, 'loss': values['loss'], 'accuracy': values['accuracy'], 'count': values['total'], 'checkpoint_policy': 'final_epoch_primary'})
            for metric, value in epoch.get('support', {}).items():
                token_rows.append({**common, 'source': 'train_online', 'epoch': epoch['epoch'], 'metric': metric, 'value': value})
        evaluation = rd / 'evaluation'
        valid_cells = []
        if evaluation.exists() and row['training_status'] == 'completed':
            for corruption, severity in [('clean', 'test')] + corruption_cells(entry['dataset']):
                path = evaluation / f'{corruption}__{severity}.json'
                if not path.exists():
                    continue
                cell = json.loads(path.read_text())
                prediction_path = path.with_suffix('.npz')
                if cell['total'] != 10000 or cell['checkpoint_sha256'] != row['checkpoint_sha256'] or not prediction_path.exists() or file_hash(prediction_path) != cell['predictions_sha256']:
                    raise ValueError(f'Invalid saved cell evidence: {path}')
                predictions = np.load(prediction_path, allow_pickle=False)
                if len(predictions['sample_id']) != 10000 or len(set(predictions['sample_id'])) != 10000:
                    raise ValueError(f'Invalid prediction identity coverage: {prediction_path}')
                correct = int((predictions['label'] == predictions['prediction']).sum())
                if correct != cell['correct'] or not math.isclose(cell['accuracy'], correct / 10000, abs_tol=1e-12):
                    raise ValueError(f'Prediction/count mismatch: {path}')
                cell_rows.append({**common, 'corruption': corruption, 'severity': severity, 'accuracy': cell['accuracy'], 'error': cell['error'], 'loss': cell['loss'], 'correct': correct, 'total': 10000})
                for metric, distribution in cell.get('support', {}).items():
                    for summary, value in distribution.items():
                        token_rows.append({**common, 'source': f'{corruption}/{severity}', 'epoch': '', 'metric': metric + '/' + summary, 'value': value})
                if corruption == 'clean':
                    row.update(clean_accuracy=cell['accuracy'], clean_error=cell['error'], clean_count=10000)
                else:
                    valid_cells.append(cell)
            row['corruption_cells'] = len(valid_cells)
            row['corruption_count'] = sum(cell['total'] for cell in valid_cells)
            if len(valid_cells) == len(corruption_cells(entry['dataset'])):
                summary = aggregate_cells(valid_cells, corruption_cells(entry['dataset']))
                row['mCE_raw'] = summary['mCE_raw']
                row['evaluation_status'] = 'complete' if row['clean_count'] == 10000 else 'corruptions_complete_clean_missing'
            elif valid_cells or row['clean_count']:
                row['evaluation_status'] = 'partial'
        benchmark, stages = collect_benchmark(rd, common)
        timing_rows.extend(benchmark); stage_rows.extend(stages)
        seed_rows.append(row)
    return {'experiment_scope': scope, 'manifest_state': manifest.get('state'), 'seeds': seed_rows, 'cells': cell_rows, 'curves': curve_rows, 'tokens': token_rows, 'timings': timing_rows, 'stages': stage_rows}



def collect_benchmark(rd, common):
    """Audit actual cell summaries and derive balanced mixtures from raw repeats."""
    timings, stages = [], []
    mixtures = defaultdict(list)
    mixture_meta, mixture_cells = {}, defaultdict(set)
    mixture_sessions, mixture_devices = defaultdict(set), defaultdict(set)
    pattern = '*/' + '*/batch*/cells/*/summary.json'
    paths = sorted((rd / 'benchmark').glob(pattern)) if (rd / 'benchmark').exists() else []
    if not paths:
        return timings, stages
    completed = json.loads((rd / 'completed.json').read_text())
    config = json.loads((rd / 'config.json').read_text())
    checkpoint_sha256 = file_hash(rd / 'final.pt')
    if checkpoint_sha256 != completed['checkpoint_sha256'] or config['config_hash'] != rd.name:
        raise ValueError(f'Benchmark run completion/configuration invalid: {rd}')
    for path in paths:
        data = json.loads(path.read_text())
        identity = data['identity']
        if data['state'] != 'complete' or identity.get('study_role') != 'primary':
            continue
        if identity['config_hash'] != rd.name:
            raise ValueError(f'Benchmark configuration identity mismatch: {path}')
        if identity['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError(f'Benchmark checkpoint identity mismatch: {path}')
        if identity['normalization_hash'] != config['normalization_hash']:
            raise ValueError(f'Benchmark normalization identity mismatch: {path}')
        for name in data['file_hashes']:
            if file_hash(path.parent / name) != data['file_hashes'][name]:
                raise ValueError(f'Benchmark source hash mismatch: {path.parent / name}')
        measurement = data.get('measurement', {})
        if measurement.get('hardware_metadata_file') and file_hash(measurement['hardware_metadata_file']) != measurement['hardware_metadata_sha256']:
            raise ValueError(f'Benchmark hardware session metadata hash mismatch: {path}')
        common_cell = {**common, 'execution': identity['execution'], 'trained_execution': identity['trained_execution'],
                       'hardware_id': identity['hardware_identity'], 'batch_size': data['batch_size'],
                       'corruption': data['corruption'], 'severity': data['severity'],
                       'input_group': 'clean' if data['corruption'] == 'clean' else f"{data['corruption']}/{data['severity']}",
                       'summary_path': str(path), 'checkpoint_sha256': identity['checkpoint_sha256'],
                       'normalization_hash': identity['normalization_hash'],
                       'measurement_session_id': measurement.get('measurement_session_id', ''),
                       'measurement_device_uuid': measurement.get('measurement_device_uuid', ''),
                       'measurement_host': measurement.get('host', ''),
                       'measurement_scheduler_job_id': measurement.get('scheduler_job_id', ''),
                       'hardware_metadata_file': measurement.get('hardware_metadata_file', ''),
                       'hardware_metadata_sha256': measurement.get('hardware_metadata_sha256', '')}
        for scope, values in data['scopes'].items():
            timings.append({**common_cell, 'scope': scope, **{key: value for key, value in values.items() if not isinstance(value, (dict, list))},
                            'sustained_images_per_second': values['sustained_throughput']['images_per_second'],
                            'peak_allocated_bytes': (values.get('memory') or {}).get('max_memory_allocated'),
                            'peak_reserved_bytes': (values.get('memory') or {}).get('max_memory_reserved')})
        raw = read_jsonl(path.parent / 'timings.jsonl')
        if data['warmup_batches_per_scope'] < 50 or data['measured_batches_per_scope'] < 200:
            raise ValueError(f'Insufficient primary timing repetitions: {path}')
        for scope in data['scopes']:
            measured_rows = [row for row in raw if row['scope'] == scope]
            if len(measured_rows) != data['measured_batches_per_scope'] or len({row['repeat_id'] for row in measured_rows}) != len(measured_rows):
                raise ValueError(f'Incomplete/duplicate primary repetitions: {path}:{scope}')
        if data['corruption'] != 'clean':
            for scope in data['scopes']:
                measured = [float(row['elapsed_ms']) for row in raw if row['scope'] == scope]
                if len(measured) != data['measured_batches_per_scope'] or len(measured) < 200:
                    raise ValueError(f'Incomplete primary measured batches: {path}:{scope}')
                key = (identity['execution'], identity['hardware_identity'], data['batch_size'], scope)
                mixtures[key].extend(measured); mixture_meta[key] = common_cell
                mixture_cells[key].add((data['corruption'], str(data['severity'])))
                if measurement.get('measurement_session_id'):
                    mixture_sessions[key].add(str(measurement['measurement_session_id']))
                if measurement.get('measurement_device_uuid'):
                    mixture_devices[key].add(str(measurement['measurement_device_uuid']))
        profile = json.loads((path.parent / 'profile.json').read_text())
        # Fine kernel tables stay in profile.json; display named method scopes.
        for event in profile['events']:
            if '/' in event['name'] or event['name'] in ('input_conversion', 'frontend_color_blur_subtraction', 'native_sparsification', 'native_support', 'frozen_normalization_and_patch_layout'):
                stages.append({**common_cell, 'scope': 'profiler_diagnostic', 'stage': event['name'], 'count': event['count'],
                               'cpu_self_ms': event['cpu_self_ms'], 'device_self_ms': event['device_self_ms'],
                               'cpu_total_ms_inclusive': event['cpu_total_ms_inclusive'], 'device_total_ms_inclusive': event['device_total_ms_inclusive'],
                               'stage_intervals_additive': False, 'profiled_call_wall_ms': profile['profiled_call_wall_ms'],
                               'matched_uninstrumented_wall_ms': profile['matched_uninstrumented_wall_ms'], 'profiling_overhead_ratio': profile['profiling_overhead_ratio']})
    expected = {(c, str(s)) for c, s in corruption_cells(common['dataset'])}
    for key, measurements in mixtures.items():
        if mixture_cells[key] != expected:
            continue
        values = np.asarray(measurements)
        meta = mixture_meta[key]
        timings.append({**meta, 'corruption': 'balanced_15_primary', 'severity': 'all_released', 'input_group': 'balanced_corruption', 'scope': key[3],
                        'mean_ms': float(values.mean()), 'median_ms': float(np.median(values)), 'p95_ms': float(np.quantile(values, .95)),
                        'count': len(values), 'serial_images_per_second': len(values) * key[2] / (values.sum() / 1000),
                        'amortized_mean_ms_per_image': float(values.mean()) / key[2],
                        'measurement_session_id': 'balanced_mixture_see_measurement_sessions',
                        'measurement_device_uuid': 'balanced_mixture_see_measurement_device_uuids',
                        'measurement_sessions': json.dumps(sorted(mixture_sessions[key])),
                        'measurement_device_uuids': json.dumps(sorted(mixture_devices[key])),
                        'aggregate_definition': 'equal 200 measured batches per official corruption/severity cell; pooled within seed, matched hardware class, and execution; all valid timings retained'})
    return timings, stages

def aggregate_conditions(seed_rows, scope=None):
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row['dataset'], row['architecture'], *_condition(row))].append(row)
    outputs = []
    for (dataset, architecture, representation, execution, percent), rows in groups.items():
        if len(rows) != 3 or {int(row['seed']) for row in rows} != {0, 1, 2}:
            raise ValueError('Primary condition lacks its full three-seed registry')
        metadata = training_metadata(rows[0])
        if any(training_metadata(row) != metadata for row in rows):
            raise ValueError('Mixed training protocol/count in aggregate condition')
        result = dict(dataset=dataset, architecture=architecture, representation=representation, execution=execution, sparsity_percent=percent, condition=condition_label(representation, execution, percent), **metadata)
        for metric in ('clean_accuracy', 'clean_error', 'mCE_raw'):
            values = [float(row[metric]) for row in rows if row.get(metric) not in (None, '')]
            result[metric + '_mean'], result[metric + '_sd'] = mean_sd(values)
            result[metric + '_seeds'] = len(values)
        result['trained_seeds'] = sum(row['training_status'] == 'completed' for row in rows)
        result['fully_evaluated_seeds'] = sum(row['evaluation_status'] == 'complete' for row in rows)
        outputs.append(result)
    orders = {dataset: {condition: i for i, condition in enumerate(configurations(dataset, scope))} for dataset in ('mnist', 'cifar10')}
    return sorted(outputs, key=lambda row: (row['dataset'], row['architecture'], orders[row['dataset']][_condition(row)]))


def paired_changes(seed_rows):
    baseline = {(row['dataset'], row['architecture'], int(row['seed'])): row for row in seed_rows if row['representation'] == 'raw'}
    rows = []
    for row in seed_rows:
        base = baseline[(row['dataset'], row['architecture'], int(row['seed']))]
        for metric in ('clean_accuracy', 'mCE_raw'):
            if row.get(metric) is not None and base.get(metric) is not None:
                rows.append({key: row[key] for key in ('dataset', 'architecture', 'representation', 'execution', 'sparsity_percent', 'seed')} | training_metadata(row) | {'metric': metric, 'method_minus_baseline_percentage_points': 100 * (float(row[metric]) - float(base[metric]))})
    return rows


def save_figure(fig, stem, writer=None):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.' + extension), dpi=300, bbox_inches='tight')
    if writer:
        writer.add_figure(stem.name, fig, 0, close=False)
    plt.close(fig)


def metric_curve(rows, metric, ylabel, output, writer):
    available = [row for row in rows if _float(row.get(metric + '_mean')) is not None]
    if not available:
        return []
    fig, ax = plt.subplots(figsize=(8, 5.6), layout='constrained')
    for rep in REPRESENTATIONS:
        curve = sorted((row for row in available if row['representation'] == rep and row['execution'] == 'compact'), key=lambda row: int(float(row['sparsity_percent'])))
        if curve:
            ax.errorbar([int(float(row['sparsity_percent'])) for row in curve], [100 * _float(row[metric + '_mean']) for row in curve], yerr=[100 * (_float(row[metric + '_sd']) or 0) for row in curve], color=COLORS[rep], marker=MARKERS[rep], capsize=3, label=rep.replace('_', ' '))
        dense = [row for row in available if row['representation'] == rep and row['execution'] == 'dense']
        if dense:
            ax.scatter([0], [100 * _float(dense[0][metric + '_mean'])], marker='D', facecolors='none', edgecolors=[COLORS[rep]], label=rep.replace('_', ' ') + ' dense 0%')
    raw = [row for row in available if row['representation'] == 'raw']
    if raw:
        ax.axhline(100 * _float(raw[0][metric + '_mean']), color='black', linestyle='--', label='Raw dense')
    ax.set(xlabel='Requested coefficient sparsity (%)', ylabel=ylabel, xticks=[0, 20, 40, 60, 80])
    ax.legend(fontsize=9)
    save_figure(fig, output, writer)
    save_csv(output.with_suffix('.csv'), available)
    return [str(output.with_suffix('.png'))]


def corruption_plots(cells, output, writer, scope=None):
    corrupt = [row for row in cells if row['corruption'] != 'clean']
    if not corrupt:
        return []
    by = defaultdict(list)
    for row in corrupt:
        by[(_condition(row), row['corruption'], int(row['seed']))].append(float(row['error']))
    aggregate = defaultdict(list)
    for (condition, corruption, seed), values in by.items():
        required = 5 if cells[0]['dataset'] == 'cifar10' else 1
        if len(values) == required:
            aggregate[(condition, corruption)].append(float(np.mean(values)))
    names = list(dict.fromkeys(c for c, _ in corruption_cells(cells[0]['dataset'])))
    conditions = configurations(cells[0]['dataset'], scope)
    grid = np.full((len(conditions), len(names)), np.nan)
    plot_data = []
    for i, condition in enumerate(conditions):
        for j, corruption in enumerate(names):
            values = aggregate.get((condition, corruption), [])
            mean, sd = mean_sd(values)
            if mean is not None:
                grid[i, j] = 100 * mean
            plot_data.append(dict(dataset=cells[0]['dataset'], architecture=cells[0]['architecture'], **training_metadata(cells[0]), condition=condition_label(*condition), corruption=corruption, mean_error_percent=None if mean is None else 100 * mean, seed_sd_percent=None if sd is None else 100 * sd, seeds=len(values)))
    figures = []
    if np.isfinite(grid).any():
        fig, ax = plt.subplots(figsize=(17, 11), layout='constrained')
        sns.heatmap(grid, mask=~np.isfinite(grid), cmap='mako', vmin=0, vmax=100, xticklabels=names, yticklabels=[condition_label(*c) for c in conditions], ax=ax, cbar_kws={'label': 'Mean error (%)'})
        ax.tick_params(axis='both', labelsize=10); ax.set(xlabel='Corruption', ylabel='Trained configuration')
        plt.setp(ax.get_xticklabels(), rotation=50, ha='right')
        stem = output / 'per_corruption_error'; save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), plot_data); figures.append(str(stem.with_suffix('.png')))
    if cells[0]['dataset'] == 'cifar10':
        for rep in REPRESENTATIONS:
            fig, ax = plt.subplots(figsize=(8, 5.6), layout='constrained'); points = []
            for percent in (0, 20, 40, 60, 80):
                means, sds, levels = [], [], []
                for severity in range(1, 6):
                    values = []
                    for seed in (0, 1, 2):
                        selected = [float(row['error']) for row in corrupt if row['representation'] == rep and row['execution'] == 'compact' and int(float(row['sparsity_percent'])) == percent and int(row['severity']) == severity and int(row['seed']) == seed]
                        if len(selected) == 15:
                            values.append(float(np.mean(selected)))
                    mean, sd = mean_sd(values)
                    if mean is not None:
                        levels.append(severity); means.append(100 * mean); sds.append(100 * (sd or 0)); points.append(dict(dataset=cells[0]['dataset'], architecture=cells[0]['architecture'], **training_metadata(cells[0]), representation=rep, sparsity_percent=percent, severity=severity, mean_error_percent=100 * mean, seed_sd_percent=None if sd is None else 100 * sd, seeds=len(values)))
                if levels:
                    ax.errorbar(levels, means, yerr=sds, marker='o', label=f'{percent}%')
            if points:
                ax.set(xlabel='Severity', ylabel='Mean corruption error (%)', xticks=range(1, 6)); ax.legend(title='Sparsity', fontsize=10)
                stem = output / f'severity_{rep}'; save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), points); figures.append(str(stem.with_suffix('.png')))
            else:
                plt.close(fig)
    return figures


def learning_curves(rows, output, writer, scope=None):
    figures = []
    if not rows:
        return figures
    for condition in configurations(rows[0]['dataset'], scope):
        selected = [row for row in rows if _condition(row) == condition]
        if not selected:
            continue
        for metric in ('loss', 'accuracy'):
            fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
            for split, linestyle in [('train_online', '-'), ('heldout_val', '--')]:
                for seed in (0, 1, 2):
                    curve = sorted((row for row in selected if row['split'] == split and int(row['seed']) == seed), key=lambda row: int(row['epoch']))
                    if curve:
                        ax.plot([int(row['epoch']) + 1 for row in curve], [float(row[metric]) * (100 if metric == 'accuracy' else 1) for row in curve], color=sns.color_palette('colorblind')[seed], linestyle=linestyle, label=f'{split}, seed {seed}')
            ax.set(xlabel='Epoch', ylabel='Accuracy (%)' if metric == 'accuracy' else 'Cross-entropy', title=condition_label(*condition)); ax.legend(fontsize=9)
            name = f'learning_{condition[0]}_{condition[1]}_p{condition[2]}_{metric}'
            stem = output / name; save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), selected); figures.append(str(stem.with_suffix('.png')))
    return figures


def normalize_timing(row):
    """Canonical summary keys, without turning unsupported scopes into numbers."""
    result = dict(row)
    aliases = {'median_ms': ('median_ms', 'median_wall_ms', 'wall_median_ms'), 'p95_ms': ('p95_ms', 'p95_wall_ms', 'wall_p95_ms'), 'mean_ms': ('mean_ms', 'mean_wall_ms', 'wall_mean_ms')}
    for target, names in aliases.items():
        result[target] = next((_float(row[name]) for name in names if name in row and _float(row[name]) is not None), None)
    result['input_group'] = row.get('input_group', row.get('source', row.get('input', 'clean')))
    result['batch_size'] = int(row.get('batch_size', 1))
    result['hardware_id'] = row.get('hardware_id', row.get('gpu_name', 'unspecified'))
    return result


def timing_overviews(rows, output, writer, experiment_scope=None):
    """Dataset-specific views, seed points, mean +/- between-seed sample SD."""
    rows = [normalize_timing(row) for row in rows]
    groups = defaultdict(list)
    for row in rows:
        if row.get('scope') in SCOPES and row['input_group'] in ('clean', 'balanced_corruption', 'corruption_balanced') and row.get('execution') != 'dense_masked':
            groups[(row['scope'], row['batch_size'], row['input_group'], row['hardware_id'])].append(row)
    figures, tables = [], []
    for (scope, batch_size, input_group, hardware), selected in groups.items():
        conditions = configurations(selected[0]['dataset'], experiment_scope)
        for statistic in ('median_ms', 'p95_ms'):
            if not any(row[statistic] is not None for row in selected):
                continue
            fig, ax = plt.subplots(figsize=(11, 11), layout='constrained'); data = []
            for y, condition in enumerate(conditions):
                seed_values = {}
                for row in selected:
                    if _condition(row) == condition and row[statistic] is not None:
                        if int(row['seed']) in seed_values:
                            raise ValueError('Multiple summaries for same seed/condition/hardware; session aggregation must be explicit')
                        seed_values[int(row['seed'])] = row[statistic]
                for seed, value in sorted(seed_values.items()):
                    ax.scatter(value, y + (seed - 1) * 0.12, color=sns.color_palette('colorblind')[seed], marker=('o', 's', '^')[seed], s=32)
                mean, sd = mean_sd(list(seed_values.values()))
                if mean is not None:
                    ax.errorbar(mean, y, xerr=sd or 0, color='black', marker='|', markersize=14, capsize=3)
                else:
                    ax.text(0.99, y, 'not measured', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8, color='gray')
                data.append(dict(dataset=selected[0]['dataset'], architecture=selected[0]['architecture'], **training_metadata(selected[0]), condition=condition_label(*condition), scope=scope, batch_size=batch_size, input_group=input_group, hardware_id=hardware, statistic=statistic, seeds=len(seed_values), mean_seed_statistic_ms=mean, between_seed_sd_ms=sd, **{f'seed{seed}_ms': seed_values.get(seed) for seed in (0, 1, 2)}))
            ax.set(yticks=range(len(conditions)), yticklabels=[condition_label(*c) for c in conditions], xlabel=f'{scope}: {statistic.replace("_ms", "")} (ms/batch)'); ax.invert_yaxis(); ax.tick_params(axis='y', labelsize=11)
            safe_hardware = ''.join(c if c.isalnum() else '_' for c in hardware)
            stem = output / f'latency_{len(conditions)}_{scope}_b{batch_size}_{input_group}_{safe_hardware}_{statistic}'
            save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), data); figures.append(str(stem.with_suffix('.png'))); tables.extend(data)
    return figures, tables



def timing_comparisons(rows, condition_rows, output, writer, experiment_scope=None):
    """Seed-matched online curves and observed accuracy/robustness tradeoffs."""
    normalized = [normalize_timing(row) for row in rows]
    selected = [row for row in normalized if row.get('scope') in SCOPES and row['input_group'] in ('clean', 'balanced_corruption') and row['execution'] != 'dense_masked']
    groups = defaultdict(list)
    for row in selected:
        groups[(row['scope'], row['batch_size'], row['input_group'], row['hardware_id'])].append(row)
    figures = []
    condition_lookup = {_condition(row): row for row in condition_rows}
    for (scope, batch, source, hardware), values in groups.items():
        baseline = {int(row['seed']): row for row in values if row['representation'] == 'raw'}
        points = []
        for row in values:
            base = baseline.get(int(row['seed']))
            speedup = base['median_ms'] / row['median_ms'] if base and base['median_ms'] is not None and row['median_ms'] else None
            points.append({**row, 'paired_seed_ratio_of_median_speedup': speedup})
        hardware_short = hardware[:12]
        for metric, ylabel in [('median_ms', 'Median online latency (ms/batch)'), ('paired_seed_ratio_of_median_speedup', 'Matched raw / method median latency')]:
            if scope == 'loader_to_cpu_prediction' and metric != 'paired_seed_ratio_of_median_speedup':
                continue
            fig, ax = plt.subplots(figsize=(8, 5.6), layout='constrained'); any_data = False
            for rep in REPRESENTATIONS:
                xx, yy, sd = [], [], []
                for percent in (0, 20, 40, 60, 80):
                    observed = [row[metric] for row in points if row['representation'] == rep and row['execution'] == 'compact' and _condition(row)[2] == percent and row[metric] is not None]
                    mean, deviation = mean_sd(observed)
                    if mean is not None:
                        xx.append(percent); yy.append(mean); sd.append(deviation or 0); any_data = True
                if xx:
                    ax.errorbar(xx, yy, yerr=sd, color=COLORS[rep], marker=MARKERS[rep], capsize=3, label=rep.replace('_', ' '))
            if any_data:
                if metric.endswith('speedup'):
                    ax.axhline(1, color='black', linestyle='--')
                ax.set(xlabel='Requested coefficient sparsity (%)', ylabel=ylabel, xticks=[0, 20, 40, 60, 80]); ax.legend(fontsize=10)
                stem = output / f'{metric}_{scope}_b{batch}_{source}_{hardware_short}'
                save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), points); figures.append(str(stem.with_suffix('.png')))
            else:
                plt.close(fig)
        if scope == 'loader_to_cpu_prediction':
            continue
        for accuracy_metric, ylabel in [('clean_accuracy', 'Clean test accuracy (%)'), ('mCE_raw', 'Mean corruption error (%)')]:
            fig, ax = plt.subplots(figsize=(8, 5.6), layout='constrained'); plotted = []
            for condition in configurations(values[0]['dataset'], experiment_scope):
                accuracy = condition_lookup.get(condition)
                if not accuracy or _float(accuracy.get(accuracy_metric + '_mean')) is None:
                    continue
                latencies = [row['median_ms'] for row in points if _condition(row) == condition and row['median_ms'] is not None]
                latency, sd = mean_sd(latencies)
                if latency is None:
                    continue
                rep, execution, percent = condition
                y = 100 * _float(accuracy[accuracy_metric + '_mean'])
                ax.errorbar(latency, y, xerr=sd or 0, yerr=100 * (_float(accuracy.get(accuracy_metric + '_sd')) or 0), color=COLORS.get(rep, 'black'), marker='D' if execution == 'dense' else MARKERS.get(rep, 'o'), capsize=3)
                if percent is not None:
                    ax.annotate(f'{percent}%', (latency, y), xytext=(4, 3), textcoords='offset points', fontsize=9)
                plotted.append(dict(dataset=accuracy['dataset'], architecture=accuracy['architecture'], **training_metadata(accuracy), condition=condition_label(*condition), mean_seed_median_ms=latency, between_seed_sd_ms=sd, **{accuracy_metric + '_percent': y}))
            if plotted:
                ax.set(xlabel=f'{scope} (ms/batch)', ylabel=ylabel)
                stem = output / f'tradeoff_{accuracy_metric}_{scope}_b{batch}_{source}_{hardware_short}'
                save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), plotted); figures.append(str(stem.with_suffix('.png')))
            else:
                plt.close(fig)
    return figures


def benchmark_coverage(evidence):
    scopes = (*SCOPES, 'cached_input_model_only_diagnostic')
    expected = set()
    for row in evidence['seeds']:
        executions = ('compact', 'dense_masked') if row['execution'] == 'compact' else ('dense',)
        for execution in executions:
            for batch in (1, 8):
                for scope in scopes:
                    for source in ('clean', 'balanced_corruption'):
                        expected.add((row['registry_id'], execution, batch, scope, source))
    observed_by_hardware = defaultdict(set)
    for raw in evidence['timings']:
        row = normalize_timing(raw)
        if row['input_group'] in ('clean', 'balanced_corruption'):
            observed_by_hardware[row['hardware_id']].add((row['registry_id'], row['execution'], row['batch_size'], row['scope'], row['input_group']))
    complete = any(expected <= observed for observed in observed_by_hardware.values())
    return {'expected_summary_conditions_per_hardware': len(expected), 'observed_by_hardware': {key: len(value & expected) for key, value in observed_by_hardware.items()}, 'complete_on_one_matched_hardware': complete}

def token_plots(rows, output, writer):
    groups = defaultdict(list)
    for row in rows:
        if row['source'] == 'clean/test' and row['execution'] == 'compact':
            groups[row['metric']].append(row)
    figures = []
    for metric, selected in groups.items():
        if not any(term in metric for term in ('zero_fraction', 'retained', 'active', 'padding', 'executed', 'window')) or not metric.endswith('/mean'):
            continue
        fig, ax = plt.subplots(figsize=(8, 5.6), layout='constrained')
        for rep in REPRESENTATIONS:
            xx, yy, ss = [], [], []
            for percent in (0, 20, 40, 60, 80):
                values = [float(row['value']) for row in selected if row['representation'] == rep and int(float(row['sparsity_percent'])) == percent]
                mean, sd = mean_sd(values)
                if mean is not None:
                    xx.append(percent); yy.append(mean); ss.append(sd or 0)
            if xx:
                ax.errorbar(xx, yy, yerr=ss, color=COLORS[rep], marker=MARKERS[rep], label=rep.replace('_', ' '), capsize=3)
        ax.set(xlabel='Requested coefficient sparsity (%)', ylabel=metric.removesuffix('/mean').replace('_', ' ')); ax.legend(fontsize=10)
        stem = output / ('support_' + metric.replace('/', '_')); save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), selected); figures.append(str(stem.with_suffix('.png')))
    return figures



def balanced_stage_rows(rows):
    """Keep clean profiles separate; average each complete corruption cell once.

    Profile names may have several input-shape groups, which are first summed
    within their one call and stage name. Across cells, the arithmetic mean is
    a balanced diagnostic interval, never an additive online total.
    """
    clean, corrupt = [], defaultdict(list)
    for row in rows:
        if row.get('corruption') == 'clean' or row.get('input_group') == 'clean':
            clean.append({**row, 'input_group': 'clean'})
        else:
            key = (row['registry_id'], row['execution'], str(row['batch_size']), row['hardware_id'], row['stage'])
            corrupt[key].append(row)
    output = clean
    time_keys = ('cpu_self_ms', 'device_self_ms', 'cpu_total_ms_inclusive', 'device_total_ms_inclusive')
    for key, grouped in corrupt.items():
        expected = {(c, str(s)) for c, s in corruption_cells(grouped[0]['dataset'])}
        per_cell = defaultdict(list)
        for row in grouped:
            per_cell[(row['corruption'], str(row['severity']))].append(row)
        if set(per_cell) != expected:
            # Not every cell executed this stage, so no all-cell stage mean is
            # invented. The fine per-cell table still contains all evidence.
            continue
        aggregate = {**grouped[0], 'input_group': 'balanced_corruption', 'corruption': 'balanced_15_primary', 'severity': 'all_released', 'contributing_cells': len(per_cell), 'aggregate_definition': 'equal mean over every official corruption/severity cell, within independently trained seed'}
        for metric in time_keys:
            aggregate[metric] = float(np.mean([sum(float(row[metric]) for row in cell) for cell in per_cell.values()]))
        output.append(aggregate)
    return output

def stage_plots(rows, output, writer, experiment_scope=None):
    """Separate reported stage intervals; never invent an additive residual."""
    if not rows:
        return []
    numeric = [row for row in balanced_stage_rows(rows) if any(_float(row.get(k)) is not None for k in ('self_cuda_time_us', 'self_device_time_us', 'cpu_time_total_us', 'mean_ms', 'device_total_ms_inclusive'))]
    if not numeric:
        return []
    figures = []
    grouped = defaultdict(list)
    for row in numeric:
        grouped[(row.get('scope', 'profiler_diagnostic'), row.get('batch_size', 1), row.get('hardware_id', 'unspecified'), row['input_group'])].append(row)
    for (scope, batch, hardware, input_group), selected in grouped.items():
        # Profiler nested totals are not stacked or summed into online latency.
        names = sorted({str(row.get('stage', row.get('name', 'unknown'))) for row in selected})
        conditions = configurations(selected[0]['dataset'], experiment_scope)
        grid = np.full((len(conditions), len(names)), np.nan)
        for i, condition in enumerate(conditions):
            for j, name in enumerate(names):
                values = []
                for row in selected:
                    if _condition(row) == condition and str(row.get('stage', row.get('name'))) == name:
                        value = _float(row.get('mean_ms'))
                        if value is None:
                            value = _float(row.get('device_total_ms_inclusive'))
                        if value is None:
                            raw = next((_float(row[key]) for key in ('self_cuda_time_us', 'self_device_time_us') if _float(row.get(key)) is not None), None)
                            value = None if raw is None else raw / 1000
                        if value is not None:
                            values.append(value)
                if values:
                    grid[i, j] = float(np.mean(values))
        if not np.isfinite(grid).any():
            continue
        fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.65), 11), layout='constrained')
        sns.heatmap(grid, mask=~np.isfinite(grid), cmap='viridis', xticklabels=names, yticklabels=[condition_label(*c) for c in conditions], ax=ax, cbar_kws={'label': 'Reported stage interval (ms)'})
        ax.tick_params(labelsize=9); plt.setp(ax.get_xticklabels(), rotation=60, ha='right'); ax.set_xlabel(f'{input_group}: inclusive diagnostic stages, not an additive total')
        stem = output / f'stages_{scope}_b{batch}_{input_group}_{hardware[:12]}'; save_figure(fig, stem, writer); save_csv(stem.with_suffix('.csv'), selected); figures.append(str(stem.with_suffix('.png')))
    return figures



def preprocessing_figures(root):
    scope = read_scope(root)
    source = root / 'outputs' / 'preprocessing_pilot' / 'summary.csv'
    if not source.exists():
        return []
    rows = read_csv(source)
    figures = []
    for dataset in ('mnist', 'cifar10'):
        for representation in representations(dataset, scope):
            selected = [row for row in rows if row['dataset'] == dataset and row['representation'] == representation]
            if not selected:
                continue
            fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
            for patch, marker in [(1, 'o'), (2, 's'), (4, '^')]:
                values = sorted((row for row in selected if int(row['patch_size']) == patch), key=lambda row: float(row['sparsity_percent']))
                if values:
                    ax.plot([float(row['sparsity_percent']) for row in values], [100 * float(row['empty_patch_fraction']) for row in values], marker=marker, label=f'Patch {patch}x{patch}')
            ax.set(xlabel='Requested coefficient sparsity (%)', ylabel='Empty patches (%)', xticks=[0, 20, 40, 60, 80]); ax.legend(fontsize=11)
            stem = root / 'outputs' / 'figures' / dataset / 'preprocessing' / f'empty_patches_{representation}'
            with SummaryWriter(str(root / 'outputs' / 'tensorboard' / 'study_summary' / 'heldout_val' / dataset / 'preprocessing' / 'report')) as writer:
                save_figure(fig, stem, writer)
            save_csv(stem.with_suffix('.csv'), selected); figures.append(str(stem.with_suffix('.png')))
            primary = [row for row in selected if int(row['patch_size']) == 2]
            if primary:
                fig, ax = plt.subplots(figsize=(7.5, 5.5), layout='constrained')
                primary = sorted(primary, key=lambda row: float(row['sparsity_percent']))
                ax.plot([float(row['sparsity_percent']) for row in primary], [100 * float(row['achieved_zero_fraction']) for row in primary], color=COLORS[representation], marker=MARKERS[representation])
                ax.plot([0, 80], [0, 80], color='gray', linestyle='--')
                ax.set(xlabel='Requested coefficient sparsity (%)', ylabel='Achieved native zero fraction (%)', xticks=[0, 20, 40, 60, 80])
                stem = root / 'outputs' / 'figures' / dataset / 'preprocessing' / f'achieved_sparsity_{representation}'
                with SummaryWriter(str(root / 'outputs' / 'tensorboard' / 'study_summary' / 'heldout_val' / dataset / 'preprocessing' / 'report')) as writer:
                    save_figure(fig, stem, writer)
                save_csv(stem.with_suffix('.csv'), primary); figures.append(str(stem.with_suffix('.png')))
    return figures

def normalization_rows(root):
    scope = read_scope(root)
    rows = []
    for path in sorted((root / 'normalization').glob('*_heldout_val.json')) if (root / 'normalization').exists() else []:
        data = json.loads(path.read_text())
        representation = data.get('representation', data.get('frontend_config', {}).get('representation'))
        if representation not in representations(data['dataset'], scope):
            continue
        if not all(key in data for key in ('mean', 'sigma', 'effective_std')):
            continue
        for channel, (mean, sigma, effective) in enumerate(zip(data['mean'], data['sigma'], data['effective_std'])):
            rows.append(dict(artifact=str(path), dataset=data.get('dataset'), representation=data.get('representation', data.get('frontend_config', {}).get('representation')), channel=channel, mean=mean, measured_std=sigma, effective_std=effective, guarded=data['guarded_channels'][channel], artifact_sha256=data.get('artifact_sha256'), image_count=data.get('images', data.get('image_count'))))
    return rows


def format_metric(row, metric):
    value = row.get(metric + '_mean')
    if value in (None, ''):
        return 'not measured'
    sd = row.get(metric + '_sd')
    text = f'{100 * float(value):.2f}'
    if sd not in (None, ''):
        text += f' +/- {100 * float(sd):.2f}'
    return text + f' ({row[metric + "_seeds"]}/3 seeds)'


def research_interpretation(evidence):
    """Six research questions answered only from complete paired saved evidence."""
    experiment_scope = evidence.get('experiment_scope')
    seeds = evidence.get('seeds', [])
    tokens = evidence.get('tokens', [])
    timings = [normalize_timing(row) for row in evidence.get('timings', [])]
    groups = [(d, a) for d in ('mnist', 'cifar10') for a in ('vit_small', 'swin_tiny')]
    lines = ['**Contrast versus raw input and agreement across seeds.** Positive clean-accuracy changes and positive corruption-error reductions favor contrast. Ranges below include every fully observed contrast condition, including dense controls, with each dataset\'s active-condition denominator; no best condition is selected.']
    lines += ['', '| Dataset / architecture | Clean gain, pp | Corruption-error reduction, pp |', '|---|---|---|']
    def describe(pairs, expected):
        if not pairs:
            return f'Incomplete: 0/{expected} conditions with three paired seeds'
        means = [float(np.mean(values)) for values in pairs]
        plus = sum(all(value > 0 for value in values) for values in pairs)
        minus = sum(all(value < 0 for value in values) for values in pairs)
        return f'{min(means):+.2f} to {max(means):+.2f}; {len(pairs)}/{expected} complete; all-three-seed positive {plus}, negative {minus}'
    for dataset, architecture in groups:
        rows = [row for row in seeds if row['dataset'] == dataset and row['architecture'] == architecture]
        lookup = {(_condition(row), int(row['seed'])): row for row in rows}
        summaries = []
        for metric, sign in [('clean_accuracy', 1), ('mCE_raw', -1)]:
            pairs = []
            for condition in configurations(dataset, experiment_scope)[1:]:
                values = []
                for seed in (0, 1, 2):
                    method, raw = lookup.get((condition, seed)), lookup.get((('raw', 'dense', None), seed))
                    if method and raw and _float(method.get(metric)) is not None and _float(raw.get(metric)) is not None:
                        values.append(sign * 100 * (float(method[metric]) - float(raw[metric])))
                if len(values) == 3:
                    pairs.append(values)
            summaries.append(describe(pairs, len(configurations(dataset, experiment_scope)) - 1))
        lines.append(f'| {dataset} / {architecture} | {summaries[0]} | {summaries[1]} |')
    lines += ['', '**Effect of coefficient sparsity.** Endpoint changes below compare compact 80% against compact 0% within each seed; all five requested sparsity settings must be observed for all three seeds before a representation is summarized. Endpoint differences do not establish monotonicity.']
    sparsity_rows = []
    for dataset, architecture in groups:
        rows = [row for row in seeds if row['dataset'] == dataset and row['architecture'] == architecture]
        lookup = {(_condition(row), int(row['seed'])): row for row in rows}
        for representation in representations(dataset, experiment_scope):
            changes = []
            for metric in ('clean_accuracy', 'mCE_raw'):
                complete = all((entry := lookup.get(((representation, 'compact', percent), seed))) is not None and _float(entry.get(metric)) is not None for percent in (0, 20, 40, 60, 80) for seed in (0, 1, 2))
                if complete:
                    values = [100 * (float(lookup[((representation, 'compact', 80), seed)][metric]) - float(lookup[((representation, 'compact', 0), seed)][metric])) for seed in (0, 1, 2)]
                    mean, sd = mean_sd(values)
                    changes.append(f'{mean:+.2f} +/- {sd:.2f} pp; signs ' + '/'.join('+' if value > 0 else '-' if value < 0 else '0' for value in values))
                else:
                    changes.append('incomplete five-level, three-seed comparison')
            if any(not change.startswith('incomplete') for change in changes):
                sparsity_rows.append((dataset, architecture, representation, *changes))
    if sparsity_rows:
        lines += ['', '| Dataset / architecture / representation | Clean accuracy, 80%-0% | Mean corruption error, 80%-0% |', '|---|---|---|']
        lines += [f'| {d} / {a} / {r} | {clean} | {corrupt} |' for d, a, r, clean, corrupt in sparsity_rows]
    else:
        lines += ['The complete five-level, three-seed sparsity comparisons are not yet available.']
    lines += ['', '**Tokens removed and Swin hierarchy.** The following are fractions of the original spatial grid, not FLOP or speedup claims. Corrupted-input values require every official corruption/severity cell and are averaged equally within each seed before seed averaging.']
    token_rows = []
    stage_rows = []
    for dataset, architecture in groups:
        rows = [row for row in tokens if row['dataset'] == dataset and row['architecture'] == architecture and row['execution'] == 'compact']
        original_grid = 14 if dataset == 'mnist' else 16
        sources = {'clean': {'clean/test'}, 'corrupted': {f'{c}/{s}' for c, s in corruption_cells(dataset)}}
        compact_conditions = [condition for condition in configurations(dataset, experiment_scope) if condition[1] == 'compact']
        def values_by_condition(metric, required_sources, transform):
            results = []
            for representation, execution, percent in compact_conditions:
                per_seed = []
                for seed in (0, 1, 2):
                    matched = [row for row in rows if _condition(row) == (representation, execution, percent) and int(row['seed']) == seed and row['metric'] == metric and row['source'] in required_sources]
                    if len(matched) == len(required_sources) and {row['source'] for row in matched} == required_sources:
                        per_seed.append(float(np.mean([transform(float(row['value'])) for row in matched])))
                if len(per_seed) == 3:
                    results.append(float(np.mean(per_seed)))
            return results
        for source, required in sources.items():
            removed = values_by_condition('retained_patches/mean', required, lambda value: 100 * (1 - value / original_grid ** 2))
            if removed:
                token_rows.append(f'| {dataset} / {architecture} / {source} | {min(removed):.2f} to {max(removed):.2f}% | {len(removed)}/{len(compact_conditions)} |')
            if architecture == 'swin_tiny':
                grids = (14, 7, 4, 2) if dataset == 'mnist' else (16, 8, 4, 2)
                ranges = []
                for stage, grid in enumerate(grids):
                    occupancy = values_by_condition(f'stages/{stage}/blocks/0/active_per_image/mean', required, lambda value, grid=grid: 100 * value / grid ** 2)
                    ranges.append(f'S{stage + 1}: {min(occupancy):.1f}-{max(occupancy):.1f}% ({len(occupancy)}/{len(compact_conditions)})' if occupancy else f'S{stage + 1}: incomplete')
                if any('incomplete' not in value for value in ranges):
                    stage_rows.append(f'- {dataset}, {source}: ' + '; '.join(ranges) + '.')
    if token_rows:
        lines += ['', '| Dataset / architecture / input | Initial image patches removed | Three-seed conditions |', '|---|---|---|', *token_rows]
        if len(token_rows) < 8:
            lines += ['Unlisted dataset/architecture/input token comparisons remain incomplete.']
    else:
        lines += ['Complete three-seed clean/corrupted token-removal results are unavailable.']
    if stage_rows:
        lines += ['', 'Swin active-grid occupancy after successive merge boundaries (S1 is before the first merge):', *stage_rows]
        lines += ['Occupancy approaching 100% locates stages becoming dense; an observed range is retained rather than assigning one density threshold to every image.']
    else:
        lines += ['Swin stage-density evidence is incomplete, so the stage at which merging restores density cannot yet be identified.']
    lines += ['', '**Measured online gains at batches 1 and 8.** Speedups below are raw median latency divided by method median latency, paired by seed, input group, batch, and matched hardware class. Ranges include all active contrast conditions for the dataset, separately for GPU-resident and pinned-host input.']
    speed_rows = []
    for dataset, architecture in groups:
        rows = [row for row in timings if row.get('dataset') == dataset and row.get('architecture') == architecture and row.get('execution') != 'dense_masked']
        hardware_ids = sorted({row['hardware_id'] for row in rows})
        for hardware in hardware_ids:
            for scope in SCOPES[:2]:
                for batch in (1, 8):
                    for source in ('clean', 'balanced_corruption'):
                        selected = [row for row in rows if row['hardware_id'] == hardware and row['scope'] == scope and row['batch_size'] == batch and row['input_group'] == source]
                        lookup = {(_condition(row), int(row['seed'])): row for row in selected}
                        condition_ratios = []
                        for condition in configurations(dataset, experiment_scope)[1:]:
                            ratios = []
                            for seed in (0, 1, 2):
                                method, raw = lookup.get((condition, seed)), lookup.get((('raw', 'dense', None), seed))
                                if method and raw and method['median_ms'] is not None and raw['median_ms'] is not None and method['median_ms'] > 0:
                                    ratios.append(raw['median_ms'] / method['median_ms'])
                            if len(ratios) == 3:
                                condition_ratios.append(ratios)
                        if condition_ratios:
                            means = [float(np.mean(values)) for values in condition_ratios]
                            fast = sum(all(value > 1 for value in values) for values in condition_ratios)
                            slow = sum(all(value < 1 for value in values) for values in condition_ratios)
                            speed_rows.append(f'| {dataset} / {architecture} | {scope} / B{batch} / {source} / {hardware[:12]} | {min(means):.3f}-{max(means):.3f}x | {len(means)}/{len(configurations(dataset, experiment_scope)) - 1}; faster all three {fast}, slower all three {slow} |')
    if speed_rows:
        lines += ['', '| Dataset / architecture | Scope / batch / input / hardware | Mean paired speedup range | Coverage and seed agreement |', '|---|---|---|---|', *speed_rows]
        if len(speed_rows) < 32:
            lines += ['Unlisted GPU/host, batch-1/batch-8, and clean/corrupted timing groups remain incomplete.']
    else:
        lines += ['GPU/host speedups at batches 1 and 8 are incomplete; no gain after online frontend overhead is established yet.']
    lines += ['', '**Architecture differences and remaining uncertainty.** The separate ViT and Swin rows above preserve their observed clean/corruption effects, initial removal, hierarchical activity, and online latency. Where matched architecture/seed/cell evidence is missing, a general ranking is unsupported; even complete ranges can favor different methods for different conditions. Native coefficient zeros, removed image patches, executed padding, and latency are distinct measurements. All-three-seed sign agreement is descriptive evidence, not a significance test; differences within timing variability remain uncertain. MNIST grayscale results do not establish color robustness.']
    return lines



def recipe_provenance(root, evidence):
    """Actual saved run recipes take precedence over explicitly planned settings."""
    root = Path(root).resolve()
    manifest_path = root / 'experiment_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pending = manifest.get('batch_selection_required', False) and manifest.get('batch_selection', {}).get('state') != 'verified'
    output = []
    for dataset in ('mnist', 'cifar10'):
        for architecture in ('vit_small', 'swin_tiny'):
            observed = defaultdict(set)
            for row in evidence.get('seeds', []):
                if row.get('dataset') != dataset or row.get('architecture') != architecture or not row.get('run_directory'):
                    continue
                path = Path(row['run_directory']).resolve() / 'config.json'
                if not path.is_relative_to(root):
                    raise ValueError('Recipe evidence belongs to a different campaign root')
                if not path.exists():
                    continue
                config = json.loads(path.read_text())
                if config.get('dataset') != dataset or config.get('architecture') != architecture:
                    raise ValueError('Saved recipe group differs from result evidence')
                recipe = config['recipe']
                batch = recipe.get('batch_size', 8)
                observed[(batch, recipe.get('effective_batch_size', batch), recipe.get('eval_batch_size', 8), recipe.get('lr'))].add(str(path))
            if observed:
                for values, paths in sorted(observed.items(), key=lambda item: str(item[0])):
                    output.append(dict(dataset=dataset, architecture=architecture, train_batch_size=values[0], effective_batch_size=values[1], eval_batch_size=values[2], base_lr=values[3], status=f'saved run configs ({len(paths)} runs)', source=';'.join(sorted(paths))))
                continue
            recipe = dict(manifest.get('recipe', {}))
            recipe.update(manifest.get('recipe_by_group', {}).get(dataset + '/' + architecture, {}))
            frozen = manifest.get('groups', {}).get(dataset + '/' + architecture, {}).get('frozen_recipe')
            if frozen:
                recipe = dict(frozen)
            batch = recipe.get('batch_size') if not pending else None
            output.append(dict(dataset=dataset, architecture=architecture, train_batch_size=batch, effective_batch_size=recipe.get('effective_batch_size', batch) if not pending else None, eval_batch_size=recipe.get('eval_batch_size', 8) if recipe and not pending else None, base_lr=recipe.get('lr'), status='planned; batch selection pending' if pending else 'planned from manifest; no saved run config' if recipe else 'not recorded', source=str(manifest_path)))
    return output


def write_report(root, evidence, conditions, timing_table, normalizations, figures):
    experiment_scope = evidence.get('experiment_scope', read_scope(root))
    expected = expected_main_count(experiment_scope)
    trained = sum(row['training_status'] == 'completed' for row in evidence['seeds'])
    evaluated = sum(row['evaluation_status'] == 'complete' for row in evidence['seeds'])
    lines = ['# Sparse contrast classification benchmark', '', f'Campaign: `{root.name}`. Evidence root: `{root.resolve()}`. Only this campaign\'s manifest and run directories contribute results; prior batch-8 campaign results are not pooled into a larger-batch rerun.', '', 'Both headline tables use protocol `heldout_val`: MNIST Train N=55,000 and CIFAR-10 Train N=45,000, each with 5,000 held-out validation images. Counts in completed-run metadata must match these exact partitions.', '', '| Dataset | Architecture | Condition | Clean test accuracy (%) | Mean corruption error (%) |', '|---|---|---|---|---|']
    for row in conditions:
        lines.append(f'| {row["dataset"]} | {row["architecture"]} | {row["condition"]} | {format_metric(row, "clean_accuracy")} | {format_metric(row, "mCE_raw")} |')
    lines += ['', 'Latency table training protocol: `heldout_val`; MNIST Train N=55,000; CIFAR-10 Train N=45,000.', '', '| Dataset | Architecture | Batch | Scope | Input | Condition | Latency (ms/batch), mean seed median +/- sample SD | Seeds |', '|---|---|---|---|---|---|---|---|']
    headline = [row for row in timing_table if row['scope'] in SCOPES[:2] and row['statistic'] == 'median_ms']
    if headline:
        for row in headline:
            mean, sd = row['mean_seed_statistic_ms'], row['between_seed_sd_ms']
            value = 'not measured' if mean is None else f'{mean:.3f}' + (f' +/- {sd:.3f}' if sd is not None else '')
            lines.append(f'| {row["dataset"]} | {row["architecture"]} | {row["batch_size"]} | {row["scope"]} | {row["input_group"]} | {row["condition"]} | {value} | {row["seeds"]}/3 |')
    else:
        for batch in (1, 8):
            for scope in SCOPES[:2]:
                for dataset in ('mnist', 'cifar10'):
                    lines.append(f'| {dataset} | Both | {batch} | {scope} | Clean and balanced corruption | All {len(configurations(dataset, experiment_scope))} configurations | not measured | 0/3 |')
    lines += ['', '**Training and evaluation recipe provenance.** Batch sizes below come from saved run configurations where available; otherwise the table explicitly marks planned manifest values. The learning rate is the configured base rate before warmup/cosine scheduling. Evaluation batch is for full validation/test/corruption passes; isolated latency remains measured separately at batches 1 and 8.', '', '| Dataset | Architecture | Train batch | Effective batch | Evaluation batch | Base LR | Evidence |', '|---|---|---|---|---|---|---|']
    for row in recipe_provenance(root, evidence):
        show = lambda key: 'not finalized' if row[key] is None else f'{row[key]:g}'
        lines.append(f'| {row["dataset"]} | {row["architecture"]} | {show("train_batch_size")} | {show("effective_batch_size")} | {show("eval_batch_size")} | {show("base_lr")} | {row["status"]} |')
    lines += ['', f'Primary campaign coverage: {trained}/{expected} final checkpoints completed; {evaluated}/{expected} fully evaluated. Manifest state: `{evidence["manifest_state"]}`. Missing runs, cells, and seed results remain in `outputs/tables/run_coverage.csv`; they are never removed from a denominator. Pilots and optional refits are excluded from these primary aggregates.', '',
              'Clean evaluation requires all 10,000 official test images. Each MNIST-C checkpoint requires 15 fixed-severity cells and 150,000 predictions; each CIFAR-10-C checkpoint requires 75 cells and 750,000 predictions. The reported mean corruption error is the unnormalized macro-average across the 15 corruption means. CIFAR-10-C averages all five severities within each corruption first. Aggregate uncertainty is sample SD across independently trained seeds. Repeated batches and corruption cells are not training seeds.', '',
              'The primary protocol uses 55,000 MNIST or 45,000 CIFAR-10 training examples and 5,000 genuinely held-out validation examples. Training curves distinguish online augmented training from unaugmented held-out validation. Primary results use the final epoch, not test-selected or validation-selected checkpoints.', '',
              'Per-corruption and severity evidence: `outputs/tables/cells.csv`. Prediction identities, labels, and correctness are stored beside each run\'s evaluation JSON. Paired percentage-point changes against the matching raw seed are in `outputs/tables/paired_changes.csv`. A missing corruption/severity cell prevents a completed mean corruption result.', '',
              'Latency records and stage evidence: `outputs/tables/timings.csv`, `outputs/tables/stages.csv`, and the originating run directories recorded there. Primary online scopes include the frontend, coefficient sorting when nonzero sparsity is requested, support, normalization, compaction, and the model. Host-input timing additionally includes H2D from pinned decoded input. Disk/decoding and CPU prediction return are excluded from those two scopes and reported only in the practical loader scope. Profiler intervals are shown separately and are never added into an invented total. Hardware types are never pooled. Within one matched hardware class, resumed cells may span physical GPUs or measurement sessions; per-cell device UUID, session ID, host, scheduler job, and immutable hardware-metadata hashes are retained in the timing/stage tables, with explicit session/device unions for balanced mixtures. Batch latency divided by batch size is an amortized cost, not a batch-one request measurement.', '',
              '| Dataset | Representation | Channel | Mean | Measured std | Effective std | Guarded |', '|---|---|---|---|---|---|---|']
    for row in normalizations:
        lines.append(f'| {row["dataset"]} | {row["representation"]} | {row["channel"]} | {row["mean"]:.7g} | {row["measured_std"]:.7g} | {row["effective_std"]:.7g} | {row["guarded"]} |')
    if not normalizations:
        lines.append('| All | All | All | not fitted | not fitted | not fitted | pending |')
    lines += ['', 'Raw-input constants are the requested published dataset-specific recipes. Contrast means and population standard deviations are fitted at 0% sparsity to the permitted clean training partition only, then frozen for every sparsity, seed, architecture, and evaluation cell. Evaluation distributions never update those constants. The active dataset-specific representation scope is recorded in `experiment_scope.json`; withdrawn MNIST color encodings and their outputs are excluded. MNIST already has sparse raw backgrounds; absent a raw-empty-patch control, a MNIST speedup cannot be attributed uniquely to contrast. These are small-image ViT-Small and Swin-Tiny adaptations with native 28/32-pixel inputs and 2x2 patches, not ImageNet configuration reproductions or a state-of-the-art comparison.', '',
              'Scientific figures and their CSV data are under `outputs/figures/`. Shared signed-input panels, fixed training-derived display scales, and exact NPZ tensors/masks/IDs are under `outputs/previews/`. Rendering never changes model tensors. Per-run raw/augmented prediction panels and released-corruption panels are under each run\'s `diagnostics/` directory.', '',
              'TensorBoard event files: `outputs/tensorboard/`. Start with `tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006` in the project environment; access the host through the site-approved tunnel. Study figures are mirrored under `outputs/tensorboard/study_summary/heldout_val/<dataset>/<architecture>/report/`. Per-run paths distinguish protocol, representation, compact/dense execution, sparsity, seed, and configuration hash.', '']
    estimate_path = root / 'outputs' / 'runtime_estimate.json'
    if estimate_path.exists():
        estimate = json.loads(estimate_path.read_text())
        total = estimate.get('all_training_reference_gpu_hours', estimate.get('all_232_training_reference_gpu_hours') if expected == 228 else None)
        if estimate.get('expected_main_runs') != expected:
            total = None
        if estimate.get('state') == 'provisional_complete_group_coverage' and total is not None:
            lines += [f'[Training-only runtime estimate](outputs/runtime_estimate.json): {float(total):.1f} reference GPU-hours for {expected} main runs plus four pilots, projected from observed raw-pilot epochs on the recorded GPU classes. This is a provisional training reference, not a whole-campaign completion forecast; contrast-specific costs, checkpoint/setup, evaluation, latency benchmarking, and queue/resource availability are excluded.']
        else:
            missing = 'an estimate matching the active experiment scope' if estimate.get('expected_main_runs') != expected else ', '.join(estimate.get('missing_groups', [])) or 'pilot group measurements'
            lines += [f'[Training-only runtime estimate](outputs/runtime_estimate.json) is incomplete; missing committed pilot-epoch measurements: {missing}. No full-campaign GPU-hour or elapsed-time total is inferred from partial coverage.']
        lines += ['']
    else:
        lines += ['The training-only runtime estimate is not yet available; committed pilot epochs are required before projecting GPU-hours.', '']
    lines += research_interpretation(evidence)
    if figures:
        lines += ['', 'Generated figure index:']
        lines += [f'- [{Path(path).stem}]({Path(path).relative_to(root)})' for path in figures]
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')



def log_saved_summary(writer, condition_rows, seed_rows, paired_rows, root):
    """Mirror saved tables and resolved configs without inventing missing seeds."""
    for row in condition_rows:
        representation, execution, percent = _condition(row)
        prefix = f'conditions/{representation}/{execution}/p{percent if percent is not None else "NA"}'
        for metric in ('clean_accuracy', 'clean_error', 'mCE_raw'):
            for summary in ('mean', 'sd', 'seeds'):
                value = _float(row.get(f'{metric}_{summary}'))
                if value is not None:
                    writer.add_scalar(f'{prefix}/{metric}/{summary}', value, 0)
    paired_groups = defaultdict(list)
    for row in paired_rows:
        representation, execution, percent = _condition(row)
        prefix = f'paired/{representation}/{execution}/p{percent if percent is not None else "NA"}/{row["metric"]}'
        value = float(row['method_minus_baseline_percentage_points'])
        writer.add_scalar(f'{prefix}/seed{row["seed"]}_percentage_points', value, 0)
        paired_groups[prefix].append(value)
    for prefix, values in paired_groups.items():
        mean, sd = mean_sd(values)
        writer.add_scalar(prefix + '/mean_percentage_points', mean, 0)
        writer.add_scalar(prefix + '/seeds', len(values), 0)
        if sd is not None:
            writer.add_scalar(prefix + '/sample_sd_percentage_points', sd, 0)
    for row in seed_rows:
        metrics = {f'final/{name}': float(row[name]) for name in ('clean_accuracy', 'clean_error', 'mCE_raw') if _float(row.get(name)) is not None}
        if not metrics:
            continue
        run_path = Path(row['run_directory'])
        config = json.loads((run_path / 'config.json').read_text())
        hparams = {key: config[key] for key in ('dataset', 'architecture', 'representation', 'execution', 'seed', 'protocol', 'precision', 'normalization_hash', 'config_hash')}
        hparams['sparsity_percent'] = config['sparsity_percent'] if config['sparsity_percent'] is not None else 'NA'
        for key, value in config['recipe'].items():
            if isinstance(value, (bool, int, float, str)):
                hparams['recipe/' + key] = value
            else:
                hparams['recipe/' + key] = json.dumps(value, sort_keys=True)
        writer.add_hparams(hparams, metrics, run_name='hparams/' + config['config_hash'], global_step=0)
    writer.flush()

def generate(root=ROOT, plots=True):
    root = Path(root)
    evidence = collect_evidence(root)
    experiment_scope = evidence['experiment_scope']
    expected = expected_main_count(experiment_scope)
    tables = root / 'outputs' / 'tables'; tables.mkdir(parents=True, exist_ok=True)
    conditions = aggregate_conditions(evidence['seeds'], experiment_scope)
    save_csv(tables / 'run_coverage.csv', evidence['seeds']); save_csv(tables / 'conditions.csv', conditions)
    for key in ('cells', 'curves', 'tokens', 'timings', 'stages'):
        save_csv(tables / f'{key}.csv', evidence[key])
    save_csv(tables / 'paired_changes.csv', paired_changes(evidence['seeds']))
    normals = normalization_rows(root); save_csv(tables / 'normalization.csv', normals)
    figures, timing_table = [], []
    saved_conditions = read_csv(tables / 'conditions.csv')
    saved_seeds = read_csv(tables / 'run_coverage.csv')
    saved_pairs = read_csv(tables / 'paired_changes.csv')
    for dataset in ('mnist', 'cifar10'):
        for architecture in ('vit_small', 'swin_tiny'):
            choose = lambda rows: [row for row in rows if row.get('dataset') == dataset and row.get('architecture') == architecture]
            with SummaryWriter(str(root / 'outputs' / 'tensorboard' / 'study_summary' / 'heldout_val' / dataset / architecture / 'report')) as writer:
                log_saved_summary(writer, choose(saved_conditions), choose(saved_seeds), choose(saved_pairs), root)
                writer.add_scalar('coverage/completed_primary_seeds', sum(int(row['fully_evaluated_seeds']) for row in choose(saved_conditions)), 0)
    # Every plot consumes the saved table again, never hardcoded benchmark values.
    if plots:
        figures += preprocessing_figures(root)
        sources = {key: read_csv(tables / f'{key}.csv') for key in ('conditions', 'cells', 'curves', 'tokens', 'timings', 'stages')}
        for dataset in ('mnist', 'cifar10'):
            for architecture in ('vit_small', 'swin_tiny'):
                selected = {key: [row for row in rows if row.get('dataset') == dataset and row.get('architecture') == architecture] for key, rows in sources.items()}
                output = root / 'outputs' / 'figures' / dataset / architecture
                with SummaryWriter(str(root / 'outputs' / 'tensorboard' / 'study_summary' / 'heldout_val' / dataset / architecture / 'report')) as writer:
                    figures += metric_curve(selected['conditions'], 'clean_accuracy', 'Clean test accuracy (%)', output / 'clean_accuracy', writer)
                    figures += metric_curve(selected['conditions'], 'mCE_raw', 'Mean corruption error (%)', output / 'mean_corruption_error', writer)
                    figures += corruption_plots(selected['cells'], output, writer, experiment_scope)
                    figures += learning_curves(selected['curves'], output, writer, experiment_scope)
                    figures += token_plots(selected['tokens'], output, writer)
                    images, rows = timing_overviews(selected['timings'], output, writer, experiment_scope)
                    figures += images; timing_table += [{**row, 'dataset': dataset, 'architecture': architecture} for row in rows]
                    figures += timing_comparisons(selected['timings'], selected['conditions'], output, writer, experiment_scope)
                    figures += stage_plots(selected['stages'], output, writer, experiment_scope)
                    writer.add_scalar('coverage/completed_primary_seeds', sum(int(row['fully_evaluated_seeds']) for row in selected['conditions']), 0)
    save_csv(tables / 'latency_overview.csv', timing_table)
    atomic_json(tables / 'coverage.json', {'expected_runs': expected, 'trained_runs': sum(row['training_status'] == 'completed' for row in evidence['seeds']), 'evaluated_runs': sum(row['evaluation_status'] == 'complete' for row in evidence['seeds']), 'figures': figures, 'benchmark_coverage': benchmark_coverage(evidence), 'all_requested_work_complete': all(row['evaluation_status'] == 'complete' for row in evidence['seeds']) and benchmark_coverage(evidence)['complete_on_one_matched_hardware']})
    write_report(root, evidence, conditions, timing_table, normals, figures)
    return {'expected_runs': expected, 'completed_evaluations': sum(row['evaluation_status'] == 'complete' for row in evidence['seeds']), 'figures': len(figures), 'report': str(root / 'REPORT.md')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.root, not args.no_plots), indent=2))


if __name__ == '__main__':
    main()
