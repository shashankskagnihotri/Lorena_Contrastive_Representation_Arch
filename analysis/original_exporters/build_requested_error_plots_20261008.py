#!/usr/bin/env python3
"""Render requested corruption facets from complete saved accuracy evidence only."""
import ast
import csv
import datetime
import hashlib
import io
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'sparse_contrast_benchmark_large_batch'
NEW = PROJECT / 'training_dense_testing_sparse'
OUT = PROJECT / 'plots'
DETAIL = OUT / 'requested_comparisons'
BASE_SOURCE = '7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'
COLORS = {'raw': '#000000', 'color_opponency': '#7B3294', 'single_color': '#E69F00', 'grayscale': '#0072B2'}
LABELS = {'raw': 'Raw baseline', 'color_opponency': 'Color opponency', 'single_color': 'Single color', 'grayscale': 'Grayscale contrast'}
ARCHITECTURES = ('vit_small', 'swin_tiny')
ARCH_LABELS = {'vit_small': 'ViT-Small', 'swin_tiny': 'Swin-Tiny'}
SPARSE_LEVELS = (0, 20, 40, 60, 80)
DENSE_LEVELS = tuple(range(0, 91, 10))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path, provenance, kind='json'):
    data = Path(path).read_bytes()
    provenance['inputs'].append({'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(data).hexdigest()})
    text = data.decode()
    return list(csv.DictReader(io.StringIO(text))) if kind == 'csv' else json.loads(text) if kind == 'json' else text


def write_csv(path, rows):
    assert rows
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def corruption_order(provenance):
    text = read(BASE / 'outputs/source' / BASE_SOURCE / 'sparse_contrast/data.py', provenance, 'text')
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('MNIST_CORRUPTIONS', 'CIFAR10_CORRUPTIONS'):
                    values[target.id] = ast.literal_eval(node.value)
    assert all(len(v) == 15 for v in values.values()) and len(values) == 2
    return {'mnist': ('clean', *values['MNIST_CORRUPTIONS']), 'cifar10': ('clean', *values['CIFAR10_CORRUPTIONS'])}


def expected_cells(dataset, order):
    return {('clean', 'test')} | {(c, str(s)) for c in order[dataset][1:]
                                 for s in (('fixed',) if dataset == 'mnist' else range(1, 6))}


def validate_cells(rows, dataset, order):
    actual = [(r['corruption'], r['severity']) for r in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected_cells(dataset, order)
    for row in rows:
        assert int(row['total']) == 10000
        accuracy = int(row['correct']) / int(row['total'])
        assert math.isclose(accuracy, float(row['accuracy']), abs_tol=1e-12)
        assert math.isclose(1 - accuracy, float(row['error']), abs_tol=1e-12)


def collapse(rows, metadata, order):
    result = []
    for corruption in order[metadata['dataset']]:
        cells = [r for r in rows if r['corruption'] == corruption]
        severities = [r['severity'] for r in cells]
        expected = ['test'] if corruption == 'clean' else ['fixed'] if metadata['dataset'] == 'mnist' else ['1', '2', '3', '4', '5']
        assert sorted(severities) == sorted(expected)
        # Average severity errors within each seed before calculating seed SD.
        error = statistics.mean(100 * (1 - int(r['correct']) / int(r['total'])) for r in cells)
        result.append(dict(metadata, corruption=corruption, severity_aggregation='clean' if corruption == 'clean' else
                           'fixed' if metadata['dataset'] == 'mnist' else 'equal_mean_of_1_2_3_4_5',
                           severity_count=len(cells), test_images_per_severity=10000, error_percent=error))
    return result


def collect(provenance, order):
    parent_coverage = read(BASE / 'outputs/tables/coverage.json', provenance)
    coverage = read(NEW / 'outputs/tables/coverage.json', provenance)
    manifest = read(NEW / 'study_manifest.json', provenance)
    assert parent_coverage['evaluated_runs'] == parent_coverage['expected_runs'] == 156
    assert coverage['accuracy_complete'] and coverage['completed_evaluation_cases'] == 612
    assert coverage['completed_evaluation_cells'] == 35352
    assert digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}) == manifest['manifest_hash'] == coverage['manifest_hash']
    cases = {c['test_config_hash']: c for c in manifest['cases']}
    runs = read(BASE / 'outputs/tables/run_coverage.csv', provenance, 'csv')
    parent_cells = read(BASE / 'outputs/tables/cells.csv', provenance, 'csv')
    new_runs = read(NEW / 'outputs/tables/accuracy_seeds.csv', provenance, 'csv')
    new_cells = read(NEW / 'outputs/tables/evaluation_cells.csv', provenance, 'csv')
    assert len(runs) == 156 and len(parent_cells) == 9336
    assert len(new_runs) == 612 and len(new_cells) == 35352
    assert {r['test_config_hash'] for r in new_runs} == set(cases)
    grouped_parent, grouped_new = defaultdict(list), defaultdict(list)
    for row in parent_cells:
        grouped_parent[row['registry_id']].append(row)
    for row in new_cells:
        grouped_new[row['test_config_hash']].append(row)
    assert len(grouped_parent) == 156 and set(grouped_new) == set(cases)
    selected, raw_cells, raw_checkpoints = [], {}, {}
    for run in runs:
        ds, rep, execution = run['dataset'], run['representation'], run['execution']
        assert run['training_status'] == 'completed' and run['evaluation_status'] == 'complete'
        cells = grouped_parent[run['registry_id']]
        validate_cells(cells, ds, order)
        assert all(all(r[k] == run[k] for k in ('dataset', 'architecture', 'representation', 'execution', 'sparsity_percent', 'seed', 'protocol', 'training_count')) for r in cells)
        if rep == 'raw':
            assert execution == 'dense'
            family, level = 'raw_dense_reference', 0
            key = (ds, run['architecture'], int(run['seed']))
            raw_cells[key] = {(r['corruption'], r['severity']): int(r['correct']) for r in cells}
            raw_checkpoints[key] = run['checkpoint_sha256']
        elif execution == 'compact':
            family, level = 'training_sparse_testing_sparse', int(run['sparsity_percent'])
            assert level in SPARSE_LEVELS
        else:
            continue
        selected += collapse(cells, dict(dataset=ds, architecture=run['architecture'], representation=rep,
            family=family, trained_execution=execution, inference_execution=execution, sparsity_percent=level,
            seed=int(run['seed']), protocol=run['protocol'], training_count=int(run['training_count']),
            source_id=run['registry_id'], checkpoint_sha256=run['checkpoint_sha256'],
            training_config_hash=Path(run['run_directory']).name), order)
    for run in new_runs:
        case = cases[run['test_config_hash']]
        assert run['state'] == 'complete'
        assert int(run['completed_cells']) == int(run['expected_cells']) == len(expected_cells(case['dataset'], order))
        for field in ('checkpoint_sha256', 'training_config_hash', 'source_hash', 'normalization_hash'):
            assert run[field] == case[field]
        cells = grouped_new[case['test_config_hash']]
        validate_cells(cells, case['dataset'], order)
        for row in cells:
            assert all(row[k] == run[k] for k in ('dataset', 'architecture', 'representation', 'trained_execution',
                'inference_execution', 'sparsification_domain', 'inference_sparsity_percent', 'seed',
                'test_config_hash', 'training_config_hash', 'checkpoint_sha256', 'normalization_hash', 'source_hash', 'protocol', 'training_count'))
        if case['representation'] == 'raw' and case['inference_execution'] == 'dense':
            key = (case['dataset'], case['architecture'], case['seed'])
            assert case['checkpoint_sha256'] == raw_checkpoints[key]
            assert {(r['corruption'], r['severity']): int(r['correct']) for r in cells} == raw_cells[key]
        if case['representation'] == 'raw' or case['trained_execution'] != 'dense' or case['inference_execution'] != 'compact':
            continue
        selected += collapse(cells, dict(dataset=case['dataset'], architecture=case['architecture'],
            representation=case['representation'], family='training_dense_testing_sparse',
            trained_execution='dense', inference_execution='compact', sparsity_percent=case['inference_sparsity_percent'],
            seed=case['seed'], protocol=run['protocol'], training_count=int(run['training_count']),
            source_id=case['test_config_hash'], checkpoint_sha256=case['checkpoint_sha256'],
            training_config_hash=case['training_config_hash']), order)
    assert len(selected) == 5952
    provenance['accuracy_coverage'] = {'parent_runs': 156, 'parent_cells': 9336, 'followup_cases': 612,
        'followup_cells': 35352, 'selected_parent_runs': 132, 'selected_followup_cases': 240,
        'selected_seed_corruption_rows': len(selected), 'raw_reference_exact_cross_study_checks': 12}
    return selected


def aggregate(seed_rows, order):
    fields = ('dataset', 'architecture', 'representation', 'family', 'trained_execution', 'inference_execution',
              'sparsity_percent', 'protocol', 'training_count', 'corruption', 'severity_aggregation', 'severity_count', 'test_images_per_severity')
    groups = defaultdict(list)
    for row in seed_rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r['seed'])
        assert [r['seed'] for r in group] == [0, 1, 2]
        errors = [r['error_percent'] for r in group]
        result.append(dict(zip(fields, key), error_percent_mean=statistics.mean(errors),
            error_percent_sample_sd=statistics.stdev(errors), seeds=3,
            **{f'error_percent_seed{i}': errors[i] for i in range(3)}))
    for ds in ('mnist', 'cifar10'):
        reps = ('grayscale',) if ds == 'mnist' else ('color_opponency', 'single_color', 'grayscale')
        for arch in ARCHITECTURES:
            for corruption in order[ds]:
                panel = [r for r in result if r['dataset'] == ds and r['architecture'] == arch and r['corruption'] == corruption]
                assert len([r for r in panel if r['representation'] == 'raw']) == 1
                assert {r['representation'] for r in panel} == {'raw', *reps}
                for rep in reps:
                    for family, levels in [('training_sparse_testing_sparse', SPARSE_LEVELS), ('training_dense_testing_sparse', DENSE_LEVELS)]:
                        found = [r for r in panel if r['representation'] == rep and r['family'] == family]
                        assert sorted(r['sparsity_percent'] for r in found) == list(levels)
    assert len(result) == 1984
    return result


def render(rows, seed_rows, provenance, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import seaborn as sns
    sns.set_theme(context='talk', style='whitegrid', font='serif', font_scale=1.0)
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    outputs = []
    for ds, name in [('mnist', 'MNIST'), ('cifar10', 'CIFAR10')]:
        stem = name + '_error_by_corruption'
        chosen = [r for r in rows if r['dataset'] == ds]
        write_csv(OUT / (stem + '.csv'), chosen)
        write_csv(DETAIL / (stem + '_seeds.csv'), [r for r in seed_rows if r['dataset'] == ds])
        outputs.extend([OUT / (stem + '.csv'), DETAIL / (stem + '_seeds.csv')])
        reps = ('grayscale',) if ds == 'mnist' else ('color_opponency', 'single_color', 'grayscale')
        pdf = OUT / (stem + '.pdf')
        with PdfPages(pdf, metadata={'Title': name + ' error by corruption', 'Subject': 'Complete three-seed accuracy evidence; compact0-trained follow-up excluded'}) as pages:
            for arch in ARCHITECTURES:
                fig, axes = plt.subplots(4, 4, figsize=(22, 18.6), sharex=True, sharey=True)
                for ax, corruption in zip(axes.flat, order[ds]):
                    panel = [r for r in chosen if r['architecture'] == arch and r['corruption'] == corruption]
                    baseline = next(r for r in panel if r['representation'] == 'raw')
                    mean, sd = baseline['error_percent_mean'], baseline['error_percent_sample_sd']
                    ax.axhspan(max(0, mean - sd), min(100, mean + sd), color=COLORS['raw'], alpha=.08, linewidth=0)
                    ax.axhline(mean, color=COLORS['raw'], linewidth=2, zorder=5)
                    for rep in reps:
                        for family, style, marker in [('training_sparse_testing_sparse', '-', 'o'), ('training_dense_testing_sparse', '--', 's')]:
                            points = sorted((r for r in panel if r['representation'] == rep and r['family'] == family), key=lambda r: r['sparsity_percent'])
                            x = [r['sparsity_percent'] for r in points]
                            y = [r['error_percent_mean'] for r in points]
                            sd = [r['error_percent_sample_sd'] for r in points]
                            ax.fill_between(x, [max(0, a-b) for a,b in zip(y,sd)], [min(100, a+b) for a,b in zip(y,sd)], color=COLORS[rep], alpha=.09, linewidth=0)
                            ax.plot(x, y, color=COLORS[rep], linestyle=style, marker=marker, markersize=4.5,
                                    linewidth=2.1, markerfacecolor=COLORS[rep] if style == '-' else 'white', markeredgewidth=1.2)
                    title = 'Clean (IID)' if corruption == 'clean' else corruption.replace('_', ' ').capitalize()
                    ax.set_title(title, fontsize=17, pad=10)
                    ax.set_xlim(0, 90); ax.set_ylim(0, 100)
                    ax.set_xticks([0, 20, 40, 60, 80, 90]); ax.set_yticks([0, 20, 40, 60, 80, 100])
                    ax.tick_params(labelsize=13)
                    ax.grid(alpha=.28)
                name_label = 'CIFAR-10' if ds == 'cifar10' else 'MNIST'
                fig.suptitle(f'{name_label} | {ARCH_LABELS[arch]} | classification error by corruption', fontsize=25, y=.985)
                severity = 'Each corruption: mean of severities 1–5 within each seed' if ds == 'cifar10' else 'Each MNIST-C corruption: its official fixed severity'
                train_n = '45,000' if ds == 'cifar10' else '55,000'
                fig.text(.5, .950, f'{severity} | heldout_val | training N = {train_n}', ha='center', fontsize=16)
                handles = [Line2D([0],[0],color=COLORS['raw'],linewidth=2,label='Grayscale baseline' if ds == 'mnist' else 'RGB baseline')]
                handles += [Line2D([0],[0],color=COLORS[rep],linewidth=3,label=LABELS[rep]) for rep in reps]
                handles += [Line2D([0],[0],color='#444444',linestyle='-',marker='o',label='Train sparse; test at same sparsity'),
                            Line2D([0],[0],color='#444444',linestyle='--',marker='s',markerfacecolor='white',label='Train dense; test sparse')]
                fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.932), ncol=3 if ds == 'cifar10' else 4,
                           fontsize=14, frameon=False, columnspacing=1.5, handlelength=3)
                fig.supxlabel('Imposed inference sparsity (%)', fontsize=20, y=.036)
                fig.supylabel('Classification error (%)', fontsize=20, x=.013)
                fig.text(.5,.010,'Lines: mean over seeds 0, 1, 2; bands: ±1 sample SD. Compact inference at 0% can remove naturally empty patches.', ha='center',fontsize=13)
                fig.subplots_adjust(left=.060,right=.985,bottom=.085,top=.852,hspace=.30,wspace=.12)
                pages.savefig(fig, dpi=300)
                png = DETAIL / f'{stem}_{arch}.png'
                svg = DETAIL / f'{stem}_{arch}.svg'
                fig.savefig(png, dpi=300)
                fig.savefig(svg, dpi=300)
                plt.close(fig)
                outputs.extend([png, svg])
        outputs.append(pdf)
    provenance['outputs'] = [{'path': str(p.relative_to(PROJECT)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in outputs]
    return outputs


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Render these publication plots in an allocated CPU job')
    DETAIL.mkdir(parents=True, exist_ok=True)
    provenance = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_id': os.environ['SLURM_JOB_ID'], 'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'inputs': [], 'colors': COLORS, 'protocol': 'heldout_val', 'seed_ids': [0,1,2],
        'solid': 'training_sparse_testing_sparse; compact trained and tested at the same imposed sparsity 0/20/40/60/80',
        'dashed': 'training_dense_testing_sparse; trained_execution=dense and inference_execution=compact; 0..90 in steps of10',
        'raw': 'Original unchanged dense raw baseline; exact correctness counts/checkpoints cross-checked against follow-up dense raw references',
        'uncertainty': 'Sample SD across three seeds after per-seed equal averaging of five CIFAR-C severities; fixed MNIST-C severity',
        'limitations': 'Derived from saved complete report cell tables with exact file hashes and registry/coverage checks. No new inference, checkpoint loading, prediction-file rehashing, or latency measurement.'}
    order = corruption_order(provenance)
    seed_rows = collect(provenance, order)
    rows = aggregate(seed_rows, order)
    provenance['corruption_order'] = order
    provenance['aggregate_rows'] = len(rows)
    render(rows, seed_rows, provenance, order)
    path = DETAIL / 'error_plots_provenance.json'
    path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'state':'complete','provenance':str(path),'seed_rows':len(seed_rows),'aggregate_rows':len(rows),'pdf_pages':4}), flush=True)


if __name__ == '__main__':
    main()
