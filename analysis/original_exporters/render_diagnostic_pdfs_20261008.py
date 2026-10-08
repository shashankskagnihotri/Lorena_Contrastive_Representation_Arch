#!/usr/bin/env python3
"""Publish saved diagnostic tensors; never load models, datasets or checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

PROJECT = Path('/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch')
BASE_HASH = '7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'
ARRAY_KEYS = ('raw', 'native_0', 'native_sparse', 'normalized_0', 'normalized_sparse',
              'coefficient_support', 'patch_support', 'execution_patch_support',
              'labels', 'sample_ids', 'native_scales', 'normalized_scales', 'mean', 'std')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_dir(study, case):
    return (study / 'outputs/cases' / case['dataset'] / case['architecture'] /
            case['representation'] / ('trained_' + case['trained_execution']) /
            ('infer_' + case['inference_execution']) /
            ('p' + str(case['inference_sparsity_percent'])) / ('seed' + str(case['seed'])) /
            case['test_config_hash'])


def cells(dataset):
    if dataset == 'cifar10':
        return [('clean', 'test'), ('brightness', 3), ('contrast', 3), ('fog', 3)]
    return [('clean', 'test'), ('brightness', 'fixed'), ('fog', 'fixed')]


def renderer(project):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ['SPARSE_CONTRAST_ROOT'] = str(project / 'sparse_contrast_benchmark_large_batch')
    frozen = project / 'sparse_contrast_benchmark_large_batch/outputs/source' / BASE_HASH
    sys.path.insert(0, str(frozen))
    import torch
    torch.set_num_threads(1)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context='talk', style='whitegrid', font='serif', font_scale=1.0)
    from sparse_contrast.visualize import signed_panel_figure
    return plt, signed_panel_figure


def load_panel(path, case):
    import numpy as np
    record = read(path)
    tensor_path = path.with_suffix('.npz')
    if record['state'] != 'complete' or record['tensor_sha256'] != sha(tensor_path):
        raise ValueError(f'Invalid diagnostic receipt: {path}')
    identity = record['panel_identity']['identity']
    for key in ('test_config_hash', 'training_config_hash', 'checkpoint_sha256', 'normalization_hash',
                'dataset', 'architecture', 'representation', 'seed', 'trained_execution',
                'inference_execution', 'inference_sparsity_percent'):
        if identity[key] != case[key]:
            raise ValueError(f'Diagnostic case identity mismatch {key}: {path}')
    with np.load(tensor_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ARRAY_KEYS}
    if arrays['sample_ids'].tolist() != record['panel_identity']['sample_ids']:
        raise ValueError(f'Diagnostic sample identity mismatch: {path}')
    if len(arrays['labels']) != 10 or set(arrays['labels'].tolist()) != set(range(10)):
        raise ValueError(f'Expected the original class-balanced ten-image panel: {path}')
    for key, value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite saved {key}: {path}')
    return record, arrays


def render_task(task):
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    project, output = Path(task['project']), Path(task['output'])
    plt, panel_figure = renderer(project)
    pairs = []
    for case, source in zip(task['cases'], task['sources']):
        record, arrays = load_panel(Path(source), case)
        pairs.append((case, Path(source), record, arrays))
    identical = all(np.array_equal(pairs[0][3][key], pairs[1][3][key]) for key in ARRAY_KEYS)
    to_render = pairs[:1] if identical else pairs
    products = []
    for case, source, record, arrays in to_render:
        arch_dir = 'shared_frontend' if identical else case['architecture']
        stem = (output / 'followup_fixed_examples' / case['dataset'] / case['representation'] /
                ('p' + str(case['inference_sparsity_percent'])) / arch_dir /
                f"{task['corruption']}__{task['severity']}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        pdf, png, provenance = (stem.with_suffix(ext) for ext in ('.pdf', '.png', '.json'))
        signature = dict(renderer_sha256=task['script_hash'], selected_source=str(source),
                         selected_source_sha256=sha(source), tensor_sha256=record['tensor_sha256'],
                         counterpart_source=task['sources'][1],
                         counterpart_tensor_sha256=pairs[1][2]['tensor_sha256'],
                         architectures_exactly_equal=identical)
        if provenance.exists():
            saved = read(provenance)
            if saved.get('signature') != signature or sha(pdf) != saved['pdf_sha256'] or sha(png) != saved['png_sha256']:
                raise ValueError(f'Existing diagnostic export changed: {provenance}')
            products.append(saved)
            continue
        names = record['panel_identity']['display_scales']['channel_names']
        rows, pages = [], 0
        with PdfPages(pdf, metadata={'Title': 'Saved fixed frontend diagnostic',
                                    'Subject': 'Original saved tensors; no inference or fitting',
                                    'Creator': 'matplotlib; immutable saved-tensor exporter'}) as document:
            for channel, name in enumerate(names):
                for first in (0, 5):
                    fig = panel_figure(arrays, channel, (arrays['native_scales'], arrays['normalized_scales']),
                                       arrays['labels'], arrays['sample_ids'], case['inference_sparsity_percent'], name, first)
                    title = (f"{case['dataset']} | {case['representation']} | p{case['inference_sparsity_percent']} | "
                             f"{task['corruption']} {task['severity']}\n{name}; hatched = absent patch")
                    fig.suptitle(title, fontsize=13)
                    document.savefig(fig, dpi=300, bbox_inches='tight')
                    if pages == 0:
                        fig.savefig(png, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    pages += 1
                    for sample in range(first, first + 5):
                        rows.append(dict(pdf_page=pages, channel=channel, channel_name=name,
                                         sample_id=str(arrays['sample_ids'][sample]), label=int(arrays['labels'][sample]),
                                         source_npz=str(source.with_suffix('.npz')), source_npz_sha256=record['tensor_sha256'],
                                         native_scale=float(arrays['native_scales'][channel]),
                                         normalized_scale=float(arrays['normalized_scales'][channel]),
                                         coefficient_nonzero_fraction=float(arrays['coefficient_support'][sample, channel].mean()),
                                         retained_patch_fraction=float(arrays['patch_support'][sample].mean())))
        write_csv(stem.with_suffix('.csv'), rows)
        saved = dict(signature=signature, pdf=str(pdf), png_preview=str(png), pages=pages,
                     pdf_sha256=sha(pdf), png_sha256=sha(png), plot_data_csv=str(stem.with_suffix('.csv')),
                     tensor_path=str(source.with_suffix('.npz')), original_case=case,
                     sample_ids=arrays['sample_ids'].tolist(),
                     architecture_equivalence_keys=list(ARRAY_KEYS),
                     omitted_architecture='vit_small' if identical else None,
                     prediction_scope='Frontend tensors only. No predictions or model-dependent activations displayed.',
                     preview_scope='PNG previews the first PDF page; PDF contains all ten images and every channel.')
        write_json(provenance, saved)
        products.append(saved)
    return dict(task=task['key'], architectures_exactly_equal=identical, products=products)


def normalization_and_parent(project, output):
    import numpy as np
    plt, _ = renderer(project)
    parent = project / 'sparse_contrast_benchmark_large_batch'
    preview = parent / 'outputs/previews/heldout_val'
    copied, generated = [], []
    for complete in sorted(preview.glob('*/*/*/complete.json')):
        meta = read(complete)
        target = output / 'parent_fixed_train_val_examples' / meta['dataset'] / meta['representation']
        target.mkdir(parents=True, exist_ok=True)
        refs = []
        for source in sorted(complete.parent.iterdir()):
            if source.suffix not in ('.png', '.pdf', '.npz', '.json'):
                continue
            destination = target / source.name
            if destination.exists() and sha(destination) != sha(source):
                raise ValueError(f'Existing copied diagnostic changed: {destination}')
            if not destination.exists():
                shutil.copyfile(source, destination)
            refs.append(dict(source=str(source), published=str(destination), sha256=sha(source)))
            if source.suffix == '.pdf':
                copied.append(str(destination))
        write_csv(target / 'source_catalog.csv', refs)
        arrays = {}
        for percent in (0, 60):
            with np.load(complete.parent / f'train_p{percent}.npz', allow_pickle=False) as archive:
                arrays[percent] = {k: archive[k] for k in archive.files}
        for channel, name in enumerate(meta['channel_names']):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
            histogram_rows = []
            for row, percent in enumerate((0, 60)):
                for column, domain in enumerate(('native', 'normalized')):
                    values = arrays[percent][domain + '_sparse'][:, channel].ravel()
                    scale = meta[domain + '_scales'][channel]
                    edges = np.linspace(-scale, scale, 81)
                    counts, edges = np.histogram(values, edges)
                    if int(counts.sum()) != values.size:
                        raise ValueError('Saved clean-training display range excludes diagnostic values')
                    axes[row, column].stairs(counts / values.size, edges, fill=False, linewidth=1.5)
                    axes[row, column].set(xlabel=f'{domain.capitalize()} coefficient', ylabel='Fraction per bin',
                                         title=f'{domain.capitalize()}, imposed sparsity {percent}%')
                    for index, count in enumerate(counts):
                        histogram_rows.append(dict(dataset=meta['dataset'], representation=meta['representation'],
                                                   channel=name, sparsity_percent=percent, domain=domain,
                                                   bin_left=edges[index], bin_right=edges[index + 1], count=int(count),
                                                   coefficient_count=int(values.size), fraction=count / values.size,
                                                   sample_count=len(arrays[percent]['labels']),
                                                   source_npz=str(complete.parent / f'train_p{percent}.npz')))
            fig.suptitle(f"{meta['dataset']} | {name}\nFixed ten-image clean-training diagnostic panel", fontsize=14)
            stem = output / 'normalization' / f"{meta['dataset']}_{meta['representation']}_channel{channel}_distributions"
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in ('pdf', 'svg', 'png'):
                fig.savefig(stem.with_suffix('.' + ext), dpi=300, bbox_inches='tight')
            plt.close(fig)
            write_csv(stem.with_suffix('.csv'), histogram_rows)
            generated.append(str(stem.with_suffix('.pdf')))
    table_source = parent / 'outputs/tables/normalization.csv'
    with table_source.open() as handle:
        table = list(csv.DictReader(handle))
    write_csv(output / 'normalization/fitted_statistics.csv', table)
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    ax.axis('off')
    columns = ['Dataset / representation / channel', 'Mean', 'Measured std', 'Effective std', 'Guarded', 'Fit images']
    values = [[f"{r['dataset']} / {r['representation']} / {r['channel']}",
               f"{float(r['mean']):.6g}", f"{float(r['measured_std']):.6g}",
               f"{float(r['effective_std']):.6g}", r['guarded'], r['training_count']] for r in table]
    artist = ax.table(cellText=values, colLabels=columns, loc='center', cellLoc='center',
                      colWidths=[.38, .13, .14, .14, .09, .12])
    artist.auto_set_font_size(False)
    artist.set_fontsize(10)
    artist.scale(1, 1.8)
    ax.set_title('Frozen normalization fitted on the full clean training partition', fontsize=14, pad=20)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(output / 'normalization' / ('fitted_statistics.' + ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
    generated.append(str(output / 'normalization/fitted_statistics.pdf'))
    fits = []
    for path in sorted({r['artifact'] for r in table}):
        source = Path(path)
        fit = read(source)
        for original in (source, source.parent / fit['npz_file']):
            dest = output / 'normalization/fitted_artifacts' / original.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha(dest) != sha(original):
                raise ValueError(f'Existing fit copy changed: {dest}')
            if not dest.exists():
                shutil.copyfile(original, dest)
            fits.append(dict(source=str(original), published=str(dest), sha256=sha(original)))
    write_csv(output / 'normalization/fitted_artifact_sources.csv', fits)
    return dict(copied_parent_pdfs=copied, normalization_pdfs=generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Render in an allocated CPU job, not on a login node')
    project = args.project.resolve()
    output = (args.output or project / 'plots/diagnostics').resolve()
    if output != project / 'plots/diagnostics':
        raise ValueError('This exporter owns only project/plots/diagnostics')
    output.mkdir(parents=True, exist_ok=True)
    script_hash = sha(__file__)
    study = project / 'training_dense_testing_sparse'
    manifest = read(study / 'study_manifest.json')
    frozen = project / 'sparse_contrast_benchmark_large_batch/outputs/source' / BASE_HASH
    expected_visualize = read(frozen / 'source_manifest.json')['sparse_contrast/visualize.py']
    if sha(frozen / 'sparse_contrast/visualize.py') != expected_visualize:
        raise ValueError('Frozen original renderer identity changed')
    started = datetime.now(timezone.utc).isoformat()
    policy = dict(seed=0, trained_execution='dense', inference_execution='compact',
                  sparsities=list(range(0, 100, 10)), selected_cells={x: cells(x) for x in ('cifar10', 'mnist')},
                  CIFAR_severity_policy='Predetermined middle severity 3; not selected by model outcomes.',
                  architecture_policy='Use Swin saved frontend tensors only if every displayed array and scale exactly equals ViT; otherwise retain both.',
                  samples='All ten original class-balanced fixed sample IDs, two groups of five, all native channels.',
                  unchanged_raw_reference='Dense unmodified raw p0 cases remain in the full source catalog; compact raw p0 explicitly shows natural-zero removal.',
                  fit_distribution_policy='Fixed ten-image clean-training diagnostic only, not full-partition distributions; fitted mean/std table is full training partition.',
                  scope='No new model evaluation, predictions, normalization fits or input preprocessing. Stored arrays only.')
    write_json(output / 'selection_policy.json', policy)
    catalog = []
    for case in manifest['cases']:
        catalog.append({k: case[k] for k in ('test_config_hash', 'dataset', 'architecture', 'representation', 'seed',
                                           'trained_execution', 'inference_execution', 'inference_sparsity_percent',
                                           'checkpoint_sha256', 'normalization_hash')} |
                       dict(diagnostic_directory=str(case_dir(study, case) / 'diagnostics/cells'),
                            expected_official_cell_count=76 if case['dataset'] == 'cifar10' else 16,
                            catalog_scope='All registered cases; only selected rendered tensors are rehashed by this exporter.'))
    write_csv(output / 'full_diagnostic_source_catalog.csv', catalog)
    selected = [c for c in manifest['cases'] if c['seed'] == 0 and c['trained_execution'] == 'dense' and c['inference_execution'] == 'compact']
    grouped = {}
    for case in selected:
        key = (case['dataset'], case['representation'], case['inference_sparsity_percent'])
        grouped.setdefault(key, {})[case['architecture']] = case
    if len(grouped) != 60 or len(selected) != 120:
        raise ValueError('Unexpected active dense-trained diagnostic matrix')
    tasks = []
    for key, cases_by_arch in sorted(grouped.items()):
        if set(cases_by_arch) != {'swin_tiny', 'vit_small'}:
            raise ValueError('Missing architecture counterpart')
        chosen = [cases_by_arch[a] for a in ('swin_tiny', 'vit_small')]
        for corruption, severity in cells(key[0]):
            paths = [case_dir(study, c) / 'diagnostics/cells' / f'{corruption}__{severity}.json' for c in chosen]
            tasks.append(dict(project=str(project), output=str(output), cases=chosen, sources=list(map(str, paths)),
                              corruption=corruption, severity=severity, script_hash=script_hash,
                              key='/'.join(map(str, (*key, corruption, severity)))))
    write_json(output / 'provenance.json', dict(started=started, argv=sys.argv, script_path=str(Path(__file__).resolve()),
               script_sha256=script_hash, study_manifest=str(study / 'study_manifest.json'),
               study_manifest_sha256=sha(study / 'study_manifest.json'), original_renderer_source_hash=BASE_HASH,
               original_renderer_sha256=expected_visualize, slurm_job_id=os.environ['SLURM_JOB_ID'],
               hostname=os.uname().nodename, selection_policy=policy, expected_tasks=len(tasks),
               parent_preview_policy='Byte-identical copies of original PDF/PNG/NPZ/metadata; original previews have 300 dpi PNGs.',
               new_render_policy='Matplotlib/seaborn exact requested serif whitegrid talk theme; PDF/PNG dpi=300.'))
    base_result = normalization_and_parent(project, output)
    results = []
    # Spawn avoids inheriting Matplotlib/Torch state after normalization rendering.
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(render_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            write_json(output / 'progress.json', dict(state='running', completed_tasks=len(results), expected_tasks=len(tasks),
                       updated=datetime.now(timezone.utc).isoformat()))
            if len(results) % 10 == 0:
                print(f'Completed {len(results)}/{len(tasks)} saved-tensor panels', flush=True)
    results.sort(key=lambda row: row['task'])
    products = [p for result in results for p in result['products']]
    write_json(output / 'rendered_panel_catalog.json', results)
    write_csv(output / 'rendered_panel_catalog.csv', [dict(pdf=p['pdf'], pages=p['pages'], png_preview=p['png_preview'],
              source_npz=p['tensor_path'], sample_ids=json.dumps(p['sample_ids']),
              exact_cross_architecture_equivalence=p['signature']['architectures_exactly_equal']) for p in products])
    completed = dict(state='completed', completed_at=datetime.now(timezone.utc).isoformat(), tasks=len(tasks),
                     source_catalog_cases=len(catalog), followup_pdfs=len(products),
                     followup_pdf_pages=sum(p['pages'] for p in products),
                     architecture_equal_tasks=sum(r['architectures_exactly_equal'] for r in results),
                     architecture_distinct_tasks=sum(not r['architectures_exactly_equal'] for r in results),
                     parent_pdfs=len(base_result['copied_parent_pdfs']), normalization_pdfs=len(base_result['normalization_pdfs']),
                     total_pdfs=len(products)+len(base_result['copied_parent_pdfs'])+len(base_result['normalization_pdfs']),
                     source_script_sha256=script_hash)
    write_json(output / 'completed.json', completed)
    write_json(output / 'progress.json', completed)
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == '__main__':
    main()
