import numpy as np
import pytest

from sparse_contrast.report import configurations, aggregate_conditions, mean_sd, paired_changes


def test_full_comparison_order_has_19_conditions():
    conditions = configurations()
    assert len(conditions) == len(set(conditions)) == 19
    assert conditions[0] == ('raw', 'dense', None)
    assert conditions[1:4] == [('single_color', 'dense', 0), ('grayscale', 'dense', 0), ('color_opponency', 'dense', 0)]
    assert ('single_color', 'dense', 0) != ('single_color', 'compact', 0)


def test_seed_sd_and_missingness_are_not_corruption_replicates():
    rows = [dict(dataset='mnist', architecture='vit_small', representation='raw', execution='dense', sparsity_percent=None, seed=seed, clean_accuracy=value, clean_error=None if value is None else 1-value, mCE_raw=None, training_status='completed' if value else 'planned', evaluation_status='partial' if value else 'missing') for seed, value in enumerate((.8, .9, None))]
    result = aggregate_conditions(rows)[0]
    assert result['clean_accuracy_seeds'] == 2 and result['mCE_raw_seeds'] == 0
    assert result['clean_accuracy_mean'] == pytest.approx(.85)
    assert result['clean_accuracy_sd'] == pytest.approx(np.std([.8, .9], ddof=1))
    assert result['mCE_raw_mean'] is None
    with pytest.raises(ValueError):
        aggregate_conditions(rows[:2])
    assert mean_sd([]) == (None, None)
    assert mean_sd([.4]) == (.4, None)


def test_paired_change_uses_same_seed_and_percentage_points():
    common = dict(dataset='mnist', architecture='vit_small', seed=1, execution='dense', sparsity_percent=0, mCE_raw=None)
    rows = [{**common, 'representation': 'raw', 'clean_accuracy': .8}, {**common, 'representation': 'grayscale', 'clean_accuracy': .83}]
    changes = paired_changes(rows)
    contrast = [row for row in changes if row['representation'] == 'grayscale'][0]
    assert contrast['method_minus_baseline_percentage_points'] == pytest.approx(3.)


def test_full_benchmark_completion_requires_all_summaries_not_one():
    from sparse_contrast.report import benchmark_coverage
    seeds = []
    for dataset in ('mnist', 'cifar10'):
        for architecture in ('vit_small', 'swin_tiny'):
            for representation, execution, percent in configurations():
                for seed in (0, 1, 2):
                    seeds.append(dict(registry_id=f'{dataset}/{architecture}/{representation}/{execution}/{percent}/{seed}', execution=execution))
    empty = benchmark_coverage({'seeds': seeds, 'timings': []})
    assert empty['expected_summary_conditions_per_hardware'] == 6528
    assert empty['complete_on_one_matched_hardware'] is False
    one = dict(registry_id=seeds[0]['registry_id'], execution='dense', scope='gpu_raw_to_logits', input_group='clean', hardware_id='test_gpu', batch_size=1, median_ms=1.)
    partial = benchmark_coverage({'seeds': seeds, 'timings': [one]})
    assert partial['observed_by_hardware']['test_gpu'] == 1
    assert partial['complete_on_one_matched_hardware'] is False


@pytest.mark.parametrize('stale_field', ['checkpoint_sha256', 'normalization_hash'])
def test_benchmark_rejects_stale_completed_identity(tmp_path, stale_field):
    import hashlib
    import json
    from sparse_contrast.report import collect_benchmark
    run = tmp_path / 'config123'
    cell = run / 'benchmark' / 'hardware' / 'compact' / 'batch1' / 'cells' / 'clean__test'
    cell.mkdir(parents=True)
    (run / 'final.pt').write_bytes(b'checkpoint evidence')
    sha = hashlib.sha256(b'checkpoint evidence').hexdigest()
    (run / 'completed.json').write_text(json.dumps({'checkpoint_sha256': sha}))
    (run / 'config.json').write_text(json.dumps({'config_hash': 'config123', 'normalization_hash': 'newstats'}))
    identity = {'study_role': 'primary', 'config_hash': 'config123', 'checkpoint_sha256': sha, 'normalization_hash': 'newstats'}
    identity[stale_field] = 'obsolete'
    (cell / 'summary.json').write_text(json.dumps({'state': 'complete', 'identity': identity}))
    with pytest.raises(ValueError, match='identity mismatch'):
        collect_benchmark(run, {})


def test_stage_balancing_excludes_clean_and_requires_every_cell():
    from sparse_contrast.data import corruption_cells
    from sparse_contrast.report import balanced_stage_rows
    base = dict(registry_id='run', execution='compact', batch_size=1, hardware_id='gpu', stage='transformer/block1', dataset='mnist')
    def row(corruption, severity, value):
        return {**base, 'corruption': corruption, 'severity': severity, **{key: value for key in ('cpu_self_ms', 'device_self_ms', 'cpu_total_ms_inclusive', 'device_total_ms_inclusive')}}
    clean = row('clean', 'test', 1000.)
    corruptions = [row(c, s, 2.) for c, s in corruption_cells('mnist')]
    mixed = balanced_stage_rows([clean] + corruptions)
    assert len(mixed) == 2
    assert mixed[0]['input_group'] == 'clean' and mixed[0]['device_total_ms_inclusive'] == 1000.
    assert mixed[1]['input_group'] == 'balanced_corruption' and mixed[1]['device_total_ms_inclusive'] == 2.
    assert len(balanced_stage_rows([clean] + corruptions[:-1])) == 1


def test_saved_summary_mirrors_mean_sd_and_pairs_without_fake_missing(tmp_path):
    from sparse_contrast.report import log_saved_summary
    class Writer:
        def __init__(self): self.scalars = {}
        def add_scalar(self, key, value, step): self.scalars[key] = value
        def flush(self): return None
    writer = Writer()
    condition = dict(representation='grayscale', execution='compact', sparsity_percent=20,
                     clean_accuracy_mean=.8, clean_accuracy_sd=.02, clean_accuracy_seeds=3,
                     clean_error_mean=.2, clean_error_sd=.02, clean_error_seeds=3,
                     mCE_raw_mean=None, mCE_raw_sd=None, mCE_raw_seeds=0)
    pairs = [dict(representation='grayscale', execution='compact', sparsity_percent=20,
                  metric='clean_accuracy', seed=seed, method_minus_baseline_percentage_points=value)
             for seed, value in enumerate((1., 2., 3.))]
    log_saved_summary(writer, [condition], [], pairs, tmp_path)
    assert writer.scalars['conditions/grayscale/compact/p20/clean_accuracy/mean'] == .8
    assert writer.scalars['conditions/grayscale/compact/p20/clean_accuracy/sd'] == .02
    assert 'conditions/grayscale/compact/p20/mCE_raw/mean' not in writer.scalars
    assert writer.scalars['paired/grayscale/compact/p20/clean_accuracy/mean_percentage_points'] == 2.
    assert writer.scalars['paired/grayscale/compact/p20/clean_accuracy/sample_sd_percentage_points'] == 1.


def test_loader_scope_has_seed_matched_baseline_speedup(tmp_path, monkeypatch):
    import csv
    import matplotlib.pyplot as plt
    from sparse_contrast import report
    monkeypatch.setattr(report, 'save_figure', lambda fig, stem, writer: plt.close(fig))
    common = dict(dataset='mnist', architecture='vit_small', seed=0, scope='loader_to_cpu_prediction', input_group='clean', batch_size=1, hardware_id='matched_gpu', mean_ms=2., p95_ms=3.)
    rows = [{**common, 'representation': 'raw', 'execution': 'dense', 'sparsity_percent': None, 'median_ms': 2.},
            {**common, 'representation': 'grayscale', 'execution': 'compact', 'sparsity_percent': 20, 'median_ms': 1.}]
    figures = report.timing_comparisons(rows, [], tmp_path, None)
    assert len(figures) == 1
    assert 'paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction' in figures[0]
    table = list(tmp_path.glob('*.csv'))
    assert len(table) == 1
    with table[0].open() as handle:
        saved = list(csv.DictReader(handle))
    contrast = next(row for row in saved if row['representation'] == 'grayscale')
    assert float(contrast['paired_seed_ratio_of_median_speedup']) == 2.


def test_interpretation_does_not_infer_missing_results():
    from sparse_contrast.report import research_interpretation
    text = '\n'.join(research_interpretation({'seeds': [], 'tokens': [], 'timings': []}))
    assert 'Incomplete: 0/18 conditions with three paired seeds' in text
    assert 'no gain after online frontend overhead is established yet' in text
    assert 'five-level, three-seed sparsity comparisons are not yet available' in text
    assert 'Swin stage-density evidence is incomplete' in text


def test_interpretation_reports_all_seed_agreement_only_with_three_pairs():
    from sparse_contrast.report import research_interpretation
    rows, tokens, timings = [], [], []
    for seed in (0, 1, 2):
        common = dict(dataset='mnist', architecture='vit_small', seed=seed)
        rows.extend([{**common, 'representation': 'raw', 'execution': 'dense', 'sparsity_percent': None, 'clean_accuracy': .8, 'mCE_raw': .3},
                     {**common, 'representation': 'grayscale', 'execution': 'compact', 'sparsity_percent': 0, 'clean_accuracy': .82, 'mCE_raw': .27}])
        tokens.append({**common, 'representation': 'grayscale', 'execution': 'compact', 'sparsity_percent': 0, 'source': 'clean/test', 'metric': 'retained_patches/mean', 'value': 98})
        timing = {**common, 'scope': 'gpu_raw_to_logits', 'batch_size': 1, 'input_group': 'clean', 'hardware_id': 'matched_gpu'}
        timings.extend([{**timing, 'representation': 'raw', 'execution': 'dense', 'sparsity_percent': None, 'median_ms': 2.},
                        {**timing, 'representation': 'grayscale', 'execution': 'compact', 'sparsity_percent': 0, 'median_ms': 1.}])
    full = '\n'.join(research_interpretation({'seeds': rows, 'tokens': tokens, 'timings': timings}))
    assert '+2.00 to +2.00; 1/18 complete; all-three-seed positive 1' in full
    assert '+3.00 to +3.00; 1/18 complete; all-three-seed positive 1' in full
    assert '50.00 to 50.00%' in full
    assert '2.000-2.000x | 1/18; faster all three 1' in full
    assert 'Unlisted GPU/host, batch-1/batch-8' in full
    scope = {'schema_version': 1, 'active_representations': {'mnist': ['grayscale'], 'cifar10': ['single_color', 'grayscale', 'color_opponency']}}
    revised = '\n'.join(research_interpretation({'experiment_scope': scope, 'seeds': rows, 'tokens': tokens, 'timings': timings}))
    assert '+2.00 to +2.00; 1/6 complete; all-three-seed positive 1' in revised
    assert '50.00 to 50.00% | 1/5' in revised
    assert '2.000-2.000x | 1/6; faster all three 1' in revised
    partial = '\n'.join(research_interpretation({'seeds': rows[:-1], 'tokens': [], 'timings': []}))
    assert 'all-three-seed positive 1' not in partial
    assert 'Incomplete: 0/18 conditions with three paired seeds' in partial


def test_primary_training_count_metadata_is_preserved_and_completion_is_explicit():
    from sparse_contrast.report import training_metadata, validate_completed_training_count
    row = {'dataset': 'mnist', 'protocol': 'heldout_val'}
    assert training_metadata(row) == {'protocol': 'heldout_val', 'training_count': 55000}
    assert training_metadata({'dataset': 'cifar10'})['training_count'] == 45000
    assert validate_completed_training_count({'training_count': 55000}, row) == 55000
    for completed in ({}, {'training_count': None}, {'training_count': 60000}):
        with pytest.raises(ValueError, match='Completed training count'):
            validate_completed_training_count(completed, row)
    with pytest.raises(ValueError, match='training count'):
        training_metadata({**row, 'training_count': 60000})
    rows = [dict(dataset='mnist', architecture='vit_small', representation='raw', execution='dense', sparsity_percent=None, seed=seed, protocol='heldout_val', training_count=55000, clean_accuracy=.8, clean_error=.2, mCE_raw=None, training_status='completed', evaluation_status='partial') for seed in (0, 1, 2)]
    aggregated = aggregate_conditions(rows)[0]
    assert aggregated['protocol'] == 'heldout_val' and aggregated['training_count'] == 55000
    assert all(change['training_count'] == 55000 and change['protocol'] == 'heldout_val' for change in paired_changes(rows))


def _scoped_registry(scope):
    rows = []
    for dataset in ('mnist', 'cifar10'):
        for architecture in ('vit_small', 'swin_tiny'):
            for representation, execution, percent in configurations(dataset, scope):
                for seed in (0, 1, 2):
                    rows.append(dict(registry_id=f'{dataset}/{architecture}/{representation}/{execution}/p{percent}/seed{seed}', dataset=dataset, architecture=architecture, representation=representation, execution=execution, sparsity_percent=percent, seed=seed, protocol='heldout_val'))
    return rows


def test_revised_scope_report_has_exact_156_runs_and_4416_timing_summaries(tmp_path, monkeypatch):
    import json
    from sparse_contrast import report
    from sparse_contrast.scope import read_scope
    scope = {'schema_version': 1, 'active_representations': {'mnist': ['grayscale'], 'cifar10': ['single_color', 'grayscale', 'color_opponency']}}
    (tmp_path / 'experiment_scope.json').write_text(json.dumps(scope))
    runs = _scoped_registry(read_scope(tmp_path))
    (tmp_path / 'experiment_manifest.json').write_text(json.dumps({'state': 'planned', 'runs': runs}))
    # Stale evidence outside the active manifest cannot reenter any result table.
    obsolete = tmp_path / 'outputs/runs/heldout_val/mnist/vit_small/single_color/dense/p0/seed0/stale'
    obsolete.mkdir(parents=True)
    (obsolete / 'epochs.jsonl').write_text('not valid evidence and must never be read\n')
    (tmp_path / 'outputs/runtime_estimate.json').write_text(json.dumps({'state': 'provisional_complete_group_coverage', 'expected_main_runs': 228, 'all_232_training_reference_gpu_hours': 99999.}))
    class Writer:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def add_scalar(self, *args): pass
        def flush(self): pass
    monkeypatch.setattr(report, 'SummaryWriter', Writer)
    result = report.generate(tmp_path, plots=False)
    assert result['expected_runs'] == 156
    coverage = json.loads((tmp_path / 'outputs/tables/coverage.json').read_text())
    assert coverage['benchmark_coverage']['expected_summary_conditions_per_hardware'] == 4416
    assert coverage['all_requested_work_complete'] is False
    rows = report.read_csv(tmp_path / 'outputs/tables/conditions.csv')
    assert len(rows) == 52
    assert {row['representation'] for row in rows if row['dataset'] == 'mnist'} == {'raw', 'grayscale'}
    assert len(report.read_csv(tmp_path / 'outputs/tables/run_coverage.csv')) == 156
    text = (tmp_path / 'REPORT.md').read_text()
    assert '0/156 final checkpoints completed' in text
    assert 'All 7 configurations' in text and 'All 19 configurations' in text
    assert 'Incomplete: 0/6 conditions with three paired seeds' in text
    assert 'Incomplete: 0/18 conditions with three paired seeds' in text
    assert '99999' not in text
    # A registry that reintroduces a withdrawn condition must fail, not hide it.
    runs.append({**runs[0], 'registry_id': 'withdrawn', 'representation': 'single_color'})
    (tmp_path / 'experiment_manifest.json').write_text(json.dumps({'runs': runs}))
    with pytest.raises(ValueError, match='authorized dataset-specific scope'):
        report.collect_evidence(tmp_path)


def test_revised_scope_filters_old_normalization_and_preprocessing_files(tmp_path, monkeypatch):
    import json
    import matplotlib.pyplot as plt
    from sparse_contrast import report
    scope = {'schema_version': 1, 'active_representations': {'mnist': ['grayscale'], 'cifar10': ['single_color', 'grayscale', 'color_opponency']}}
    (tmp_path / 'experiment_scope.json').write_text(json.dumps(scope))
    (tmp_path / 'normalization').mkdir()
    preprocessing = []
    for dataset in ('mnist', 'cifar10'):
        for representation in report.REPRESENTATIONS:
            channels = 1 if representation == 'grayscale' else 3
            data = dict(dataset=dataset, representation=representation, mean=[0.] * channels, sigma=[1.] * channels, effective_std=[1.] * channels, guarded_channels=[False] * channels)
            (tmp_path / 'normalization' / f'{dataset}_{representation}_heldout_val.json').write_text(json.dumps(data))
            preprocessing.append(dict(dataset=dataset, representation=representation, patch_size=2, sparsity_percent=0, empty_patch_fraction=.2, achieved_zero_fraction=.1))
    norms = report.normalization_rows(tmp_path)
    assert len(norms) == 8
    assert {row['representation'] for row in norms if row['dataset'] == 'mnist'} == {'grayscale'}
    report.save_csv(tmp_path / 'outputs/preprocessing_pilot/summary.csv', preprocessing)
    class Writer:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(report, 'SummaryWriter', Writer)
    monkeypatch.setattr(report, 'save_figure', lambda fig, stem, writer: plt.close(fig))
    figures = report.preprocessing_figures(tmp_path)
    assert len(figures) == 8
    assert all('single_color' not in path and 'color_opponency' not in path for path in figures if '/mnist/' in path)


def test_missing_scope_file_retains_legacy_fixture_counts(tmp_path):
    from sparse_contrast.scope import read_scope
    from sparse_contrast.report import benchmark_coverage
    scope = read_scope(tmp_path)
    rows = _scoped_registry(scope)
    assert len(rows) == 228
    assert benchmark_coverage({'seeds': rows, 'timings': []})['expected_summary_conditions_per_hardware'] == 6528


def test_recipe_provenance_distinguishes_actual_legacy_and_planned_group_batches(tmp_path):
    import json
    from sparse_contrast.report import recipe_provenance
    manifest = {'recipe': {'batch_size': 128, 'effective_batch_size': 128, 'eval_batch_size': 256, 'lr': .0003}, 'recipe_by_group': {'mnist/swin_tiny': {'batch_size': 64, 'effective_batch_size': 64, 'lr': .0006}}}
    (tmp_path / 'experiment_manifest.json').write_text(json.dumps(manifest))
    run = tmp_path / 'outputs/runs/legacy'
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'dataset': 'mnist', 'architecture': 'vit_small', 'recipe': {'batch_size': 8, 'effective_batch_size': 8, 'lr': .0003}}))
    evidence = {'seeds': [{'dataset': 'mnist', 'architecture': 'vit_small', 'run_directory': str(run)}]}
    rows = {(row['dataset'], row['architecture']): row for row in recipe_provenance(tmp_path, evidence)}
    actual = rows[('mnist', 'vit_small')]
    assert (actual['train_batch_size'], actual['eval_batch_size'], actual['base_lr']) == (8, 8, .0003)
    assert actual['status'] == 'saved run configs (1 runs)'
    planned = rows[('mnist', 'swin_tiny')]
    assert (planned['train_batch_size'], planned['eval_batch_size'], planned['base_lr']) == (64, 256, .0006)
    assert planned['status'].startswith('planned from manifest')
    manifest.update(batch_selection_required=True, batch_selection={'state': 'pending'})
    (tmp_path / 'experiment_manifest.json').write_text(json.dumps(manifest))
    rows = recipe_provenance(tmp_path, {'seeds': []})
    assert all(row['train_batch_size'] is None and row['eval_batch_size'] is None for row in rows)
    assert all(row['status'] == 'planned; batch selection pending' for row in rows)
    evidence['seeds'][0]['run_directory'] = str(tmp_path.parent / 'another_campaign/run')
    with pytest.raises(ValueError, match='different campaign root'):
        recipe_provenance(tmp_path, evidence)
