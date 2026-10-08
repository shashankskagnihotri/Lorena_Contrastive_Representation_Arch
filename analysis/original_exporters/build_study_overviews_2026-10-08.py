#!/usr/bin/env python3
"""Saved-evidence study overviews; run rendering in an allocated CPU job.

Writes only plots/overview/. Does not import experiment code, read checkpoints,
rehash raw timing/prediction files, or modify either campaign.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NEW = PROJECT / 'training_dense_testing_sparse'
BASE = PROJECT / 'sparse_contrast_benchmark_large_batch'
OUT = PROJECT / 'plots/overview'
METRICS = ('clean_accuracy', 'mCE_raw', 'brightness_accuracy', 'contrast_accuracy', 'fog_accuracy', 'clean_loss')
SCOPES = ('gpu_raw_to_logits', 'host_raw_to_logits', 'loader_to_cpu_prediction', 'cached_input_model_only_diagnostic')
LEVELS = (0, 40, 60, 80)
FAMILY_NAMES = {
    'dense0_compact_test': 'Train dense 0%; test compact',
    'compact0_compact_test': 'Train compact 0%; test compact',
    'matched_sparse_training': 'Train and test at matched sparsity',
    'original_dense_same_checkpoint': 'Original dense 0%, same checkpoint',
    'raw_dense_reference': 'Raw dense reference',
    'raw_compact_test': 'Raw dense training; compact pixel test',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None:
        assert actual == expected, (str(path), 'hash mismatch')
    return json.loads(data), dict(path=str(Path(path).resolve()), sha256=actual)


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def moments(values):
    assert values and all(math.isfinite(value) for value in values)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def accuracy(provenance):
    manifest, binding = read(NEW / 'study_manifest.json')
    provenance['manifest'] = binding
    assert digest({k:v for k,v in manifest.items() if k != 'manifest_hash'}) == manifest['manifest_hash']
    coverage, binding = read(NEW / 'outputs/tables/coverage.json')
    provenance['new_coverage'] = binding
    assert coverage['accuracy_complete'] and coverage['completed_evaluation_cases'] == 612
    assert coverage['completed_evaluation_cells'] == 35352 and coverage['manifest_hash'] == manifest['manifest_hash']
    cases = {case['test_config_hash']:case for case in manifest['cases']}
    path = NEW / 'outputs/tables/accuracy_seeds.csv'
    source = csv_rows(path)
    provenance['new_accuracy_table'] = dict(path=str(path), sha256=sha(path))
    assert len(source) == 612 and {row['test_config_hash'] for row in source} == set(cases)
    rows = []
    for old in source:
        case = cases[old['test_config_hash']]
        assert old['state'] == 'complete' and old['checkpoint_sha256'] == case['checkpoint_sha256']
        assert old['training_config_hash'] == case['training_config_hash'] and old['source_hash'] == case['source_hash']
        assert int(old['completed_cells']) == int(old['expected_cells']) == (16 if old['dataset'] == 'mnist' else 76)
        family = ('raw_dense_reference' if case['inference_execution'] == 'dense' else 'raw_compact_test') if case['representation'] == 'raw' else case['trained_execution'] + '0_compact_test'
        row = dict(study='training_dense_testing_sparse', dataset=case['dataset'], architecture=case['architecture'],
            representation=case['representation'], family=family, trained_execution=case['trained_execution'],
            inference_execution=case['inference_execution'], sparsification_domain=case['sparsification_domain'],
            sparsity_percent=case['inference_sparsity_percent'], seed=case['seed'], protocol=old['protocol'],
            training_count=int(old['training_count']), test_count_per_cell=int(old['test_count_per_cell']),
            config_hash=case['test_config_hash'], training_config_hash=case['training_config_hash'], checkpoint_sha256=case['checkpoint_sha256'])
        row.update({metric:float(old[metric]) if old.get(metric) not in ('', None) else None for metric in METRICS})
        rows.append(row)
    parent_coverage, binding = read(BASE / 'outputs/tables/coverage.json')
    provenance['parent_coverage'] = binding
    # Parent accuracy is independent of whether its final timing report is rendering.
    provenance['parent_all_requested_work_complete_at_read'] = parent_coverage.get('all_requested_work_complete')
    path = BASE / 'outputs/tables/cells.csv'
    parent_cells = csv_rows(path)
    provenance['parent_cell_table'] = dict(path=str(path), sha256=sha(path))
    path = BASE / 'outputs/tables/run_coverage.csv'
    parent_runs = csv_rows(path)
    provenance['parent_run_table'] = dict(path=str(path), sha256=sha(path))
    assert len(parent_runs) == 156
    grouped = defaultdict(list)
    for cell in parent_cells:
        grouped[cell['registry_id']].append(cell)
    for run in parent_runs:
        cells = grouped[run['registry_id']]
        ds, rep, execution = run['dataset'], run['representation'], run['execution']
        assert len(cells) == (16 if ds == 'mnist' else 76)
        assert len({(cell['corruption'],cell['severity']) for cell in cells}) == len(cells)
        assert all(int(cell['total']) == 10000 for cell in cells)
        clean = [cell for cell in cells if cell['corruption'] == 'clean']
        assert len(clean) == 1
        corrupt = [cell for cell in cells if cell['corruption'] != 'clean']
        family = 'raw_dense_reference' if rep == 'raw' else 'original_dense_same_checkpoint' if execution == 'dense' else 'matched_sparse_training'
        row = dict(study='training_sparse_testing_sparse', dataset=ds, architecture=run['architecture'], representation=rep,
            family=family, trained_execution=execution, inference_execution=execution,
            sparsification_domain='none' if execution == 'dense' else 'native_contrast',
            sparsity_percent=0 if run['sparsity_percent'] == '' else int(run['sparsity_percent']), seed=int(run['seed']),
            protocol=run['protocol'], training_count=int(run['training_count']), test_count_per_cell=10000,
            config_hash=Path(run['run_directory']).name, training_config_hash=Path(run['run_directory']).name,
            checkpoint_sha256=run['checkpoint_sha256'], clean_accuracy=float(clean[0]['accuracy']),
            clean_loss=float(clean[0]['loss']), mCE_raw=statistics.mean(float(cell['error']) for cell in corrupt))
        assert math.isclose(row['mCE_raw'], float(run['mCE_raw']), abs_tol=1e-12)
        for name in ('brightness', 'contrast', 'fog'):
            subset = [float(cell['accuracy']) for cell in corrupt if cell['corruption'] == name]
            row[name + '_accuracy'] = statistics.mean(subset) if subset else None
        rows.append(row)
    assert len(rows) == 768
    original = {row['training_config_hash']:row for row in rows if row['study'] == 'training_sparse_testing_sparse'}
    for case in manifest['sources']:
        ref = original[case['training_config_hash']]
        assert ref['checkpoint_sha256'] == case['checkpoint_sha256']
    return manifest, rows


def aggregate(rows):
    groups = defaultdict(list)
    fields = ('study', 'dataset', 'architecture', 'representation', 'family', 'trained_execution', 'inference_execution', 'sparsification_domain', 'sparsity_percent', 'protocol', 'training_count')
    for row in rows:
        groups[tuple(row[key] for key in fields)].append(row)
    result = []
    for keys, group in sorted(groups.items()):
        assert sorted(row['seed'] for row in group) == [0, 1, 2]
        row = dict(zip(fields, keys), completed_seeds=3, expected_seeds=3)
        for metric in METRICS:
            values = [item[metric] for item in group if item[metric] is not None]
            row[metric + '_mean'], row[metric + '_sd'] = moments(values) if values else (None, None)
        result.append(row)
    assert len(result) == 256
    return result


def comparisons(rows):
    parent = [row for row in rows if row['study'] == 'training_sparse_testing_sparse']
    matched = {(r['dataset'],r['architecture'],r['representation'],r['sparsity_percent'],r['seed']):r for r in parent if r['family']=='matched_sparse_training'}
    raw = {(r['dataset'],r['architecture'],r['seed']):r for r in parent if r['family']=='raw_dense_reference'}
    original = {r['training_config_hash']:r for r in parent}
    seed_rows = []
    for row in rows:
        if row['study'] != 'training_dense_testing_sparse' or row['sparsity_percent'] not in LEVELS:
            continue
        references = {'raw_dense':raw[row['dataset'],row['architecture'],row['seed']], 'original_same_checkpoint':original[row['training_config_hash']]}
        match = matched.get((row['dataset'],row['architecture'],row['representation'],row['sparsity_percent'],row['seed']))
        if match is not None:
            references['matched_sparse_training'] = match
        for ref_name, ref in references.items():
            for metric in METRICS:
                if row[metric] is None or ref[metric] is None:
                    continue
                scale = 1 if metric == 'clean_loss' else 100
                seed_rows.append({k:row[k] for k in ('dataset','architecture','representation','family','sparsity_percent','seed','protocol','training_count')} |
                    dict(reference=ref_name, metric=metric, units='cross_entropy' if scale==1 else 'percentage_points',
                         value=scale*row[metric], reference_value=scale*ref[metric], delta=scale*(row[metric]-ref[metric]),
                         checkpoint_sha256=row['checkpoint_sha256'], reference_checkpoint_sha256=ref['checkpoint_sha256']))
    groups = defaultdict(list)
    keys = ('dataset','architecture','representation','family','sparsity_percent','protocol','training_count','reference','metric','units')
    for row in seed_rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summary = []
    for values, group in sorted(groups.items()):
        assert sorted(row['seed'] for row in group) == [0,1,2]
        mean, sd = moments([row['delta'] for row in group])
        summary.append(dict(zip(keys,values), seeds=3, mean=statistics.mean(row['value'] for row in group),
            reference_mean=statistics.mean(row['reference_value'] for row in group), paired_delta_mean=mean,
            paired_delta_sample_sd=sd, positive_seeds=sum(row['delta']>0 for row in group), negative_seeds=sum(row['delta']<0 for row in group)))
    return seed_rows, summary


def latency(manifest, provenance):
    state, binding = read(NEW / 'outputs/campaign.json')
    provenance['latency_campaign_snapshot'] = binding
    cases = {case['test_config_hash']:case for case in manifest['cases']}
    expected = {worker['worker_id']:worker for worker in manifest['latency_workers']}
    assert len(expected) == 2424 and {w['worker_id'] for w in state['latency_workers']} == set(expected)
    contract_doc, binding = read(NEW / 'outputs/benchmarks/hardware_contract.json')
    provenance['latency_hardware_contract'] = binding
    contract = contract_doc['contract']
    provenance['hardware_contract_fields'] = contract
    hw_cache, panels, records, bindings = {}, {}, [], []
    completed = [w for w in state['latency_workers'] if w['state'] == 'completed']
    for index, worker in enumerate(completed, 1):
        registered = expected[worker['worker_id']]
        for name,value in registered.items():
            assert worker[name] == value
        case = cases[worker['test_config_hash']]
        summary, binding = read(worker['summary_path'], worker['summary_sha256'])
        assert summary['state'] == 'complete' and summary['case'] == case
        assert summary['execution'] == worker['execution'] and summary['batch_size'] == worker['batch_size']
        assert summary['num_shards'] == worker['num_shards'] == 1
        expected_count = 16 if case['dataset'] == 'mnist' else 76
        assert len(summary['cells']) == len(summary['expected_cells']) == expected_count
        assert {(c['corruption'],str(c['severity'])) for c in summary['cells']} == {(c,str(s)) for c,s in summary['expected_cells']}
        assert summary['hardware_contract_sha256'] == provenance['latency_hardware_contract']['sha256']
        refs = [ref for ref in summary['cells'] if ref['corruption']=='clean' and ref['severity']=='test']
        assert len(refs)==1
        cell, cbinding = read(refs[0]['path'], refs[0]['sha256'])
        assert cell['state']=='complete' and cell['batch_size']==worker['batch_size']
        for name in ('test_config_hash','training_config_hash','checkpoint_sha256','normalization_hash','source_hash','trained_execution','inference_execution','inference_sparsity_percent','sparsification_domain'):
            assert cell['identity'][name] == case[name]
        assert cell['identity']['execution']==worker['execution']
        measurement=cell['measurement']; hs=measurement['hardware_metadata_sha256']
        if hs not in hw_cache:
            hw_cache[hs]=read(measurement['hardware_metadata_file'],hs)[0]
        hw=hw_cache[hs]
        assert all(hw.get(k)==v for k,v in contract.items())
        assert hw['device_uuid']==measurement['measurement_device_uuid'] and hw['measurement_session_id']==measurement['measurement_session_id']
        panel_key=(case['dataset'],case['architecture'],case['seed'],worker['batch_size'])
        ph=cell['file_hashes']['batch_membership.json']
        assert panel_key not in panels or panels[panel_key]==ph
        panels[panel_key]=ph
        assert cell['warmup_batches_per_scope']==50 and cell['measured_batches_per_scope']==200
        family=('raw_dense_reference' if case['inference_execution']=='dense' else 'raw_compact_test') if case['representation']=='raw' else case['trained_execution']+'0_compact_test'
        for scope in SCOPES:
            value=cell['scopes'][scope];assert value['count']==200
            records.append(dict(dataset=case['dataset'],architecture=case['architecture'],representation=case['representation'],family=family,
                trained_execution=case['trained_execution'],inference_execution=case['inference_execution'],execution=worker['execution'],
                sparsity_percent=case['inference_sparsity_percent'],seed=case['seed'],batch_size=worker['batch_size'],scope=scope,
                protocol='heldout_val',training_count=55000 if case['dataset']=='mnist' else 45000,
                hardware_identity=measurement['hardware_identity'],median_ms=value['median_ms'],p95_ms=value['p95_ms'],panel_sha256=ph,
                worker_id=worker['worker_id'],test_config_hash=case['test_config_hash'],measurement_device_uuid=measurement['measurement_device_uuid']))
        bindings.append(dict(worker_id=worker['worker_id'],summary=binding,clean_cell=cbinding,measurement=measurement))
        if index%100==0:
            print(f'Checked {index}/{len(completed)} sealed latency workers',flush=True)
    provenance['sealed_latency_bindings']=bindings
    coverage=dict(snapshot_updated=state['updated'],expected_workers=2424,sealed_full_workers=len(completed),
        worker_states=dict(Counter(w['state'] for w in state['latency_workers'])),sealed_clean_cells=len(completed),
        hardware_sessions=len(hw_cache),timing_raw_rehashed=False)
    groups=defaultdict(list)
    fields=('dataset','architecture','representation','family','trained_execution','inference_execution','execution','sparsity_percent','batch_size','scope','protocol','training_count','hardware_identity')
    baseline={(r['dataset'],r['architecture'],r['seed'],r['batch_size'],r['scope'],r['hardware_identity']):r for r in records if r['family']=='raw_dense_reference' and r['execution']=='dense'}
    for record in records:
        groups[tuple(record[k] for k in fields)].append(record)
    conditions=[]
    for values, group in sorted(groups.items()):
        assert len({r['seed'] for r in group})==len(group)
        row=dict(zip(fields,values),completed_seeds=len(group),expected_seeds=3,mean_seed_median_ms=None,sample_sd_ms=None,
                 paired_speedup_mean=None,paired_speedup_sample_sd=None,faster_seeds=None)
        refs=[baseline.get((r['dataset'],r['architecture'],r['seed'],r['batch_size'],r['scope'],r['hardware_identity'])) for r in group]
        if sorted(r['seed'] for r in group)==[0,1,2] and all(refs):
            row['mean_seed_median_ms'],row['sample_sd_ms']=moments([r['median_ms'] for r in group])
            ratios=[ref['median_ms']/r['median_ms'] for r,ref in zip(group,refs)]
            row['paired_speedup_mean'],row['paired_speedup_sample_sd']=moments(ratios)
            row['faster_seeds']=sum(ratio>1 for ratio in ratios)
        conditions.append(row)
    coverage['primary_complete_three_seed_condition_batches']=sum(r['scope']=='gpu_raw_to_logits' and r['execution']==r['inference_execution'] and r['paired_speedup_mean'] is not None for r in conditions)
    return records,conditions,coverage


def save_figure(fig,stem,rows,figures):
    import matplotlib.pyplot as plt
    for extension in ('png','pdf','svg'):
        fig.savefig(OUT/(stem+'.'+extension),dpi=300,bbox_inches='tight')
    write_csv(OUT/(stem+'.csv'),rows)
    plt.close(fig)
    figures.append(stem)


def render_accuracy(conditions,figures):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context='talk',style='whitegrid',font='serif',font_scale=1.0)
    palette=sns.color_palette('colorblind',6)
    styles={'dense0_compact_test':(palette[0],'-','o'),'compact0_compact_test':(palette[2],'--','s'),
            'matched_sparse_training':(palette[1],'-','D'),'original_dense_same_checkpoint':(palette[0],':',None),
            'raw_dense_reference':('black','--',None),'raw_compact_test':(palette[4],'-','o')}
    labels={'clean_accuracy':'Clean accuracy (%) ↑','mCE_raw':'Mean corruption error (%) ↓','brightness_accuracy':'Brightness accuracy (%) ↑',
            'contrast_accuracy':'Contrast accuracy (%) ↑','fog_accuracy':'Fog accuracy (%) ↑','clean_loss':'Clean cross-entropy ↓'}
    for ds,arch in [('mnist','vit_small'),('mnist','swin_tiny'),('cifar10','vit_small'),('cifar10','swin_tiny')]:
        reps=['raw','grayscale'] if ds=='mnist' else ['raw','single_color','grayscale','color_opponency']
        for rep in reps:
            selected=[r for r in conditions if r['dataset']==ds and r['architecture']==arch and
                ((r['representation']==rep and not(r['study']=='training_dense_testing_sparse' and r['family']=='raw_dense_reference')) or
                 (r['representation']=='raw' and r['study']=='training_sparse_testing_sparse'))]
            fig,axes=plt.subplots(2,3,figsize=(19,11.5),layout='constrained')
            families=list(dict.fromkeys(r['family'] for r in selected))
            for ax,metric in zip(axes.flat,METRICS):
                if ds=='mnist' and metric=='contrast_accuracy':
                    ax.text(.5,.5,'MNIST-C has no\ncontrast corruption',ha='center',va='center',transform=ax.transAxes)
                    ax.set_axis_off();continue
                for family in families:
                    group=sorted((r for r in selected if r['family']==family),key=lambda r:r['sparsity_percent'])
                    color,line,marker=styles[family];factor=1 if metric=='clean_loss' else 100
                    assert all(r[metric+'_mean'] is not None for r in group)
                    x=[r['sparsity_percent'] for r in group];y=[factor*r[metric+'_mean'] for r in group];sd=[factor*r[metric+'_sd'] for r in group]
                    if family in ('original_dense_same_checkpoint','raw_dense_reference'):
                        assert len(group)==1
                        ax.axhline(y[0],color=color,linestyle=line,linewidth=2,label=FAMILY_NAMES[family])
                        ax.axhspan(y[0]-sd[0],y[0]+sd[0],color=color,alpha=.07)
                    else:
                        ax.errorbar(x,y,yerr=sd,color=color,linestyle=line,marker=marker,capsize=3,markersize=5,label=FAMILY_NAMES[family])
                ax.set(xlabel='Imposed inference sparsity (%)',ylabel=labels[metric],xticks=[0,20,40,60,80,90],xlim=(-2,92))
                if metric!='clean_loss':ax.set_ylim(0,102)
            handles,legend_labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc='outside lower center',ncol=2,fontsize=13)
            train_n=55000 if ds=='mnist' else 45000
            fig.suptitle(f'{ds.upper()} · {arch.replace("_"," ")} · {rep.replace("_"," ")}\nAll 3 seeds: mean ± sample SD; heldout_val, train N={train_n:,}\nCompact 0% can remove naturally empty patches; original dense references preserve them.',fontsize=17)
            save_figure(fig,f'accuracy_{ds}_{arch}_{rep}',selected,figures)


def render_latency(rows,coverage,figures):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    selected=[r for r in rows if r['execution']==r['inference_execution'] and r['family']!='raw_dense_reference']
    families=[]
    for ds in ['mnist','cifar10']:
        for arch in ['vit_small','swin_tiny']:
            for rep in (['raw','grayscale'] if ds=='mnist' else ['raw','single_color','grayscale','color_opponency']):
                for family in (['raw_compact_test'] if rep=='raw' else ['dense0_compact_test','compact0_compact_test']):
                    families.append((ds,arch,rep,family))
    for batch in (1,8):
        fig,axes=plt.subplots(1,2,figsize=(22,16),layout='constrained')
        plot_rows=[]
        for ax,scope,title in zip(axes,('gpu_raw_to_logits','host_raw_to_logits'),('GPU raw → logits','Host raw → logits')):
            values=np.full((len(families),10),np.nan)
            for y,(ds,arch,rep,family) in enumerate(families):
                for x,pct in enumerate(range(0,100,10)):
                    found=[r for r in selected if (r['dataset'],r['architecture'],r['representation'],r['family'],r['batch_size'],r['scope'],r['sparsity_percent'])==(ds,arch,rep,family,batch,scope,pct)]
                    assert len(found)<=1
                    value=found[0]['paired_speedup_mean'] if found else None
                    if value is not None:values[y,x]=value
                    plot_rows.append(dict(dataset=ds,architecture=arch,representation=rep,family=family,batch_size=batch,scope=scope,
                        sparsity_percent=pct,paired_speedup_mean=value,completed_seeds=found[0]['completed_seeds'] if found else 0,
                        expected_seeds=3,protocol='heldout_val',training_count=55000 if ds=='mnist' else 45000,
                        eligible_complete_three_seed_comparison=value is not None))
            labels=[f'{ds} {arch.replace("_"," ")} | {rep.replace("_"," ")} | '+('train compact0' if f=='compact0_compact_test' else 'train dense0') for ds,arch,rep,f in families]
            ax.set_facecolor('#e5e5e5')
            if np.isfinite(values).any():
                sns.heatmap(values,mask=~np.isfinite(values),annot=True,fmt='.2f',cmap='vlag_r',center=1,
                    vmin=min(.2,float(np.nanmin(values))),vmax=max(1.15,float(np.nanmax(values))),
                    xticklabels=list(range(0,100,10)),yticklabels=labels,ax=ax,
                    annot_kws={'fontsize':9},cbar_kws={'label':'Paired raw / method speedup (×)'})
            else:
                ax.set(xlim=(0,10),ylim=(len(families),0),xticks=np.arange(10)+.5,xticklabels=list(range(0,100,10)),yticks=np.arange(len(families))+.5,yticklabels=labels)
            ax.set(title=title,xlabel='Imposed inference sparsity (%)',ylabel='');ax.tick_params(axis='y',labelsize=10)
        fig.suptitle(f'PARTIAL compact-inference latency vs raw dense · NVIDIA RTX A6000 · batch {batch}\nOnly 3/3 seeds with sealed full workers and identical paired panels; gray cells are missing',fontsize=19)
        fig.supxlabel(f'Coverage snapshot {coverage["snapshot_updated"]}: {coverage["sealed_full_workers"]}/2424 full workers. Mean paired seed-median ratios; >1 is faster. No raw timing rehash.',fontsize=12)
        save_figure(fig,f'partial_clean_latency_batch{batch}',plot_rows,figures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accuracy-only',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--render-saved',action='store_true',help='Render existing hash-verified overview tables without recollecting campaign evidence.')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    provenance=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),limitations='Derived only from saved complete accuracy report tables and sealed full latency worker receipts. No checkpoint/prediction/rawtiming rereads. Partial latency never pools missing seeds or different hardware classes.')
    figures=[];latency_coverage=None
    if args.render_saved:
        previous,_=read(OUT/'provenance.json')
        recorded={entry['path']:entry['sha256'] for entry in previous['outputs']}
        for name in ('accuracy_seeds.csv','accuracy_conditions.csv','partial_clean_latency_conditions.csv'):
            path=OUT/name
            assert sha(path)==recorded[str(path.relative_to(PROJECT))]
        rows=csv_rows(OUT/'accuracy_seeds.csv');conditions=csv_rows(OUT/'accuracy_conditions.csv')
        for row in conditions:
            for key in ('sparsity_percent','training_count','completed_seeds','expected_seeds'):
                row[key]=int(row[key])
            for metric in METRICS:
                for suffix in ('_mean','_sd'):
                    row[metric+suffix]=float(row[metric+suffix]) if row[metric+suffix]!='' else None
        timing_conditions=csv_rows(OUT/'partial_clean_latency_conditions.csv')
        for row in timing_conditions:
            for key in ('sparsity_percent','training_count','batch_size','completed_seeds','expected_seeds','faster_seeds'):
                row[key]=int(row[key]) if row[key]!='' else None
            for key in ('mean_seed_median_ms','sample_sd_ms','paired_speedup_mean','paired_speedup_sample_sd'):
                row[key]=float(row[key]) if row[key]!='' else None
        latency_coverage,_=read(OUT/'partial_latency_coverage.json')
        assert latency_coverage==previous['latency_coverage']
        render_revision=provenance
        provenance=previous
        provenance.setdefault('collection_script',previous['script'])
        provenance.setdefault('render_revisions',[]).append(render_revision)
        provenance['script']=render_revision['script']
    else:
        manifest,rows=accuracy(provenance);conditions=aggregate(rows);paired_seed,paired=comparisons(rows)
        write_csv(OUT/'accuracy_seeds.csv',rows);write_csv(OUT/'accuracy_conditions.csv',conditions)
        write_csv(OUT/'cross_study_paired_seeds_0_40_60_80.csv',paired_seed);write_csv(OUT/'cross_study_paired_conditions_0_40_60_80.csv',paired)
        if not args.accuracy_only:
            timing_seeds,timing_conditions,latency_coverage=latency(manifest,provenance)
            write_csv(OUT/'partial_clean_latency_seeds.csv',timing_seeds);write_csv(OUT/'partial_clean_latency_conditions.csv',timing_conditions)
            (OUT/'partial_latency_coverage.json').write_text(json.dumps(latency_coverage,indent=2)+'\n')
    if not args.no_plots:
        render_accuracy(conditions,figures)
        if latency_coverage is not None:render_latency(timing_conditions,latency_coverage,figures)
    provenance['accuracy_coverage']=dict(new_cases=612,parent_runs=156,new_cells=35352,parent_cells=9336,all_seeds=3,condition_rows=len(conditions))
    provenance['latency_coverage']=latency_coverage;provenance['figures']=figures
    provenance['outputs']=[dict(path=str(path.relative_to(PROJECT)),sha256=sha(path)) for path in sorted(OUT.glob('*')) if path.is_file() and path.suffix in ('.csv','.png','.pdf','.svg')]
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['# Cross-study overviews','',
        'All accuracy panels use all three seeds and show mean ± sample SD. Protocol: heldout_val; train N=55,000 MNIST and45,000 CIFAR-10. Each official test cell has10,000 images. mCE means unnormalized macro corruption error.','',
        'The original dense result and compact-test0% result use the SAME checkpoint. Naturally empty patches can disappear at0%, changing attention/pooling. Dense0-trained and compact0-trained checkpoint families stay separate. Orange matched-sparsity curves are separately trained parent models. Raw pixel sparsification is separate from native contrast sparsification.','',
        'Latency is partial and uses only sealed full workers with all three matching seeds, one frozen A6000 hardware/CPU/thread class, and equal panel hashes. Blank heatmap cells mean incomplete coverage, never zero latency. Raw JSONL timings were not rehashed for these overviews.','',
        '[All seed metrics](accuracy_seeds.csv) · [Condition means/SD](accuracy_conditions.csv) · [Paired0/40/60/80 comparisons](cross_study_paired_conditions_0_40_60_80.csv) · [Provenance](provenance.json)','']
    for stem in figures:
        lines.append(f'- {stem}: [PNG]({stem}.png) · [PDF]({stem}.pdf) · [SVG]({stem}.svg) · [CSV]({stem}.csv)')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(state='complete',output=str(OUT),figures=len(figures),accuracy_rows=len(rows),latency_coverage=latency_coverage),indent=2),flush=True)


if __name__=='__main__':
    main()
