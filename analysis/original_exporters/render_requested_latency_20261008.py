#!/usr/bin/env python3
"""Render the requested latency comparison from saved, validated evidence only."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from collections import Counter

PROJECT = Path('/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch')
BASE = PROJECT / 'sparse_contrast_benchmark_large_batch'
NEW = PROJECT / 'training_dense_testing_sparse'
OUT = PROJECT / 'plots'
SCOPE = 'gpu_raw_to_logits'
COLORS = {'raw': 'black', 'color_opponency': '#7B3294', 'single_color': '#E69F00', 'grayscale': '#0072B2'}
NAMES = {'raw': 'Raw baseline: RGB / grayscale', 'color_opponency': 'Color opponency', 'single_color': 'Single color', 'grayscale': 'Grayscale contrast'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    raw = Path(path).read_bytes()
    if expected is not None:
        assert hashlib.sha256(raw).hexdigest() == expected, str(path)
    return json.loads(raw)


def rows(path):
    with Path(path).open() as stream:
        yield from csv.DictReader(stream)


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write_csv(path, records):
    columns = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run plot rendering in the allocated CPU job')
    parent_table = BASE / 'outputs/tables/timings.csv'
    followup_table = OUT / 'overview/partial_clean_latency_seeds.csv'
    old_contract_path = BASE / 'outputs/benchmarks/cohorts/a6000-2026-10-04/hardware_contract.json'
    new_contract_path = NEW / 'outputs/benchmarks/hardware_contract.json'
    contract = read(old_contract_path)['contract']
    assert contract == read(new_contract_path)['contract']
    assert contract['gpu'] == 'NVIDIA RTX A6000'
    assert contract['cpu_model'] == 'AMD EPYC 7413 24-Core Processor'
    hw = contract['hardware_identity']
    overview_path = OUT / 'overview/provenance.json'
    overview = read(overview_path)
    recorded_followup = [r for r in overview['outputs'] if r['path'] == 'plots/overview/partial_clean_latency_seeds.csv']
    assert len(recorded_followup) == 1 and sha(followup_table) == recorded_followup[0]['sha256']
    manifest_path = NEW / 'study_manifest.json'
    cases = {c['test_config_hash']: c for c in read(manifest_path)['cases']}
    references = {r['worker_id']: r for r in overview['sealed_latency_bindings']}
    panels, provenance_cells, seed_rows, parent_sources, remeasurements = {}, [], [], {}, []

    def verify_cell(path, expected_hash=None):
        cell = read(path, expected_hash)
        assert cell['state'] == 'complete' and cell['corruption'] == 'clean' and cell['severity'] == 'test'
        assert cell['measurement']['hardware_identity'] == hw
        assert cell['warmup_batches_per_scope'] == 50 and cell['measured_batches_per_scope'] == 200
        value = cell['scopes'][SCOPE]
        assert value['count'] == 200 and value['latency_unit'] == 'milliseconds_per_batch'
        assert math.isfinite(value['median_ms']) and value['median_ms'] > 0
        identity = cell['identity']
        key = (identity['dataset'], identity['architecture'], int(identity['seed']), int(cell['batch_size']))
        ph = cell['file_hashes']['batch_membership.json']
        assert sha(Path(path).parent / 'batch_membership.json') == ph
        assert key not in panels or panels[key] == ph, ('Cross-study sample panel mismatch', key)
        panels[key] = ph
        provenance_cells.append(dict(**binding(path), panel_sha256=ph, measurement=cell['measurement']))
        return cell

    for r in rows(parent_table):
        if not (r['hardware_id'] == hw and r['scope'] == SCOPE and r['corruption'] == 'clean' and r['severity'] == 'test'):
            continue
        execution = r['execution']
        if execution not in ('dense', 'compact'):
            continue
        cell = verify_cell(Path(r['summary_path']))
        assert float(r['median_ms']) == cell['scopes'][SCOPE]['median_ms']
        ident = cell['identity']
        source_key = (r['dataset'], r['architecture'], r['representation'], int(r['seed']), int(r['batch_size']))
        if execution == 'dense':
            assert source_key not in parent_sources
            parent_sources[source_key] = (r, ident)
        if r['representation'] == 'raw':
            assert execution == 'dense'
            family, level = 'raw_dense_reference', None
        elif execution == 'compact':
            family, level = 'matched_sparse_training', int(r['sparsity_percent'])
            assert level in (0, 20, 40, 60, 80)
        else:
            continue
        seed_rows.append(dict(dataset=r['dataset'], architecture=r['architecture'], batch_size=int(r['batch_size']),
            representation=r['representation'], family=family, sparsity_percent=level, seed=int(r['seed']),
            median_ms=float(r['median_ms']), scope=SCOPE, hardware_identity=hw, panel_sha256=cell['file_hashes']['batch_membership.json'],
            checkpoint_sha256=ident['checkpoint_sha256'], training_config_hash=ident['config_hash'],
            summary_path=r['summary_path'], summary_sha256=sha(r['summary_path']), source_study='training_sparse_testing_sparse'))

    for r in rows(followup_table):
        if r['scope'] != SCOPE or r['hardware_identity'] != hw or r['trained_execution'] != 'dense':
            continue
        is_baseline = r['representation'] == 'raw' and r['inference_execution'] == r['execution'] == 'dense'
        is_curve = r['representation'] != 'raw' and r['inference_execution'] == r['execution'] == 'compact'
        if not (is_baseline or is_curve):
            continue
        ref = references[r['worker_id']]
        cell = verify_cell(Path(ref['clean_cell']['path']), ref['clean_cell']['sha256'])
        ident, case = cell['identity'], cases[r['test_config_hash']]
        for key in ('test_config_hash','training_config_hash','checkpoint_sha256','normalization_hash','source_hash','trained_execution','inference_execution','inference_sparsity_percent'):
            assert ident[key] == case[key]
        assert float(r['median_ms']) == cell['scopes'][SCOPE]['median_ms']
        assert r['panel_sha256'] == cell['file_hashes']['batch_membership.json']
        source_key = (r['dataset'], r['architecture'], r['representation'], int(r['seed']), int(r['batch_size']))
        original, original_identity = parent_sources[source_key]
        assert case['training_config_hash'] == original_identity['config_hash']
        assert case['checkpoint_sha256'] == original_identity['checkpoint_sha256']
        assert case['normalization_hash'] == original_identity['normalization_hash']
        if is_baseline:
            remeasurements.append(dict(dataset=r['dataset'], architecture=r['architecture'], batch_size=int(r['batch_size']), seed=int(r['seed']),
                original_median_ms=float(original['median_ms']), followup_median_ms=float(r['median_ms']),
                followup_minus_original_ms=float(r['median_ms'])-float(original['median_ms']),
                same_checkpoint_sha256=case['checkpoint_sha256'], same_panel_sha256=r['panel_sha256']))
            continue
        seed_rows.append(dict(dataset=r['dataset'], architecture=r['architecture'], batch_size=int(r['batch_size']),
            representation=r['representation'], family='dense_trained_sparse_inference', sparsity_percent=int(r['sparsity_percent']), seed=int(r['seed']),
            median_ms=float(r['median_ms']), scope=SCOPE, hardware_identity=hw, panel_sha256=r['panel_sha256'],
            checkpoint_sha256=case['checkpoint_sha256'], training_config_hash=case['training_config_hash'],
            summary_path=ref['clean_cell']['path'], summary_sha256=ref['clean_cell']['sha256'],
            source_study='training_dense_testing_sparse', test_config_hash=case['test_config_hash']))

    aggregates = []
    for dataset in ('mnist','cifar10'):
        reps = ('grayscale',) if dataset == 'mnist' else ('color_opponency','single_color','grayscale')
        for arch in ('vit_small','swin_tiny'):
            for batch in (1,8):
                planned = [('raw','raw_dense_reference',None)]
                planned += [(rep,'matched_sparse_training',p) for rep in reps for p in (0,20,40,60,80)]
                planned += [(rep,'dense_trained_sparse_inference',p) for rep in reps for p in range(0,91,10)]
                for rep,family,level in planned:
                    selected = [r for r in seed_rows if (r['dataset'],r['architecture'],r['batch_size'],r['representation'],r['family'],r['sparsity_percent']) == (dataset,arch,batch,rep,family,level)]
                    seeds = sorted(r['seed'] for r in selected)
                    assert len(seeds) == len(set(seeds)) and set(seeds) <= {0,1,2}
                    complete = seeds == [0,1,2]
                    if family != 'dense_trained_sparse_inference':
                        assert complete, (dataset,arch,batch,rep,family,level,seeds)
                    values = [r['median_ms'] for r in selected]
                    aggregates.append(dict(dataset=dataset,architecture=arch,batch_size=batch,representation=rep,family=family,
                        sparsity_percent=level,expected_seeds=3,available_seeds=len(seeds),seed_ids=';'.join(map(str,seeds)),
                        plotted=complete,mean_seed_median_ms=statistics.mean(values) if complete else None,
                        sample_sd_ms=statistics.stdev(values) if complete else None,scope=SCOPE,latency_unit='milliseconds_per_batch',hardware_identity=hw))
    assert len(remeasurements) == 24
    write_csv(OUT/'latency_vs_sparsity.csv',aggregates)
    write_csv(OUT/'latency_vs_sparsity_seeds.csv',seed_rows)
    write_csv(OUT/'latency_vs_sparsity_raw_reference_comparison.csv',remeasurements)
    render(aggregates)
    source_paths = [parent_table,followup_table,overview_path,manifest_path,old_contract_path,new_contract_path,OUT/'overview/partial_latency_coverage.json']
    provenance = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script=binding(__file__),
        scheduler_job_id=os.environ['SLURM_JOB_ID'],sources=[binding(p) for p in source_paths],hardware_contract=contract,
        followup_snapshot=read(OUT/'overview/partial_latency_coverage.json'),
        policy=dict(scope='Clean GPU-resident raw-input-to-logits ONLINE latency',unit='milliseconds per batch',
            aggregation='Mean and sample SD (ddof=1) of per-seed medians, exactly seeds0,1,2; no averaging incomplete seeds',
            solid='Original compact training and compact testing at matched0/20/40/60/80%',
            dashed='Original dense0%-trained checkpoints tested compact at0..90%; compact0%-trained family excluded',
            raw='Single original dense-raw baseline reused as common reference. Follow-up remeasurements remain separate and do not rescale any latency.',
            hardware='Identical frozen GPU/CPU/thread/backend class; independent devices/sessions/timestamps retained in cell provenance.',
            gaps='Missing complete three-seed points are NaN and never connected; unmeasured tails are not extrapolated.',
            evidence='CSV medians rechecked against committed cell summaries; exact panel files rehashed; raw timing JSONL not rehashed for this presentation.'),
        cells=provenance_cells,coverage=dict(Counter((r['family']+(' complete' if r['plotted'] else ' incomplete')) for r in aggregates)),
        outputs=[binding(OUT/name) for name in ['latency_vs_sparsity.pdf','latency_vs_sparsity.csv','latency_vs_sparsity_seeds.csv','latency_vs_sparsity_raw_reference_comparison.csv']])
    (OUT/'latency_vs_sparsity.provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    verify_pdf()
    print(json.dumps({'state':'complete','coverage':provenance['coverage'],'pdf':str(OUT/'latency_vs_sparsity.pdf')},indent=2),flush=True)


def render(records):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import seaborn as sns
    sns.set_theme(context='talk',style='whitegrid',font='serif',rc={'figure.dpi':300,'savefig.dpi':300,'pdf.fonttype':42,'axes.titleweight':'regular'})
    target=OUT/'latency_vs_sparsity.pdf';temporary=target.with_name('.latency_vs_sparsity.tmp.pdf')
    with PdfPages(temporary,metadata={'Title':'Clean online latency versus sparsity','Subject':'Measured three-seed means and sample SD; matched A6000/EPYC7413 class'}) as pdf:
        for batch in (1,8):
            fig,axes=plt.subplots(2,2,figsize=(16,13.0),sharex=True)
            fig.subplots_adjust(left=.09,right=.98,top=.83,bottom=.16,hspace=.31,wspace=.21)
            fig.suptitle(f'Clean online latency versus sparsity | Batch size {batch}',fontsize=23,y=.99)
            colors=[Line2D([0],[0],color=COLORS[r],lw=2.5,label=NAMES[r]) for r in ('raw','color_opponency','single_color','grayscale')]
            fig.legend(handles=colors,loc='upper center',bbox_to_anchor=(.54,.955),ncol=4,frameon=False,fontsize=15)
            styles=[Line2D([0],[0],color='.3',lw=2.5,marker='o',label='Solid: trained and tested at matching sparsity'),
                    Line2D([0],[0],color='.3',lw=2.5,ls='--',marker='s',label='Dashed: trained dense; tested sparse')]
            fig.legend(handles=styles,loc='upper center',bbox_to_anchor=(.54,.914),ncol=2,frameon=False,fontsize=14)
            completeness=[]
            for row,dataset in enumerate(('mnist','cifar10')):
                reps=('grayscale',) if dataset=='mnist' else ('color_opponency','single_color','grayscale')
                for col,arch in enumerate(('vit_small','swin_tiny')):
                    ax=axes[row,col];subset=[r for r in records if (r['dataset'],r['architecture'],r['batch_size'])==(dataset,arch,batch)]
                    raw=next(r for r in subset if r['family']=='raw_dense_reference')
                    mean,sd=raw['mean_seed_median_ms'],raw['sample_sd_ms']
                    ax.axhline(mean,color='black',lw=2.0,zorder=2)
                    ax.axhspan(mean-sd,mean+sd,color='black',alpha=.08,lw=0)
                    for rep in reps:
                        for family,ls,marker in [('matched_sparse_training','-','o'),('dense_trained_sparse_inference','--','s')]:
                            points=sorted((r for r in subset if r['family']==family and r['representation']==rep),key=lambda r:r['sparsity_percent'])
                            x=np.array([r['sparsity_percent'] for r in points]);y=np.array([r['mean_seed_median_ms'] if r['plotted'] else np.nan for r in points]);sd=np.array([r['sample_sd_ms'] if r['plotted'] else np.nan for r in points])
                            ax.plot(x,y,color=COLORS[rep],ls=ls,marker=marker,ms=5.5,lw=2.1,zorder=3)
                            ax.fill_between(x,y-sd,y+sd,color=COLORS[rep],alpha=.12,lw=0)
                    complete=sum(r['plotted'] for r in subset if r['family']=='dense_trained_sparse_inference');expected=10*len(reps)
                    ax.text(.03,.05,f'Dashed coverage: {complete}/{expected} points',ha='left',va='bottom',transform=ax.transAxes,fontsize=11.5,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
                    ax.set_title(('MNIST' if dataset=='mnist' else 'CIFAR-10')+' | '+('ViT-Small' if arch=='vit_small' else 'Swin-Tiny'),fontsize=19)
                    ax.set_xlim(0,90);ax.set_xticks(range(0,91,10));ax.set_ylim(bottom=0);ax.tick_params(labelsize=12)
                    ax.set_xlabel('Inference sparsity (%)',fontsize=16);ax.set_ylabel('Latency (ms / batch)',fontsize=16)
                    ax.tick_params(labelbottom=True);ax.grid(axis='x',alpha=.18);ax.grid(axis='y',alpha=.4)
            fig.text(.5,.103,'Mean of three seed medians; bands show sample SD. Each seed: 50 warm-up + 200 measured batches.',ha='center',fontsize=12)
            fig.text(.5,.078,'NVIDIA RTX A6000 | AMD EPYC 7413 | FP32 | 4 CPU threads. Missing three-seed points are not connected.',ha='center',fontsize=12)
            fig.text(.5,.053,'Black line: original dense-raw common reference. Exact checkpoints and timing panels match across studies; sessions differ.',ha='center',fontsize=11.2)
            fig.text(.5,.029,'Follow-up coverage snapshot: 2026-10-08 11:24 UTC. Raw-pixel sparse sweeps and compact-trained 0% follow-up family are excluded.',ha='center',fontsize=10.8)
            pdf.savefig(fig,dpi=300);plt.close(fig)
    os.replace(temporary,target)


def verify_pdf():
    sys.path.insert(0,str(PROJECT/'debugging/plot_tools'))
    import pypdfium2 as pdfium
    document=pdfium.PdfDocument(OUT/'latency_vs_sparsity.pdf')
    assert len(document)==2
    checks=[];previews=OUT/'requested_comparisons/previews';previews.mkdir(parents=True,exist_ok=True)
    for i in range(2):
        page=document[i];text=page.get_textpage().get_text_range()
        for label in ('MNIST','CIFAR-10','ViT-Small','Swin-Tiny','Latency','Inference sparsity'):
            assert label in text,(i,label)
        assert f'Batch size {1 if i==0 else 8}' in text
        preview=previews/f'latency_vs_sparsity_page{i+1}.png'
        image=page.render(scale=1.2).to_pil();image.save(preview)
        checks.append(dict(page=i+1,width_points=page.get_width(),height_points=page.get_height(),preview=binding(preview)))
    audit=dict(state='passed',pdf=binding(OUT/'latency_vs_sparsity.pdf'),pages=checks,raster_export_dpi=300,vector_text_and_lines=True)
    (OUT/'requested_comparisons/latency_pdf_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


def render_saved():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run rendering in an allocated CPU job')
    provenance_path=OUT/'latency_vs_sparsity.provenance.json';provenance=read(provenance_path)
    path=OUT/'latency_vs_sparsity.csv'
    expected=next(x['sha256'] for x in provenance['outputs'] if x['path']==str(path))
    assert sha(path)==expected
    records=list(rows(path))
    for r in records:
        r['batch_size']=int(r['batch_size']);r['plotted']=r['plotted']=='True'
        r['sparsity_percent']=int(r['sparsity_percent']) if r['sparsity_percent'] else None
        for key in ('mean_seed_median_ms','sample_sd_ms'):
            r[key]=float(r[key]) if r[key] else None
    render(records);verify_pdf()
    provenance.setdefault('render_revisions',[]).append(dict(script=binding(__file__),scheduler_job_id=os.environ['SLURM_JOB_ID'],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),change='Clarify raw versus grayscale contrast legend and move coverage labels clear of curves; unchanged plotted CSV'))
    provenance['outputs']=[binding(x['path']) for x in provenance['outputs']]
    provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print('Presentation rerender complete; saved numerical CSV unchanged',flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--render-saved']:render_saved()
    elif not sys.argv[1:]:main()
    else:raise SystemExit('Usage: render_requested_latency.py [--render-saved]')
