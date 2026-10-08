#!/usr/bin/env python3
"""Create a publication snapshot without moving or editing live experiment trees.

Run inside a CPU allocation. Raw run trees are inventoried by file metadata, not
rehash-read; scientific summaries and copied source bytes are actually hashed.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / 'sparse_contrast_benchmark_large_batch'
FOLLOWUP = ROOT / 'training_dense_testing_sparse'
SOURCES = {
    'training': (PRIMARY, 'source', '9359e4f10c0c2184eef4af8885edc7076bded30188775cc14222f79819f648c4'),
    'measurement': (PRIMARY, 'source', '7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0'),
    'inference': (FOLLOWUP, 'source', '8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb'),
    'validation_repair': (FOLLOWUP, 'validation_source', '3c965bf1a2c9f3f11bb14c0d6ad3354a2266e7d0763b2146b5a8fbc69d314769'),
    'controller_repair': (FOLLOWUP, 'orchestration_source', 'fcfbdad09dd2aa67d5e62616a5ed77f19f205b324d0474ad15af15bef15466f7'),
}
COPIED = []
OMITTED = []

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def copy(source, target):
    source, target = Path(source), Path(target)
    if source.stat().st_size >= 50 * 1024**2:
        OMITTED.append(dict(path=str(source.relative_to(ROOT)), bytes=source.stat().st_size,
                            reason='Large local scientific artifact; excluded from ordinary Git.'))
        return
    before = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(target) != before:
        raise FileExistsError('Refuse to overwrite a changed publication file: ' + str(target))
    if not target.exists():
        shutil.copyfile(source, target)
    if sha(source) != before or sha(target) != before:
        raise RuntimeError('Source changed during snapshot: ' + str(source))
    COPIED.append(dict(source=str(source.relative_to(ROOT)), target=str(target.relative_to(ROOT)),
                       sha256=before, bytes=target.stat().st_size))

def tree(source, target):
    for path in sorted(Path(source).rglob('*')):
        if path.is_file() and not any(x in ('__pycache__', '.pytest_cache', '.git') for x in path.parts):
            copy(path, Path(target) / path.relative_to(source))

def inventory(path):
    counts = Counter()
    sizes = Counter()
    largest = []
    total = 0
    files = 0
    for base, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '.pytest_cache')]
        for name in names:
            item = Path(base) / name
            try:
                stat = item.lstat()
            except FileNotFoundError:
                # Active atomic writers may remove temporary files between reads.
                counts['vanished_during_snapshot'] += 1
                continue
            if item.is_symlink():
                counts['symlinks_not_followed'] += 1
                continue
            files += 1
            total += stat.st_size
            suffix = item.suffix or '<none>'
            counts[suffix] += 1
            sizes[suffix] += stat.st_size
            if stat.st_size >= 100 * 1024**2:
                largest.append(dict(path=str(item.relative_to(ROOT)), bytes=stat.st_size))
    return dict(path=str(path.relative_to(ROOT)), files=files, logical_bytes=total,
                counts=dict(counts), bytes_by_suffix=dict(sizes),
                over_100MiB=sorted(largest, key=lambda x: -x['bytes']),
                method='Metadata-only snapshot of a live tree; no content hash or atomic total claimed.')

def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run the full export/inventory inside a CPU allocation')
    for role, (parent, group, identity) in SOURCES.items():
        tree(parent / 'outputs' / group / identity, ROOT / 'provenance/sources' / identity)
    tree(PRIMARY / 'outputs/source' / SOURCES['measurement'][2] / 'sparse_contrast', ROOT / 'src/sparse_contrast')
    tree(FOLLOWUP / 'outputs/source' / SOURCES['inference'][2] / 'dense_sparse', ROOT / 'src/dense_sparse')
    for dirname, source in [('training_sparse_testing_sparse', PRIMARY), ('training_dense_testing_sparse', FOLLOWUP),
                            ('legacy_batch8', ROOT / 'sparse_contrast_benchmark')]:
        target = ROOT / 'experiments' / dirname
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in ('.md', '.json', '.py', '.sh', '.ini') and path.name != 'STATUS.json':
                copy(path, target / path.name)
        for name in ('configs', 'normalization', 'tests', 'scripts'):
            if (source / name).is_dir():
                tree(source / name, target / name)
        for path in sorted((source / 'provenance').glob('*')) if (source / 'provenance').is_dir() else []:
            if path.is_file() and path.suffix in ('.json', '.py', '.sh', '.txt'):
                copy(path, target / 'provenance' / path.name)
        if dirname == 'legacy_batch8':
            tree(source / 'sparse_contrast', target / 'sparse_contrast')
        for path in sorted((source / 'outputs/tables').glob('*')):
            if path.is_file() and path.suffix in ('.csv', '.json'):
                copy(path, ROOT / 'results' / dirname / path.name)
    for name in ('environment.yml', 'requirements.lock.txt', 'conda-explicit.txt'):
        copy(PRIMARY / name, ROOT / 'provenance/environment' / name)
    for path in sorted((PRIMARY / 'outputs/runs').glob('heldout_val/*/*/*/*/*/*/*/completed.json')):
        copy(path, ROOT / 'results/training_sparse_testing_sparse/run_receipts' / path.parent.name / path.name)
    for name in ('build_requested_error_plots_20261008.py', 'render_requested_latency_20261008.py',
                 'build_study_overviews_2026-10-08.py', 'render_diagnostic_pdfs_20261008.py',
                 'build_preliminary_clean_latency_2026-10-05.py'):
        copy(ROOT / 'debugging' / name, ROOT / 'analysis/original_exporters' / name)
    roles = {role: dict(source_hash=value[2], published_path='provenance/sources/' + value[2])
             for role, value in SOURCES.items()}
    record = dict(created=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ['SLURM_JOB_ID'],
                  sources=roles, copied=COPIED, omitted_large_files=OMITTED,
                  no_live_source_or_checkpoint_modified=True)
    (ROOT / 'provenance/publication_snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Publication copied and hashed:', len(COPIED), 'files', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory_rows = list(pool.map(inventory, [PRIMARY / 'outputs', FOLLOWUP / 'outputs',
                                                   ROOT / 'sparse_contrast_benchmark/outputs', ROOT / 'plots']))
    (ROOT / 'provenance/local_artifact_inventory.json').write_text(json.dumps(
        dict(created=datetime.now(timezone.utc).isoformat(), entries=inventory_rows), indent=2) + '\n')
    for row in inventory_rows:
        print(row['path'], row['files'], row['logical_bytes'], flush=True)

if __name__ == '__main__':
    main()
