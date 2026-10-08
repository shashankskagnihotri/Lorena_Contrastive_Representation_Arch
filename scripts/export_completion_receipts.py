#!/usr/bin/env python3
"""Copy primary checkpoint completion receipts by registry identity, not glob depth.

This reads receipt JSON only, never the 128 GiB checkpoint collection. Saved
checkpoint hashes identify the artifacts; this export does not rehash the blobs.
"""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run the original-cluster receipt export in a CPU allocation')
    registry = json.loads((ROOT / 'experiments/training_sparse_testing_sparse/experiment_manifest.json').read_text())
    records = []
    for run in registry['runs']:
        config_hash = run['resolved_config_hash']
        source = ROOT / 'sparse_contrast_benchmark_large_batch/outputs/runs' / run['protocol'] / run['registry_id'] / config_hash / 'completed.json'
        raw = source.read_bytes()
        receipt = json.loads(raw)
        if receipt.get('config_hash') != config_hash or not receipt.get('checkpoint_sha256'):
            raise ValueError(f'Receipt identity mismatch: {source}')
        target = ROOT / 'results/training_sparse_testing_sparse/run_receipts' / config_hash / 'completed.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise FileExistsError(f'Refusing changed receipt: {target}')
        if not target.exists():
            target.write_bytes(raw)
        records.append(dict(source=str(source.relative_to(ROOT)), target=str(target.relative_to(ROOT)),
                            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), checkpoint_sha256=receipt['checkpoint_sha256']))
    if len(records) != registry['expected_runs'] or len(records) != 156:
        raise ValueError('Primary completion receipt coverage is incomplete')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), job_id=os.environ['SLURM_JOB_ID'],
                    copied=records, checkpoint_bytes_rehashed=False)
    (ROOT / 'provenance/completion_receipts.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported and verified {len(records)} completed main-run receipts')

if __name__ == '__main__':
    main()
