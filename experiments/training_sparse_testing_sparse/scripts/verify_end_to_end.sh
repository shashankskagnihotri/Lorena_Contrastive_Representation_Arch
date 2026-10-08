#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/test_campaign.py tests/test_report.py --junitxml=outputs/checks/controller-report-tests.xml
if [ ! -f outputs/checks/runtime-completed.json ]; then
    bash scripts/verify_runtime.sh
fi
python report.py
python - <<'PY'
import json
from sparse_contrast.common import ROOT,atomic_json
core=json.loads((ROOT/'outputs/checks/core-completed.json').read_text())
smoke=json.loads((ROOT/'outputs/checks/smoke-completed.json').read_text())
overfit=list((ROOT/'outputs/checks/overfit').glob('*/*/*/completed.json'))
runtime=json.loads((ROOT/'outputs/checks/runtime-completed.json').read_text())
records=[json.loads(p.read_text()) for p in overfit]
assert core['passed'] and smoke['passed'] and runtime['passed']
assert {(r['dataset'],r['architecture']) for r in records}=={(d,a) for d in ['mnist','cifar10'] for a in ['vit_small','swin_tiny']}
assert len(records)==4 and all(r['state']=='verified' and r['final']['train_monitor_accuracy']==1.0 and r['final']['train_monitor_loss']<0.1 and len(r['sample_ids'])==len(set(r['sample_ids']))==20 for r in records)
atomic_json(ROOT/'outputs/checks/completed.json',{'passed':True,'core':core,'smoke':smoke,'overfit_artifacts':[str(p) for p in overfit]})
PY
