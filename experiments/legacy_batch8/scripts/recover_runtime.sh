#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/test_benchmark.py --junitxml=outputs/checks/benchmark-repair-tests.xml
python - <<'PY'
import json,os,subprocess,sys
from sparse_contrast.campaign import source_snapshot,PYTHON
from sparse_contrast.common import ROOT,atomic_json
source,directory=source_snapshot()
subprocess.run([PYTHON,str(directory/'smoke.py'),'--device','cuda'],check=True,cwd=directory)
subprocess.run([PYTHON,str(directory/'scripts/check_tensorboard_server.py')],check=True,cwd=directory)
atomic_json(ROOT/'outputs/checks/runtime-completed.json',{'passed':True,'source_hash':source,'supersedes_jobs':['353360','353364'],'previous_failure':'source mutation and bare GPU UUID identified and fixed; prior artifacts preserved'})
PY
