#!/usr/bin/env bash
set -euo pipefail
export PATH=/ceph/sagnihot/miniconda3/envs/sparse-contrast-bench/bin:$PATH
export SPARSE_BENCH_TEST_DEVICE=cuda
python -m pytest tests/test_frontend.py tests/test_normalization.py tests/test_data.py tests/test_models.py tests/test_training.py tests/test_metrics.py --junitxml=outputs/checks/core-tests.xml
python compute_normalization_stats.py --data-root /ceph/sagnihot/datasets --device cpu --threads 4 --output-dir normalization
python - <<'PY'
from sparse_contrast.common import atomic_json,ROOT,hardware
atomic_json(ROOT/'outputs/checks/core-completed.json',{'passed':True,'hardware':hardware(),'scope':'core tests and complete frozen normalization; smoke/overfit gate separate'})
PY
