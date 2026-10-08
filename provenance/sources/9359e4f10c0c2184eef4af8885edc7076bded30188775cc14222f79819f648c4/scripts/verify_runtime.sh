#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/test_visualize.py tests/test_benchmark.py tests/test_frontend_gpu.py --junitxml=outputs/checks/runtime-tests.xml
python scripts/preprocessing_pilot.py --device cuda
python smoke.py --device cuda
python scripts/check_tensorboard_server.py
python - <<'PY'
from sparse_contrast.common import atomic_json,ROOT
atomic_json(ROOT/'outputs/checks/runtime-completed.json',{'passed':True})
PY
